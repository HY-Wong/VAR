# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
# https://github.com/zacheberhart/Maximum-Mean-Discrepancy-Variational-Autoencoder/blob/master/MMD-VAE%20(InfoVAE).ipynb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator import DiscriminatorHL, DiscriminatorCEL, weights_init
from .lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def cross_entropy_d_loss(logits_real, logits_fake):
    labels_real = torch.ones(logits_real.shape[0], dtype=torch.float, device=logits_real.device).unsqueeze(1)
    labels_fake = torch.zeros(logits_fake.shape[0], dtype=torch.float, device=logits_fake.device).unsqueeze(1)
    loss_real = F.binary_cross_entropy_with_logits(logits_real, labels_real)
    loss_fake = F.binary_cross_entropy_with_logits(logits_fake, labels_fake)
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def focal_freq_loss(inp, rec_inp, alpha=1):
    diff = (inp - rec_inp) ** 2
    weight = diff.detach().sqrt() ** alpha
    weight = torch.nan_to_num(weight, nan=0.0)
    weight = torch.clamp(weight, min=0.0, max=1.0)  # ensure weight is in range [0, 1]
    loss = weight * diff
    return loss.mean()


def compute_kernel(x, y):
    dim = x.size(1)
    kernel_input = ((x[:, None, :] - y[None, :, :]) ** 2).mean(dim=2) / dim
    return torch.exp(-kernel_input) # (x_size, y_size)


def mmd_loss(f):
    f = f.view(f.shape[0], -1)
    true_f = torch.randn_like(f, requires_grad=False) # match size and device
    x_kernel = compute_kernel(true_f, true_f)
    y_kernel = compute_kernel(f, f)
    xy_kernel = compute_kernel(true_f, f)
    loss = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return loss


class Loss(nn.Module):
    def __init__(self, args, steps_per_epoch):
        super().__init__()
        self.args = args
        assert args.rec_loss_fn in ['l1', 'l2', 'focal'], '[ERROR] Invalid reconstruction loss function!'
        print(f'[INFO] Using reconstruction loss: {args.rec_loss_fn}')
        assert args.disc_loss_fn in ['hinge', 'cross_entropy'], '[ERROR] Invalid discriminator loss function!'
        print(f'[INFO] Using discriminator loss: {args.disc_loss_fn}')

        # reconstruction loss
        if args.rec_loss_fn == 'l1':
            self.rec_loss_fn = nn.L1Loss(reduction='mean')
        elif args.rec_loss_fn == 'l2':
            self.rec_loss_fn = nn.MSELoss(reduction='mean') 
        elif args.rec_loss_fn == 'focal':
            self.rec_loss_fn = focal_freq_loss
        # perceptual loss
        self.lpips = LPIPS('vgg16_lpips.pth').eval()
        # maximum-mean-discrepancy loss
        self.mmd_loss_fn = mmd_loss
        # adversarial loss
        if args.disc_loss_fn == 'hinge':
            self.discriminator = DiscriminatorHL(
                in_channels=3, out_channels=args.out_channels, n_layers=args.n_layers
            ).apply(weights_init)
            self.disc_loss_fn = hinge_d_loss
        elif args.disc_loss_fn == 'cross_entropy':
            self.discriminator = DiscriminatorCEL(
                in_channels=3, out_channels=args.out_channels, n_layers=args.n_layers
            ).apply(weights_init)
            self.disc_loss_fn = cross_entropy_d_loss
       
        self.disc_start_step = args.disc_start_ep * steps_per_epoch

    def calculate_adaptive_weight(self, rec_loss, disc_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        disc_grads = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]

        weight = torch.norm(rec_grads) / (torch.norm(disc_grads) + 1e-4)
        weight = torch.clamp(weight, 0.0, 1e4).detach()
        return weight

    def forward(
            self, imgs, h1, h2, ll2, rec_imgs, rec_h1, rec_h2, rec_ll2,
            vq_loss, optimizer_idx, last_layer, global_step, split
        ):
        # VAE
        if optimizer_idx == 0:
            rec_loss = self.rec_loss_fn(h1, rec_h1) + self.rec_loss_fn(h2, rec_h2) + self.rec_loss_fn(ll2, rec_ll2)
            # perc_loss = torch.mean(self.lpips(imgs, rec_imgs))
            # up_ll2 = F.interpolate(ll2, size=(256, 256), mode='bilinear', align_corners=False)
            # up_rec_ll2 = F.interpolate(rec_ll2, size=(256, 256), mode='bilinear', align_corners=False)
            # perc_loss = torch.mean(self.lpips(up_ll2, up_rec_ll2))
            
            # logits_fake = self.discriminator(rec_imgs)
            # if self.args.disc_loss_fn == 'hinge':
            #     disc_loss = -torch.mean(logits_fake)
            # elif self.args.disc_loss_fn == 'cross_entropy':
            #     labels_real = torch.ones(logits_fake.shape[0], dtype=torch.float, device=logits_fake.device).unsqueeze(1)
            #     disc_loss = F.binary_cross_entropy_with_logits(logits_fake, labels_real)
            
            # try:
            #     weight = self.calculate_adaptive_weight(rec_loss, disc_loss, last_layer=last_layer)
            # except RuntimeError:
            #     assert not self.training
            #     weight = 1.0
            # if self.disc_start_step > global_step:
            #     disc_loss *= 0.0
            
            # compound loss
            vae_loss = rec_loss + self.args.lc * vq_loss # + self.args.lp * perc_loss # + weight * self.args.ld * disc_loss
            
            vae_log_dict = {
                f'{split}_vae_loss': vae_loss.clone().detach(),
                f'{split}_vae_rec_loss': rec_loss.detach(),
                f'{split}_vae_vq_loss': self.args.lc * vq_loss.detach(),
                # f'{split}_vae_perc_loss': self.args.lp * perc_loss.detach(),
                # f'{split}_vae_disc_loss': self.args.ld * disc_loss.detach(),
                # f'{split}_vae_disc_weight': weight
            }
            return vae_loss, vae_log_dict

        # Discriminator
        if optimizer_idx == 1:
            logits_real = self.discriminator(imgs.detach())
            logits_fake = self.discriminator(rec_imgs.detach())
            disc_loss = self.disc_loss_fn(logits_real, logits_fake)
            
            if self.disc_start_step > global_step:
                disc_loss *= 0.0

            disc_log_dict = {
                f'{split}_disc_loss': disc_loss.clone().detach(),
                f'{split}_logits_real': logits_real.detach().mean(),
                f'{split}_logits_fake': logits_fake.detach().mean()
            }
            return disc_loss, disc_log_dict