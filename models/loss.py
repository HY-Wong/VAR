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


class Loss(nn.Module):
    def __init__(self, args, steps_per_epoch):
        super().__init__()
        self.args = args
        assert args.disc_loss_fn in ['hinge', 'cross_entropy']

        # reconstruction loss
        if args.rec_loss_fn == 'l1':
            self.rec_loss_fn = nn.L1Loss(reduction='mean')
        elif args.rec_loss_fn == 'l2':
            self.rec_loss_fn = nn.MSELoss(reduction='mean') 
        # perceptual loss
        self.lpips = LPIPS('vgg16_lpips.pth').eval()
        # adversarial loss
        if args.disc_loss_fn == 'hinge':
            self.discriminator = DiscriminatorHL(
                in_channels=args.in_channels, out_channels=args.out_channels, n_layers=args.n_layers
            ).apply(weights_init)
            self.disc_loss_fn = hinge_d_loss
        elif args.disc_loss_fn == 'cross_entropy':
            self.discriminator = DiscriminatorCEL(
                in_channels=args.in_channels, out_channels=args.out_channels, n_layers=args.n_layers
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
            self, l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll, inp, rec_inp, 
            vq_loss, optimizer_idx, last_layer, global_step, split
        ):
        # VAE
        if optimizer_idx == 0:
            rec_loss = self.rec_loss_fn(rec_l1_hs, l1_hs) + self.rec_loss_fn(rec_l2_hs, l2_hs) + self.rec_loss_fn(rec_ll, ll)
            perc_loss = 0. # torch.mean(self.lpips(ll, rec_ll))

            logits_fake = self.discriminator(rec_inp)
            if self.args.disc_loss_fn == 'hinge':
                disc_loss = -torch.mean(logits_fake)
            elif self.args.disc_loss_fn == 'cross_entropy':
                labels_real = torch.ones(logits_fake.shape[0], dtype=torch.float, device=logits_fake.device).unsqueeze(1)
                disc_loss = F.binary_cross_entropy_with_logits(logits_fake, labels_real)
            
            try:
                weight = self.calculate_adaptive_weight(rec_loss, disc_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                weight = 1.0
            if self.disc_start_step > global_step:
                disc_loss *= 0.0
            print(f'[INFO] {self.disc_start_step} > {global_step}')
                
            # compound loss
            vae_loss = rec_loss + self.args.lc * vq_loss + self.args.lp * perc_loss + weight * self.args.ld * disc_loss
            vae_log_dict = {
                f'{split}_vae_loss': vae_loss.clone().detach(),
                f'{split}_vae_rec_loss': rec_loss.detach(),
                f'{split}_vae_vq_loss': self.args.lc * vq_loss.detach(),
                # f'{split}_vae_perc_loss': self.args.lp * perc_loss.detach(),
                f'{split}_vae_disc_loss': weight * self.args.ld * disc_loss.detach(),
                f'{split}_vae_disc_weight': weight
            }
            return vae_loss, vae_log_dict

        # Discriminator
        if optimizer_idx == 1:
            logits_real = self.discriminator(inp.detach())
            logits_fake = self.discriminator(rec_inp.detach())
            disc_loss = self.disc_loss_fn(logits_real, logits_fake)
            
            if self.disc_start_step > global_step:
                disc_loss *= 0.0

            disc_log_dict = {
                f'{split}_disc_loss': disc_loss.clone().detach(),
                f'{split}_logits_real': logits_real.detach().mean(),
                f'{split}_logits_fake': logits_fake.detach().mean()
            }
            return disc_loss, disc_log_dict