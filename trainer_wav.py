import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import numpy as np
import pywt
import wandb

from typing import List
from PIL import Image

from models.discriminator import Discriminator, weights_init
from models.lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


class VQVAE_WAV_Trainer(pl.LightningModule):
    def __init__(self, vae, args, steps_per_epoch):
        super().__init__()
        self.vae = vae
        self.lpips = LPIPS('vgg16_lpips.pth').eval()
        self.discriminator = Discriminator(
            in_channels=args.in_channels, out_channels=args.out_channels, n_layers=args.n_layers
        ).apply(weights_init)
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.disc_start_step = args.disc_start_ep * steps_per_epoch

        # reconstruction loss function
        if args.loss_fn == 'l1':
            self.rec_loss_fn = nn.L1Loss(reduction='mean')
        elif args.loss_fn == 'l2':
            self.rec_loss_fn = nn.MSELoss(reduction='mean')
        # discriminator loss function
        self.disc_loss_fn = hinge_d_loss     

        # activates manual optimization for multiple optimizers
        self.automatic_optimization = False
    
    def calculate_adaptive_weight(self, rec_loss, disc_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        disc_grads = torch.autograd.grad(disc_loss, last_layer, retain_graph=True)[0]

        ld = torch.norm(rec_grads) / (torch.norm(disc_grads) + 1e-4)
        ld = torch.clamp(ld, 0.0, 1e4).detach()
        ld = ld * self.args.ld
        return ld
    
    def get_vae_loss(
        self, l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll, inp, rec_inp, 
        vq_loss, last_layer, global_step, split
    ):
        rec_loss = self.rec_loss_fn(rec_l1_hs, l1_hs) + self.rec_loss_fn(rec_l2_hs, l2_hs) + self.rec_loss_fn(rec_ll, ll)
        perc_loss = torch.mean(self.lpips(ll, rec_ll))

        logits_fake = self.discriminator(rec_inp)
        disc_loss = torch.mean(F.relu(1. - logits_fake))
        # ld = self.calculate_adaptive_weight(rec_loss, disc_loss, last_layer)
        ld = self.args.ld
        if self.disc_start_step > global_step:
            disc_loss *= 0.0

        # compound loss
        # vae_loss = rec_loss + self.args.lc * vq_loss + ld * disc_loss
        vae_loss = rec_loss + self.args.lc * vq_loss + self.args.lp * perc_loss
        # vae_loss = rec_loss + self.args.lc * vq_loss + self.args.lp * perc_loss + ld * disc_loss

        vae_log_dict = {
            f'{split}_vae_loss': vae_loss,
            f'{split}_vae_rec_loss': rec_loss,
            f'{split}_vae_vq_loss': self.args.lc * vq_loss,
            f'{split}_perc_loss': self.args.lp * perc_loss,
            # f'{split}_vae_disc_loss': ld * disc_loss
        }
        return vae_loss, vae_log_dict

    def get_disc_loss(self, inp, rec_inp, global_step, split):
        logits_real = self.discriminator(inp)
        logits_fake = self.discriminator(rec_inp)
        disc_loss = self.disc_loss_fn(logits_real, logits_fake)
        if self.disc_start_step > global_step:
            disc_loss *= 0.0

        disc_log_dict = {
            f'{split}_disc_loss': disc_loss,
            f'{split}_logits_real': logits_real.detach().mean(),
            f'{split}_logits_fake': logits_fake.detach().mean()
        }
        return disc_loss, disc_log_dict

    def training_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        l1_hs.requires_grad_()
        l2_hs.requires_grad_()
        ll.requires_grad_()
        # normalization
        l1_hs, l2_hs, ll = l1_hs / 2**1, l2_hs / 2**2,  ll / 2**2
        rec_l1_hs, rec_l2_hs, rec_ll, _, vq_loss, inp, rec_inp = self.vae(l1_hs, l2_hs, ll)

        vae_opt, disc_opt = self.optimizers()
        
        # optimize VAE
        # adjust global step to match LR scheduler step: two optimizers -> two steps per iteration
        vae_loss, vae_log_dict = self.get_vae_loss(
            l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll, inp, rec_inp,
            vq_loss, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'train'
        )
        vae_opt.zero_grad()
        self.manual_backward(vae_loss)
        # clip gradients
        self.clip_gradients(vae_opt, gradient_clip_val=self.args.max_norm, gradient_clip_algorithm='norm')
        vae_opt.step()
        
        # optimize Discriminator
        disc_loss, disc_log_dict = self.get_disc_loss(
            inp.detach(), rec_inp.detach(), self.global_step//2+1, 'train'
        )
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        # clip gradients
        self.clip_gradients(disc_opt, gradient_clip_val=self.args.max_norm, gradient_clip_algorithm='norm')
        disc_opt.step()
        
        # learning rate scheduling
        vae_sch, disc_sch = self.lr_schedulers()
        vae_sch.step()
        disc_sch.step()

        # log
        self.log_dict(vae_log_dict, sync_dist=True)
        self.log_dict(disc_log_dict, sync_dist=True)
    
    def validation_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        # normalization
        l1_hs, l2_hs, ll = l1_hs / 2**1, l2_hs / 2**2,  ll / 2**2
        rec_l1_hs, rec_l2_hs, rec_ll, _, vq_loss, inp, rec_inp = self.vae(l1_hs, l2_hs, ll)
        
        # VAE
        vae_loss, vae_log_dict = self.get_vae_loss(
            l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll, inp, rec_inp,
            vq_loss, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'val'
        )

        # Discriminator
        disc_loss, disc_log_dict = self.get_disc_loss(
            inp.detach(), rec_inp.detach(), self.global_step//2+1, 'val'
        )

        # log
        self.log_dict(vae_log_dict, sync_dist=True)
        self.log_dict(disc_log_dict, sync_dist=True)
        
        # only plot one batch
        if batch_idx == 0:
            # denormalization
            self.coeffs = (l1_hs * 2**1, l2_hs * 2**2, ll * 2**2)
            self.rec_coeffs = (rec_l1_hs * 2**1, rec_l2_hs * 2**2, rec_ll * 2**2)
            # self.coeffs = (l1_hs, l2_hs, ll)
            # self.rec_coeffs = (rec_l1_hs, rec_l2_hs, rec_ll)

    def on_validation_epoch_end(self):
        # reference images
        if self.current_epoch == 0:
            l1_hs, l2_hs, ll = self.coeffs
            imgs = self.get_images_from_wavelet(l1_hs, l2_hs, ll)
            self.logger.experiment.log({'val_image/orig': wandb.Image(imgs)})
        # reconstructed images
        if self.current_epoch % 10 == 0:
            rec_l1_hs, rec_l2_hs, rec_ll = self.rec_coeffs
            rec_imgs = self.get_images_from_wavelet(rec_l1_hs, rec_l2_hs, rec_ll)
            self.logger.experiment.log({'val_image/recon': wandb.Image(rec_imgs)})
    
    def test_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        # normalization
        l1_hs, l2_hs, ll = l1_hs / 2**1, l2_hs / 2**2,  ll / 2**2
        rec_l1_hs, rec_l2_hs, rec_ll, _, vq_loss, inp, rec_inp = self.vae(l1_hs, l2_hs, ll)
        
        # VAE
        vae_loss, vae_log_dict = self.get_vae_loss(
            l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll, inp, rec_inp,
            vq_loss, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'test'
        )
        
        # Discriminator
        disc_loss, disc_log_dict = self.get_disc_loss(
            inp.detach(), rec_inp.detach(), self.global_step//2+1, 'test'
        )

        # log
        self.log_dict(vae_log_dict, sync_dist=True)
        self.log_dict(disc_log_dict, sync_dist=True)

        # plot five batches
        if batch_idx < 5:
            # denormalization
            l1_hs, l2_hs, ll = l1_hs * 2**1, l2_hs * 2**2, ll * 2**2
            rec_l1_hs, rec_l2_hs, rec_ll = rec_l1_hs * 2**1, rec_l2_hs * 2**2, rec_ll * 2**2
            imgs = self.get_images_from_wavelet(l1_hs, l2_hs, ll)
            rec_imgs = self.get_images_from_wavelet(rec_l1_hs, rec_l2_hs, rec_ll)
            self.logger.experiment.log({
                'test_image/orig': wandb.Image(imgs),
                'test_image/recon': wandb.Image(rec_imgs)
            })
    
    def configure_optimizers(self):
        beta1, beta2 = map(float, self.args.opt_beta.split('_'))
        total_steps = self.args.ep * self.steps_per_epoch

        def create_optimizer_and_scheduler(params, lr, start_ep, wp_ep, wp_lr, final_lr):
            """
            Helper function to create optimizer and scheduler
            """
            optimizer = torch.optim.AdamW(
                params, lr=lr, betas=(beta1, beta2), fused=self.args.opt_fuse
            )
            start_steps = start_ep * self.steps_per_epoch
            wp_steps = wp_ep * self.steps_per_epoch

            def lr_lambda(current_step):
                """
                Decay the learning rate with half-cycle cosine after warmup
                """
                if current_step < start_steps:
                    # print(f'[INFO] LR {0.0:.2f}')
                    return 0.0
                elif current_step < start_steps + wp_steps:
                    # linear warmup
                    # print(f'[INFO] LR {wp_lr + (1 - wp_lr) * (current_step - start_steps) / wp_steps:.2f} -> {current_step - start_steps} / {wp_steps}')
                    return wp_lr + (1 - wp_lr) * (current_step - start_steps) / wp_steps
                else:
                    # cosine annealing decay
                    decay_steps = total_steps - (start_steps + wp_steps)
                    progress = (current_step - (start_steps + wp_steps)) / decay_steps
                    # print(f'[INFO] LR {final_lr + (1 - final_lr) * 0.5 * (1 + np.cos(progress * np.pi)):.2f} -> ({current_step} - {start_steps + wp_steps}) / ({total_steps} - {start_steps + wp_steps})')
                    return final_lr + (1 - final_lr) * 0.5 * (1 + np.cos(progress * np.pi))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return optimizer, scheduler

        # VAE optimizer and scheduler
        params1 = [p for p in self.vae.parameters() if p.requires_grad]
        optimizer1, scheduler1 = create_optimizer_and_scheduler(
            params=params1, lr=self.args.vae_lr, start_ep=0, 
            wp_ep=self.args.vae_wp_ep, wp_lr=self.args.vae_wp_lr, final_lr=self.args.vae_final_lr
        )

        # Discriminator optimizer and scheduler
        params2 = [p for p in self.discriminator.parameters() if p.requires_grad]
        optimizer2, scheduler2 = create_optimizer_and_scheduler(
            params=params2, lr=self.args.disc_lr, start_ep=self.args.disc_start_ep, 
            wp_ep=self.args.disc_wp_ep, wp_lr=self.args.disc_wp_lr, final_lr=self.args.disc_final_lr
        )

        return [optimizer1, optimizer2], [scheduler1, scheduler2]

    def get_images_from_wavelet(self, l1_hs: torch.Tensor, l2_hs: torch.Tensor, ll: torch.Tensor) -> Image.Image:
        """
        Reconstructs and visualizes images from multi-level wavelet components
        """
        imgs = []
        # mean and std of Imagenet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        for i in range(ll.shape[0]):
            # lowest resolution to highest resolution
            highs = [
                l2_hs[i].view(3, 3, l2_hs.shape[2], l2_hs.shape[3]).cpu().numpy(),
                l1_hs[i].view(3, 3, l1_hs.shape[2], l1_hs.shape[3]).cpu().numpy()
            ]
            low = ll[i].cpu().numpy()
            
            img = self.reconstruct_multilevel_2d(highs, low, 'haar')
            # img = img * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1) # denormalize
            img = (img + 1) / 2
            img = np.clip(img, 0, 1)
            imgs.append(img)

        imgs = torch.tensor(np.stack(imgs))
        imgs = torchvision.utils.make_grid(imgs, nrow=8, padding=0)
        imgs = imgs.permute(1, 2, 0).mul_(255).numpy()
        imgs = Image.fromarray(imgs.astype(np.uint8))
        return imgs

    def reconstruct_multilevel_2d(
        self, highs: List[np.ndarray], low: np.ndarray, wavelet: str, mode: str = 'periodization'
    ) -> np.ndarray:
        """
        Reconstruct a 2D signal channel-wise using multi-level wavelet coefficients
        """
        reconstructed_channels = []
        for c in range(low.shape[0]):
            current = low[c, ...]
            for high in highs:
                coeffs = (current, (high[0, c], high[1, c], high[2, c])) # (LH, HL, HH)
                current = pywt.idwt2(coeffs, wavelet=wavelet, mode=mode)
            reconstructed_channels.append(current)
        
        # stack the reconstructed channels to form the RGB image
        reconstructed_img = np.stack(reconstructed_channels, axis=0)
        return reconstructed_img