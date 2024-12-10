import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import numpy as np
import pywt
import wandb

from torch.nn.utils import clip_grad_norm_
from typing import List
from PIL import Image


class VQVAE_WAV_Trainer(pl.LightningModule):
    def __init__(self, vae, lpips, args, steps_per_epoch):
        super().__init__()
        self.model = vae
        self.lpips = lpips
        self.args = args
        self.steps_per_epoch = steps_per_epoch

        # reconstruction loss function
        if args.loss_fn == 'l1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif args.loss_fn == 'l2':
            self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, l1_hs, l2_hs, ll):
        return self.model(l1_hs, l2_hs, ll)
    
    def training_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        rec_l1_hs, rec_l2_hs, rec_ll, usages, vq_loss = self(l1_hs, l2_hs, ll)
        
        # compound loss
        rec_loss = self.loss_fn(rec_l1_hs, l1_hs) + self.loss_fn(rec_l2_hs, l2_hs) + self.loss_fn(rec_ll, ll)
        '''
        lpips_low_loss = self.lpips(rec_ll, ll)
        lpips_loss = self.args.lp_low * torch.mean(lpips_low_loss)

        if self.args.lp_high > 0:
            lpips_high_loss = 0

            # level 1
            lpips_high_loss += self.lpips(rec_l1_hs[:, :3], l1_hs[:, :3])   # LH
            lpips_high_loss += self.lpips(rec_l1_hs[:, 3:6], l1_hs[:, 3:6]) # HL
            lpips_high_loss += self.lpips(rec_l1_hs[:, 6:], l1_hs[:, 6:])   # HH
            # level 2
            lpips_high_loss += self.lpips(rec_l2_hs[:, :3], l2_hs[:, :3])   # LH
            lpips_high_loss += self.lpips(rec_l2_hs[:, 3:6], l2_hs[:, 3:6]) # HL
            lpips_high_loss += self.lpips(rec_l2_hs[:, 6:], l2_hs[:, 6:])   # HH
            
            lpips_loss += self.args.lp_high * torch.mean(lpips_high_loss)
        '''
        lpips_loss = 0
        loss = rec_loss + self.args.lc * vq_loss + lpips_loss
        
        # log
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_rec_loss', rec_loss, sync_dist=True)
        self.log('train_vq_loss', self.args.lc * vq_loss, sync_dist=True)
        self.log('train_lpips_loss', lpips_loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        rec_l1_hs, rec_l2_hs, rec_ll, usages, vq_loss = self(l1_hs, l2_hs, ll)
        
        # compound loss
        rec_loss = self.loss_fn(rec_l1_hs, l1_hs) + self.loss_fn(rec_l2_hs, l2_hs) + self.loss_fn(rec_ll, ll)
        '''
        lpips_low_loss = self.lpips(rec_ll, ll)
        lpips_loss = self.args.lp_low * torch.mean(lpips_low_loss)

        if self.args.lp_high > 0:
            lpips_high_loss = 0

            # level 1
            lpips_high_loss += self.lpips(rec_l1_hs[:, :3], l1_hs[:, :3])   # LH
            lpips_high_loss += self.lpips(rec_l1_hs[:, 3:6], l1_hs[:, 3:6]) # HL
            lpips_high_loss += self.lpips(rec_l1_hs[:, 6:], l1_hs[:, 6:])   # HH
            # level 2
            lpips_high_loss += self.lpips(rec_l2_hs[:, :3], l2_hs[:, :3])   # LH
            lpips_high_loss += self.lpips(rec_l2_hs[:, 3:6], l2_hs[:, 3:6]) # HL
            lpips_high_loss += self.lpips(rec_l2_hs[:, 6:], l2_hs[:, 6:])   # HH
            
            lpips_loss += self.args.lp_high * torch.mean(lpips_high_loss)
        '''
        lpips_loss = 0
        loss = rec_loss + self.args.lc * vq_loss + lpips_loss
        
        # log
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_rec_loss', rec_loss, sync_dist=True)
        self.log('val_vq_loss', self.args.lc * vq_loss, sync_dist=True)
        self.log('val_lpips_loss', lpips_loss, sync_dist=True)

        if batch_idx == 0: # only plot one batch
            self.coeffs = (l1_hs, l2_hs, ll)
            self.rec_coeffs = (rec_l1_hs, rec_l2_hs, rec_ll)

    def on_validation_epoch_end(self):
        l1_hs, l2_hs, ll = self.coeffs
        rec_l1_hs, rec_l2_hs, rec_ll = self.rec_coeffs

        imgs = self.get_images_from_wavelet(l1_hs, l2_hs, ll)
        rec_imgs = self.get_images_from_wavelet(rec_l1_hs, rec_l2_hs, rec_ll)
        self.logger.experiment.log({
            'val_image/orig': wandb.Image(imgs),
            'val_image/recon': wandb.Image(rec_imgs)
        })
    
    def test_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        rec_l1_hs, rec_l2_hs, rec_ll, usages, vq_loss = self(l1_hs, l2_hs, ll)
        
        # compound loss
        rec_loss = self.loss_fn(rec_l1_hs, l1_hs) + self.loss_fn(rec_l2_hs, l2_hs) + self.loss_fn(rec_ll, ll)
        '''
        lpips_low_loss = self.lpips(rec_ll, ll)
        lpips_loss = self.args.lp_low * torch.mean(lpips_low_loss)

        if self.args.lp_high > 0:
            lpips_high_loss = 0

            # level 1
            lpips_high_loss += self.lpips(rec_l1_hs[:, :3], l1_hs[:, :3])   # LH
            lpips_high_loss += self.lpips(rec_l1_hs[:, 3:6], l1_hs[:, 3:6]) # HL
            lpips_high_loss += self.lpips(rec_l1_hs[:, 6:], l1_hs[:, 6:])   # HH
            # level 2
            lpips_high_loss += self.lpips(rec_l2_hs[:, :3], l2_hs[:, :3])   # LH
            lpips_high_loss += self.lpips(rec_l2_hs[:, 3:6], l2_hs[:, 3:6]) # HL
            lpips_high_loss += self.lpips(rec_l2_hs[:, 6:], l2_hs[:, 6:])   # HH
            
            lpips_loss += self.args.lp_high * torch.mean(lpips_high_loss)
        '''
        lpips_loss = 0
        loss = rec_loss + self.args.lc * vq_loss + lpips_loss
        
        # log
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_rec_loss', rec_loss, sync_dist=True)
        self.log('test_vq_loss', self.args.lc * vq_loss, sync_dist=True)
        self.log('test_lpips_loss', lpips_loss, sync_dist=True)

        # plot five batches
        imgs = self.get_images_from_wavelet(l1_hs, l2_hs, ll)
        rec_imgs = self.get_images_from_wavelet(rec_l1_hs, rec_l2_hs, rec_ll)
        self.logger.experiment.log({
            'test_image/orig': wandb.Image(imgs),
            'test_image/recon': wandb.Image(rec_imgs)
        })
    
    def configure_optimizers(self):
        # optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        beta1, beta2 = map(float, self.args.opt_beta.split('_'))
        optimizer = torch.optim.AdamW(
            params, lr=self.args.vae_lr, betas=(beta1, beta2), fused=self.args.opt_fuse
        )

        # lr scheduler
        total_steps = self.args.ep * self.steps_per_epoch
        warmup_steps = self.args.vae_wp_ep * self.steps_per_epoch

        def lr_lambda(current_step):
            """
            Decay the learning rate with half-cycle cosine after warmup
            """
            if current_step < warmup_steps:
                # linear warmup
                lr_ratio = self.args.vae_wp_lr
                return lr_ratio + (1 - lr_ratio) * current_step / warmup_steps
            else:
                # cosine annealing decay
                decay_steps = total_steps - warmup_steps
                progress = (current_step - warmup_steps) / decay_steps
                lr_ratio = self.args.vae_final_lr
                return lr_ratio + (1 - lr_ratio) * 0.5 * (1 + np.cos(progress * np.pi))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        optimizer_dict = {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        return optimizer_dict
    
    def on_after_backward(self):
        # specify a very high value for max_norm to avoid actual clipping if undesired
        max_norm = self.args.max_norm
        total_norm = clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        self.log('gradients_norm', total_norm)

    def get_images_from_wavelet(
        self, l1_hs: torch.Tensor, l2_hs: torch.Tensor, ll: torch.Tensor
    ) -> Image.Image:
        """
        Reconstructs and visualizes images from multi-level wavelet components.
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
            img = img * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1) # denormalize
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
        Reconstruct a 2D signal channel-wise using multi-level wavelet coefficients.
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