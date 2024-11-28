import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision
import numpy as np
import pywt
import wandb

from torch.nn.utils import clip_grad_norm_
from PIL import Image


class VQVAE_WAV_Trainer(pl.LightningModule):
    def __init__(self, vae, args):
        super().__init__()
        self.model = vae
        self.args = args
        self.l2_loss = nn.MSELoss(reduction='mean')
        # mean and std of Imagenet
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def forward(self, l1_hs, l2_hs, ll):
        return self.model(l1_hs, l2_hs, ll)
    
    def training_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        rec_l1_hs, rec_l2_hs, rec_ll, usages, vq_loss = self(l1_hs, l2_hs, ll)
        
        # compound loss
        rec_loss = self.l2_loss(rec_l1_hs, l1_hs) + self.l2_loss(rec_l2_hs, l2_hs) + self.l2_loss(rec_ll, ll)
        loss = rec_loss + self.args.beta * vq_loss
        
        # log
        self.log('train_loss', loss, sync_dist=True)
        self.log('train_rec_loss', rec_loss, sync_dist=True)
        self.log('train_vq_loss', self.args.beta * vq_loss, sync_dist=True)

        # print(f"[RESULT] Train loss: {loss.item():.4f}")
        # print(f"[RESULT] Reconstruction loss: {rec_loss.item():.4f}")
        # print(f"[RESULT] Commitment loss: {vq_loss.item():.4f}")
        return loss
    
    def validation_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        rec_l1_hs, rec_l2_hs, rec_ll, usages, vq_loss = self(l1_hs, l2_hs, ll)
        
        # compound loss
        rec_loss = self.l2_loss(rec_l1_hs, l1_hs) + self.l2_loss(rec_l2_hs, l2_hs) + self.l2_loss(rec_ll, ll)
        loss = self.args.beta * rec_loss + vq_loss
        
        # log
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_rec_loss', rec_loss, sync_dist=True)
        self.log('val_vq_loss', vq_loss, sync_dist=True)

        if batch_idx == 0: # only plot one batch
            self.coeffs = (l1_hs, l2_hs, ll)
            self.rec_coeffs = (rec_l1_hs, rec_l2_hs, rec_ll)

    def on_validation_epoch_end(self):
        l1_hs, l2_hs, ll = self.coeffs
        rec_l1_hs, rec_l2_hs, rec_ll = self.rec_coeffs

        imgs, rec_imgs = self.get_images_from_wavelet(l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll)
        self.logger.experiment.log({
            'val_image/orig': wandb.Image(imgs),
            'val_image/recon': wandb.Image(rec_imgs)
        })
    
    def test_step(self, batch, batch_idx):
        (l1_hs, l2_hs, ll), label = batch
        rec_l1_hs, rec_l2_hs, rec_ll, usages, vq_loss = self(l1_hs, l2_hs, ll)
        
        # compound loss
        rec_loss = self.l2_loss(rec_l1_hs, l1_hs) + self.l2_loss(rec_l2_hs, l2_hs) + self.l2_loss(rec_ll, ll)
        loss = self.args.beta * rec_loss + vq_loss
        
        # log
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_rec_loss', rec_loss, sync_dist=True)
        self.log('test_vq_loss', vq_loss, sync_dist=True)

        # plot five batches
        if batch_idx < 10:
            imgs, rec_imgs = self.get_images_from_wavelet(l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll)
            self.logger.experiment.log({
                'test_image/orig': wandb.Image(imgs),
                'test_image/recon': wandb.Image(rec_imgs)
            })
    
    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.args.tlr)
        return optimizer
    
    def on_after_backward(self):
        # specify a very high value for max_norm to avoid actual clipping if undesired
        max_norm = 1e6
        total_norm = clip_grad_norm_(self.model.parameters(), max_norm=max_norm)
        self.log('gradients_norm', total_norm)

    def get_images_from_wavelet(self, l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll):
        """
        """
        imgs = []
        rec_imgs = []

        for i in range(ll.shape[0]):
            # lowest resolution to highest resolution
            highs = [
                l2_hs[i].view(3, 3, l2_hs.shape[2], l2_hs.shape[3]).cpu().numpy(),
                l1_hs[i].view(3, 3, l1_hs.shape[2], l1_hs.shape[3]).cpu().numpy()
            ]
            low = ll[i].cpu().numpy()
            
            img = self.reconstruct_multilevel_2d(highs, low, 'haar')
            img = img * self.std.reshape(3, 1, 1) + self.mean.reshape(3, 1, 1)
            img = np.clip(img, 0, 1)
            imgs.append(img)
            
            rec_highs = [
                rec_l2_hs[i].view(3, 3, rec_l2_hs.shape[2], rec_l2_hs.shape[3]).cpu().numpy(),
                rec_l1_hs[i].view(3, 3, rec_l1_hs.shape[2], rec_l1_hs.shape[3]).cpu().numpy()
            ]
            rec_low = rec_ll[i].cpu().numpy()
            
            rec_img = self.reconstruct_multilevel_2d(rec_highs, rec_low, 'haar')
            rec_img = rec_img * self.std.reshape(3, 1, 1) + self.mean.reshape(3, 1, 1)
            rec_img = np.clip(rec_img, 0, 1)
            rec_imgs.append(rec_img)

        imgs = torch.tensor(np.stack(imgs))
        imgs = torchvision.utils.make_grid(imgs, nrow=8, padding=0)
        imgs = imgs.permute(1, 2, 0).mul_(255).numpy()
        imgs = Image.fromarray(imgs.astype(np.uint8))

        rec_imgs = torch.tensor(np.stack(rec_imgs))
        rec_imgs = torchvision.utils.make_grid(rec_imgs, nrow=8, padding=0)
        rec_imgs = rec_imgs.permute(1, 2, 0).mul_(255).numpy()
        rec_imgs = Image.fromarray(rec_imgs.astype(np.uint8))
        return imgs, rec_imgs

    def reconstruct_multilevel_2d(self, highs, low, wavelet, mode='periodization'):
        """
        Perform 2D wavelet reconstruction.
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