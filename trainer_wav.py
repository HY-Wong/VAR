import torch
import pytorch_lightning as pl
import torchvision
import numpy as np
import wandb

from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse

from models.loss import Loss

class VQVAE_WAV_Trainer(pl.LightningModule):
    def __init__(self, vae, args, steps_per_epoch):
        super().__init__()
        self.vae = vae
        self.loss = Loss(args, steps_per_epoch)
        # 2-level wavelet decomposition
        self.dwt = DWTForward(J=2, wave=args.wavelet, mode='zero')
        self.idwt = DWTInverse(wave=args.wavelet, mode='zero')
        self.args = args
        self.steps_per_epoch = steps_per_epoch
         
        # activates manual optimization for multiple optimizers
        self.automatic_optimization = False
    
    def forward(self, imgs):
        yl, (yh1, yh2) = self.dwt(imgs)
        yh1 = yh1.view(yh1.shape[0], -1, yh1.shape[3], yh1.shape[4]) # -> (N, C * 3, H, W)
        yh2 = yh2.view(yh2.shape[0], -1, yh2.shape[3], yh2.shape[4]) # -> (N, C * 3, H, W)
        yh1_norm, yh2_norm, yl_norm = yh1 / 2, yh2 / 4,  yl / 4 # normalization

        rec_yh1_norm, rec_yh2_norm, rec_yl_norm, yh1_, rec_yh1_, usages, f, vq_loss = self.vae(yh1_norm, yh2_norm, yl_norm)
    
        rec_yh1, rec_yh2, rec_yl = rec_yh1_norm * 2, rec_yh2_norm * 4,  rec_yl_norm * 4 # denormalization
        rec_yh1 = rec_yh1.view(rec_yh1.shape[0], 3, 3, rec_yh1.shape[2], rec_yh1.shape[3]) # -> (N, C, 3, H, W)
        rec_yh2 = rec_yh2.view(rec_yh2.shape[0], 3, 3, rec_yh2.shape[2], rec_yh2.shape[3]) # -> (N, C, 3, H, W)
        rec_imgs = self.idwt((rec_yl, [rec_yh1, rec_yh2]))
        return yh1_norm, yh2_norm, yl_norm, yh1_, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_, rec_imgs, usages, f, vq_loss

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        yh1_norm, yh2_norm, yl_norm, yh1_, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_, rec_imgs, _, f, vq_loss = self(imgs)

        vae_opt, disc_opt = self.optimizers()
        
        # optimize VAE
        # adjust global step to match LR scheduler step: two optimizers -> two steps per iteration
        vae_loss, vae_log_dict = self.loss(
            imgs, yh1_norm, yh2_norm, yl_norm, yh1_, rec_imgs, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_,
            f, vq_loss, 0, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'train'
        )
        
        vae_opt.zero_grad()
        self.manual_backward(vae_loss)
        # clip gradients
        self.clip_gradients(vae_opt, gradient_clip_val=self.args.max_norm, gradient_clip_algorithm='norm')
        vae_opt.step()
        
        # optimize Discriminator
        disc_loss, disc_log_dict = self.loss(
            imgs, yh1_norm, yh2_norm, yl_norm, yh1_, rec_imgs, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_,
            f, vq_loss, 1, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'train'
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
        for key, value in vae_log_dict.items():
            self.log(key, value, sync_dist=True)
        for key, value in disc_log_dict.items():
            self.log(key, value, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        yh1_norm, yh2_norm, yl_norm, yh1_, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_, rec_imgs, _, f, vq_loss = self(imgs)
        
        # VAE
        vae_loss, vae_log_dict = self.loss(
            imgs, yh1_norm, yh2_norm, yl_norm, yh1_, rec_imgs, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_,
            f, vq_loss, 0, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'val'
        )
        
        # Discriminator
        disc_loss, disc_log_dict = self.loss(
            imgs, yh1_norm, yh2_norm, yl_norm, yh1_, rec_imgs, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_,
            f, vq_loss, 1, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'val'
        )
        
        # log
        for key, value in vae_log_dict.items():
            self.log(key, value, sync_dist=True)      
        for key, value in disc_log_dict.items():
            self.log(key, value, sync_dist=True)
        
        # only plot one batch
        if batch_idx == 0:
            self.imgs = imgs
            self.rec_imgs = rec_imgs

    def on_validation_epoch_end(self):
        # reference images
        if self.current_epoch == 0:
            imgs = self.get_image_grid(self.imgs)
            self.logger.experiment.log({'val_image/orig': wandb.Image(imgs)})
        # reconstructed images
        if self.current_epoch % 10 == 0:
            rec_imgs = self.get_image_grid(self.rec_imgs)
            self.logger.experiment.log({'val_image/recon': wandb.Image(rec_imgs)})
    
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        yh1_norm, yh2_norm, yl_norm, yh1_, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_, rec_imgs, _, f, vq_loss = self(imgs)
        
        # VAE
        vae_loss, vae_log_dict = self.loss(
            imgs, yh1_norm, yh2_norm, yl_norm, yh1_, rec_imgs, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_,
            f, vq_loss, 0, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'test'
        )

        # Discriminator
        disc_loss, disc_log_dict = self.loss(
            imgs, yh1_norm, yh2_norm, yl_norm, yh1_, rec_imgs, rec_yh1_norm, rec_yh2_norm, rec_yl_norm, rec_yh1_,
            f, vq_loss, 1, self.vae.decoder.conv_out.weight, self.global_step//2+1, 'test'
        ) 
        
        # log
        for key, value in vae_log_dict.items():
            self.log(key, value, sync_dist=True)      
        for key, value in disc_log_dict.items():
            self.log(key, value, sync_dist=True)

        # plot five batches
        if batch_idx < 2:
            # denormalization
            imgs = self.get_image_grid(imgs)
            rec_imgs = self.get_image_grid(rec_imgs)
            self.logger.experiment.log({
                'test_image/orig': wandb.Image(imgs),
                'test_image/recon': wandb.Image(rec_imgs)
            })
    
    def configure_optimizers(self):
        beta1, beta2 = map(float, self.args.opt_beta.split('_'))
        total_steps = self.args.ep * self.steps_per_epoch

        def create_optimizer_and_scheduler(params, lr, start_ep, wp_ep, wp_lr, final_lr, logging_name):
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
                    return 0.0
                elif current_step < start_steps + wp_steps:
                    # linear warmup
                    return wp_lr + (1 - wp_lr) * (current_step - start_steps) / wp_steps
                else:
                    # cosine annealing decay
                    decay_steps = total_steps - (start_steps + wp_steps)
                    progress = (current_step - (start_steps + wp_steps)) / decay_steps
                    return final_lr + (1 - final_lr) * 0.5 * (1 + np.cos(progress * np.pi))

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda),
                'name': logging_name
            }
            return optimizer, scheduler

        # VAE optimizer and scheduler
        params1 = [p for p in self.vae.parameters() if p.requires_grad]
        optimizer1, scheduler1 = create_optimizer_and_scheduler(
            params=params1, lr=self.args.vae_lr, start_ep=0, wp_ep=self.args.vae_wp_ep, 
            wp_lr=self.args.vae_wp_lr, final_lr=self.args.vae_final_lr, logging_name='lr-vae'
        )

        # Discriminator optimizer and scheduler
        params2 = [p for p in self.loss.discriminator.parameters() if p.requires_grad]
        optimizer2, scheduler2 = create_optimizer_and_scheduler(
            params=params2, lr=self.args.disc_lr, start_ep=self.args.disc_start_ep, wp_ep=self.args.disc_wp_ep,
            wp_lr=self.args.disc_wp_lr, final_lr=self.args.disc_final_lr, logging_name='lr-disc'
        )

        return [optimizer1, optimizer2], [scheduler1, scheduler2]

    def get_image_grid(self, imgs: torch.Tensor, nrow: int = 8, padding: int = 0) -> Image.Image:
        """
        Get a grid of images suitable for saving or visualization.
        """
        imgs = torch.clamp((imgs.detach() + 1) / 2, min=0, max=1)  # normalize to [0, 1]
        imgs = torchvision.utils.make_grid(imgs, nrow=nrow, padding=padding)
        imgs = imgs.permute(1, 2, 0).mul_(255).cpu().numpy()
        return Image.fromarray(imgs.astype(np.uint8))