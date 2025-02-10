import torch
import torch.nn as nn
import torchvision
import numpy as np

from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
from tqdm import tqdm

from trainer_wav import VQVAE_WAV_Trainer
from models import build_vae
from utils import arg_util
from utils.data import ImageDataModule


def print_wavelet_info(name, tensor):
    print(f'{name} Shape: {tensor.shape}, Range: {tensor.min():.4f} to {tensor.max():.4f}, L1 Norm: {tensor.abs().sum():.4f}')


if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data loading
    image_data_module = ImageDataModule(args)
    image_test_loader = image_data_module.test_dataloader()
    dwt = DWTForward(J=2, wave=args.wavelet, mode='zero').to(device)
    idwt = DWTInverse(wave=args.wavelet, mode='zero').to(device)

    # build the model
    vae = build_vae(
        patch_nums=args.patch_nums,
        V=4096, Cvae=args.Cvae, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        init_vae=args.init_vae, init_vocab=args.init_vocab,
        ch_mult=args.ch_mult, in_channels=args.in_channels
    ).to(device)

    vae_ckpt = args.load_ckpt_path
    if vae_ckpt:
        model = VQVAE_WAV_Trainer.load_from_checkpoint(
            vae_ckpt, 
            vae=vae, 
            args=args, 
            steps_per_epoch=1
        )
    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f'[INFO] Number of trainable parameters {total_params}')
    
    # evaluate the reconstruction loss in the range of image normalization
    total_rec_loss = 0.0
    total_images = 0
    first_batch = True

    vae.eval()
    with torch.no_grad():
        for imgs, _ in tqdm(image_test_loader, desc='Processing images', leave=True):
            imgs = imgs.to(device)
            yl, (yh1, yh2) = dwt(imgs)
            yh1 = yh1.view(yh1.shape[0], -1, yh1.shape[3], yh1.shape[4]) # -> (N, C * 3, H, W)
            yh2 = yh2.view(yh2.shape[0], -1, yh2.shape[3], yh2.shape[4]) # -> (N, C * 3, H, W)
            yh1_norm, yh2_norm, yl_norm = yh1 / 2, yh2 / 4,  yl / 4 # normalization
            
            print_wavelet_info('LL', yl_norm)
            print('Level 1 - Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', yh1_norm[:, 0:3])
            print_wavelet_info('  HL (Vert.)', yh1_norm[:, 3:6])
            print_wavelet_info('  HH (Diag.)', yh1_norm[:, 6:9])
            print('Level 2 - Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', yh2_norm[:, 0:3])
            print_wavelet_info('  HL (Vert.)', yh2_norm[:, 3:6])
            print_wavelet_info('  HH (Diag.)', yh2_norm[:, 6:9])
            
            rec_yh1_norm, rec_yh2_norm, rec_yl_norm, _, _, usages, f, vq_loss = vae(yh1_norm, yh2_norm, yl_norm)

            print_wavelet_info('Reconstructed LL', rec_yl_norm)
            print('Level 1 - Reconstructed Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', rec_yh1_norm[:, 0:3])
            print_wavelet_info('  HL (Vert.)', rec_yh1_norm[:, 3:6])
            print_wavelet_info('  HH (Diag.)', rec_yh1_norm[:, 6:9])
            print('Level 2 - Reconstructed Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', rec_yh2_norm[:, 0:3])
            print_wavelet_info('  HL (Vert.)', rec_yh2_norm[:, 3:6])
            print_wavelet_info('  HH (Diag.)', rec_yh2_norm[:, 6:9])
            
            rec_yh1, rec_yh2, rec_yl = rec_yh1_norm * 2, rec_yh2_norm * 4,  rec_yl_norm * 4 # denormalization
            rec_yh1 = rec_yh1.view(rec_yh1.shape[0], 3, 3, rec_yh1.shape[2], rec_yh1.shape[3]) # -> (N, C, 3, H, W)
            rec_yh2 = rec_yh2.view(rec_yh2.shape[0], 3, 3, rec_yh2.shape[2], rec_yh2.shape[3]) # -> (N, C, 3, H, W)
            rec_imgs = idwt((rec_yl, [rec_yh1, rec_yh2]))
            
            # reconstruction loss
            l2_loss = nn.MSELoss(reduction='mean')
            rec_loss = l2_loss(rec_imgs, imgs)
            total_rec_loss += rec_loss.item() * imgs.shape[0]
            total_images += imgs.shape[0]
            # print(f'[INFO] Wavelet reconstruction loss: {l2_loss(wav_imgs, imgs).item()}')

            if first_batch:
                indices = list(range(0, 4)) + list(range(50, 54)) + list(range(100, 104)) + list(range(150, 154))
                rec_imgs = rec_imgs[indices]
                
                rec_imgs = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
                rec_imgs = torchvision.utils.make_grid(rec_imgs, nrow=8, padding=0)
                rec_imgs = rec_imgs.permute(1, 2, 0).cpu().mul_(255).numpy()
                rec_imgs = Image.fromarray(rec_imgs.astype(np.uint8))
                if args.rec_filename:
                    rec_imgs.save(args.rec_filename)
                first_batch = False
    
    avg_rec_loss = total_rec_loss / total_images
    print(f'[INFO] Load {vae_ckpt}')
    print(f'[RESULT] Reconstruction loss: {avg_rec_loss:.6f}')