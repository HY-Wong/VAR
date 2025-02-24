import torch
import torch.nn as nn
import torchvision
import numpy as np

from PIL import Image
from pytorch_wavelets import DWTForward, DWTInverse
from tqdm import tqdm

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
        V=4096, Cvae=args.Cvae, ch=128, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        init_vae=args.init_vae, init_vocab=args.init_vocab,
        ch_mult=args.ch_mult, in_channels=args.in_channels
    ).to(device)
    print(vae)
    exit(0)

    vae_ckpt = args.load_ckpt_path
    if vae_ckpt:
        # load from Lightning’s checkpoint
        ckpt = torch.load(args.load_ckpt_path, map_location='cpu', weights_only=False)
        state_dict = {k.replace('vae.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('vae.')}
        vae.load_state_dict(state_dict, strict=True)
    
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
            ll2, (h1, h2) = dwt(imgs)
            h1, h2, ll2 = h1 / 2, h2 / 4, ll2 / 4 # normalization
            
            print_wavelet_info('LL', ll2)
            print('Level 1 - Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', h1[:, 0])
            print_wavelet_info('  HL (Vert.)', h1[:, 1])
            print_wavelet_info('  HH (Diag.)', h1[:, 2])
            print('Level 2 - Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', h2[:, 0])
            print_wavelet_info('  HL (Vert.)', h2[:, 1])
            print_wavelet_info('  HH (Diag.)', h2[:, 2])

            rec_h1, rec_h2, rec_ll2, usages, vq_loss = vae(h1, h2, ll2)

            print_wavelet_info('Reconstructed LL', rec_ll2)
            print('Level 1 - Reconstructed Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', rec_h1[:, 0])
            print_wavelet_info('  HL (Vert.)', rec_h1[:, 1])
            print_wavelet_info('  HH (Diag.)', rec_h1[:, 2])
            print('Level 2 - Reconstructed Detail Coefficients:')
            print_wavelet_info('  LH (Hori.)', rec_h2[:, 0])
            print_wavelet_info('  HL (Vert.)', rec_h2[:, 1])
            print_wavelet_info('  HH (Diag.)', rec_h2[:, 0])
            
            rec_h1, rec_h2, rec_ll2 = rec_h1 * 2, rec_h2 * 4,  rec_ll2 * 4 # denormalization
            rec_imgs = idwt((rec_ll2, [rec_h1, rec_h2]))
            
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