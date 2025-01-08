import torch
import torch.nn as nn
import torchvision
import numpy as np
import pywt

from torchvision.transforms import InterpolationMode, transforms
from PIL import Image

from trainer_wav import VQVAE_WAV_Trainer
from models import build_vae
from utils import arg_util
from utils.data_wav import WaveletDataModule, ImageDataModule


def normalize_01_into_pn1(x):
    """
    Normalize x from [0, 1] to [-1, 1]
    """
    return x.add(x).add_(-1)


def get_train_and_val_aug():
    final_reso = 256
    mid_reso = round(1.125 * final_reso)

    train_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.RandomCrop((final_reso, final_reso)),
        transforms.ToTensor(), 
        normalize_01_into_pn1
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), 
        normalize_01_into_pn1
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return transforms.Compose(train_aug), transforms.Compose(val_aug)


def get_images_from_wavelet(l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll):
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
            low = ll[i].detach().cpu().numpy()
            
            img = reconstruct_multilevel_2d(highs, low, 'haar')
            imgs.append(img)
            
            rec_highs = [
                rec_l2_hs[i].view(3, 3, rec_l2_hs.shape[2], rec_l2_hs.shape[3]).cpu().numpy(),
                rec_l1_hs[i].view(3, 3, rec_l1_hs.shape[2], rec_l1_hs.shape[3]).cpu().numpy()
            ]
            rec_low = rec_ll[i].cpu().numpy()
            
            rec_img = reconstruct_multilevel_2d(rec_highs, rec_low, 'haar')
            rec_imgs.append(rec_img)

        imgs = torch.tensor(np.stack(imgs))
        rec_imgs = torch.tensor(np.stack(rec_imgs))
        return imgs, rec_imgs


def reconstruct_multilevel_2d(highs, low, wavelet, mode='periodization'):
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


if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data loading
    wavelet_data_module = WaveletDataModule(args)
    wavelet_test_loader = wavelet_data_module.test_dataloader()
    if 'imagenet' in args.data_path:
        image_path = '../datasets/imagenet-100'
    elif 'ffhq' in args.data_path:
        image_path = '../datasets/ffhq'
    else:
        raise ValueError(f'{"*"*10}  Unknown image path {"*"*10}')
    train_aug, val_aug = get_train_and_val_aug()
    image_data_module = ImageDataModule(args, image_path, train_aug, val_aug)
    image_test_loader = image_data_module.test_dataloader()

    # build the model
    vae = build_vae(
        patch_nums=args.patch_nums,
        V=4096, Cvae=256, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
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
        for wav_batch, img_batch in zip(wavelet_test_loader, image_test_loader):
            (l1_hs, l2_hs, ll), labels1 = wav_batch
            # normalization
            l1_hs, l2_hs, ll = l1_hs / 2**1, l2_hs / 2**2,  ll / 2**2
            imgs, labels2 = img_batch
            assert(torch.equal(labels1, labels2))

            """
            tensor_dict = {'imgs': imgs, 'l1_hs': l1_hs, 'l2_hs': l2_hs, 'll': ll}
            for i in range(imgs.shape[0]):
                print(f'[INFO] Example {i}')
                for key, val in tensor_dict.items():
                    example_min = val[i].min().item()
                    example_max = val[i].max().item()
                    print(f"[INFO] '{key}': Min = {example_min}, Max = {example_max}")
            """
            
            # print(f'[INFO] Original max: {imgs.max()}, min: {imgs.min()}')
            # print(f'[INFO] Reconstructed max: {rec_imgs.max():.2f}, min: {rec_imgs.min():.2f}')
            l1_hs, l2_hs, ll = l1_hs.to(device), l2_hs.to(device), ll.to(device)
            rec_l1_hs, rec_l2_hs, rec_ll, _, _, _, _ = vae(l1_hs, l2_hs, ll)
            # denormalization
            l1_hs, l2_hs, ll = l1_hs * 2**1, l2_hs * 2**2,  ll * 2**2
            rec_l1_hs, rec_l2_hs, rec_ll = rec_l1_hs * 2**1, rec_l2_hs * 2**2,  rec_ll * 2**2
            
            # reconstruction loss
            wav_imgs, rec_imgs = get_images_from_wavelet(l1_hs, l2_hs, ll, rec_l1_hs, rec_l2_hs, rec_ll)
            l2_loss = nn.MSELoss(reduction='mean')
            rec_loss = l2_loss(rec_imgs, imgs)
            total_rec_loss += rec_loss.item() * imgs.shape[0]
            total_images += imgs.shape[0]
            # print(f'[INFO] Wavelet reconstruction loss: {l2_loss(wav_imgs, imgs).item()}')

            if first_batch:
                # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

                indices = list(range(0, 4)) + list(range(50, 54)) + list(range(100, 104)) + list(range(150, 154))
                rec_imgs = rec_imgs[indices]

                # rec_imgs = torch.clamp(rec_imgs * std + mean, min=0, max=1)
                rec_imgs = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
                rec_imgs = torchvision.utils.make_grid(rec_imgs, nrow=8, padding=0)
                rec_imgs = rec_imgs.permute(1, 2, 0).cpu().mul_(255).numpy()
                rec_imgs = Image.fromarray(rec_imgs.astype(np.uint8))
                rec_imgs.save('recon_test.png')
                first_batch = False
    
    avg_rec_loss = total_rec_loss / total_images
    print(f'[INFO] Load {vae_ckpt}')
    print(f'[RESULT] Reconstruction loss: {avg_rec_loss:.6f}')