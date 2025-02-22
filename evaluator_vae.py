import torch
import torch.nn as nn
import torchvision
import numpy  as np

from torchvision.transforms import InterpolationMode, transforms
from PIL import Image

from models import build_vae_var
from utils import arg_util
from utils.data import ImageDataModule


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
        transforms.ToTensor(), normalize_01_into_pn1
    ]
    val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pn1
    ]
    return transforms.Compose(train_aug), transforms.Compose(val_aug)


if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # data loading
    train_aug, val_aug = get_train_and_val_aug()
    image_data_module = ImageDataModule(args)
    image_test_loader = image_data_module.test_dataloader()

    # build the model
    vae, _ = build_vae_var(
        device=device, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        num_classes=100, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    vae_ckpt = args.load_ckpt_path
    # vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f'[INFO] Number of trainable parameters {total_params}')
    
    # evaluate the reconstruction loss [-1, 1] range 
    total_rec_loss = 0.0
    total_images = 0
    first_batch = True

    vae.eval()
    with torch.no_grad():
        for imgs, _ in image_test_loader:
            imgs = imgs.to(device)
            rec_imgs, _, _ = vae(imgs)

            # print(f'[INFO] Original batch shape: {imgs.shape}, max: {imgs.max()}, min: {imgs.min()}')
            # print(f'[INFO] Reconstructed batch shape: {rec_imgs.shape}, max: {rec_imgs.max():.2f}, min: {rec_imgs.min():.2f}')
            l2_loss = nn.MSELoss(reduction='mean')
            rec_loss = l2_loss(rec_imgs, imgs)
            total_rec_loss += rec_loss.item() * imgs.shape[0]
            total_images += imgs.shape[0]

            if first_batch:
                # indices = list(range(0, 4)) + list(range(50, 54)) + list(range(100, 104)) + list(range(150, 154))
                # rec_imgs = rec_imgs[indices]

                rec_imgs = torchvision.utils.make_grid(rec_imgs, nrow=8, padding=0)
                rec_imgs = torch.clamp((rec_imgs + 1) / 2, min=0, max=1)
                rec_imgs = rec_imgs.permute(1, 2, 0).cpu().mul_(255).numpy()
                rec_imgs = Image.fromarray(rec_imgs.astype(np.uint8))
                rec_imgs.save('recon_baseline.png')
                first_batch = False

    avg_rec_loss = total_rec_loss / total_images
    print(f'[INFO] Load {vae_ckpt}')
    print(f'[RESULT] Reconstruction loss: {avg_rec_loss:.6f}')
