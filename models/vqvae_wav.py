"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_vae import Decoder, Encoder
from .quant import VectorQuantizer2


# class UpsampleWav(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)  
#         self.channel_reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
#     def forward(self, x):
#         x = self.upconv(x)          # upsample spatially (64x64 -> 128x128)
#         x = self.channel_reduce(x)  # reduce channels (36 -> 9)
#         return x


# class DownsampleWav(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
#         self.channel_expand = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
#     def forward(self, x):
#         x = self.conv(x)            # downsample spatially (128x128 -> 64x64)
#         x = self.channel_expand(x)  # adjust channels (9 -> 36)
#         return x


class UpsampleWav(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return F.pad(self.upconv(x), pad=(0, -1, 0, -1))


class DownsampleWav(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))
    

class VQVAE_WAV(nn.Module):
    def __init__(
        self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        ch_mult=(1, 2, 4), in_channels=12,
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=in_channels, ch_mult=ch_mult, num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                             # from vq-f16/config.yaml above
        )
        ddconfig.pop('double_z', None) # only KL-VAE should use double_z=True

        # encoding
        self.downsample_h1 = DownsampleWav(in_channels=3, out_channels=12) # (3, 128, 128) -> (12, 64, 64)
        self.encoder = Encoder(double_z=False, **ddconfig)
        
        # decoding
        self.upsample_h1 = UpsampleWav(in_channels=12, out_channels=3) # (12, 64, 64) -> (3, 128, 128)
        self.decoder = Decoder(**ddconfig)
        
        self.num_features = 4
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae * self.num_features, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, h1, h2, ll2, ret_usages=False):   # -> rec_B3HW, idx_N, loss
        VectorQuantizer2.forward

        h2 = h2.view(h2.shape[0], -1, h2.shape[3], h2.shape[4]) # -> (N, C * 3, H, W)

        inp = [self.downsample_h1(h1[:, i]) for i in range(3)]
        inp += [torch.cat([h2, ll2], dim=1)]
        # print(f'[SHAPE] Input: {inp[0].shape}')

        # (batch_size, Cvae, 16, 16)
        out = [self.quant_conv(self.encoder(x)) for x in inp]

        f = torch.cat(out, dim=1) # -> (batch_size, Cvae * num_features, 16, 16)
        f_hat, usages, vq_loss = self.quantize(f, ret_usages=ret_usages)
        # print(f'[SHAPE] Latent: {f.shape}')
        
        rec_inp = [self.decoder(self.post_quant_conv(f_hat[:, i * self.Cvae : (i + 1) * self.Cvae])) for i in range(self.num_features)]

        rec_h1 = [self.upsample_h1(rec_inp[i]) for i in range(3)]
        rec_h2 = rec_inp[3][:, :9]
        rec_ll2 = rec_inp[3][:, 9:]
        
        rec_h1 = torch.stack(rec_h1, dim=1)
        rec_h2 = rec_h2.view(rec_h2.shape[0], -1, 3, h2.shape[2], h2.shape[3]) # -> (N, C, 3, H, W)
        return rec_h1, rec_h2, rec_ll2, usages, vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)