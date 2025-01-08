# From https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py
# Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models

import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision import models
from collections import namedtuple


class LPIPS(nn.Module):
    """
    Learned perceptual metric
    """
    def __init__(self, in_channels, ckpt_path=None, pretrained=False, requires_grad=True, use_dropout=True):
        super().__init__()
        # self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=pretrained, requires_grad=requires_grad)
        # adjust for the wavelet input channels
        conv = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.net.slice1[0] = conv
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)

        if ckpt_path:
            self.load_from_pretrained(ckpt_path)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def load_from_pretrained(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')), strict=False)
        print(f'[INFO] Loaded pretrained LPIPS loss from {ckpt_path}')

    def forward(self, input, target):
        # in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        in0_input, in1_input = input, target
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        loss = 0.0
        # print(f'[INFO] Input L2: {F.mse_loss(in0_input, in1_input):.6f}')
        for kk in range(len(self.chns)):
            feats0, feats1 = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            feats0, feats1 = lins[kk].model(feats0), lins[kk].model(feats1)
            # print(f'[SHAPE] Features: {feats0.shape}')
            # print(f'[INFO] Feature-{kk} L2: {F.mse_loss(feats0, feats1):.6f}')
            loss += F.mse_loss(feats0, feats1)
        return loss


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ 
    A single linear layer which does a 1x1 conv 
    """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        # print('X', X.shape)
        # print('h_relu1_2', h_relu1_2.shape)
        # print('h_relu2_2', h_relu2_2.shape)
        # print('h_relu3_3', h_relu3_3.shape)
        # print('h_relu5_3', h_relu4_3.shape)
        # print('h_relu4_3', h_relu5_3.shape)
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)