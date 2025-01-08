# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Defines a PatchGAN discriminator
    """
    def __init__(self, in_channels=3, out_channels=64, n_layers=3):
        super().__init__()
        norm_layer = nn.BatchNorm2d # no need to use bias as BatchNorm2d has affine parameters
        sequence = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2, True)
        ]
        # gradually increase the number of channels
        ch_mult = 1
        ch_mult_prev = 1
        for n in range(1, n_layers): 
            ch_mult_prev = ch_mult
            ch_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(out_channels * ch_mult_prev, out_channels * ch_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(out_channels * ch_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ch_mult_prev = ch_mult
        ch_mult = min(2 ** n, 8)
        sequence += [
            nn.Conv2d(out_channels * ch_mult_prev, out_channels * ch_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(out_channels * ch_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(out_channels * ch_mult, 1, kernel_size=4, stride=1, padding=1)]  
        self.convs = nn.Sequential(*sequence)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.convs(input)
        x = self.sigmoid(x)
        return x


def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)