# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
import torch.nn as nn


class DiscriminatorHL(nn.Module):
    """
    Defines a PatchGAN discriminator with Hinge loss
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
            ch_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(out_channels * ch_mult_prev, out_channels * ch_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(out_channels * ch_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ch_mult_prev = ch_mult
        ch_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(out_channels * ch_mult_prev, out_channels * ch_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(out_channels * ch_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(out_channels * ch_mult, 1, kernel_size=4, stride=1, padding=1)]  
        self.conv_block = nn.Sequential(*sequence)

    def forward(self, input):
        return self.conv_block(input)


class DiscriminatorCEL(nn.Module):
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
            ch_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(out_channels * ch_mult_prev, out_channels * ch_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(out_channels * ch_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ch_mult_prev = ch_mult
        ch_mult = min(2**n, 8)
        sequence += [
            nn.Conv2d(out_channels * ch_mult_prev, out_channels * ch_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(out_channels * ch_mult),
            nn.LeakyReLU(0.2, True)
        ]

        self.conv_block = nn.Sequential(*sequence)

        # output binary class logits
        self.in_features = out_channels * ch_mult * 31 * 31
        self.classifier_head = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=out_channels * ch_mult),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=out_channels * ch_mult, out_features=1)
        )

    def forward(self, input):
        x = self.conv_block(input)
        x = x.view(-1, self.in_features)
        return self.classifier_head(x)
    

def weights_init(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, (nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)