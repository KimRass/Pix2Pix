# References:
    # https://deep-learning-study.tistory.com/646
    # https://www.tensorflow.org/tutorials/generative/pix2pix

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False, dropout=True, batchnorm=True, leaky=False):
        super().__init__()

        self.dropout = dropout
        self.batchnorm = batchnorm
        self.leaky = leaky

        if not upsample:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_ch)
        if dropout:
            self.do = nn.Dropout(0.5)
        if leaky:
            self.relu = nn.LeakyReLU(0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.dropout:
            x = self.do(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        # "C64-C128-C256-C512-C512-C512-C512-C512"
        self.layer1 = ConvolutionBlock(
            in_ch=in_ch, out_ch=64, upsample=False, dropout=False, batchnorm=False, leaky=True
        )
        self.layer2 = ConvolutionBlock(
            in_ch=64, out_ch=128, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer3 = ConvolutionBlock(
            in_ch=128, out_ch=256, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer4 = ConvolutionBlock(
            in_ch=256, out_ch=512, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer5 = ConvolutionBlock(
            in_ch=512, out_ch=512, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer6 = ConvolutionBlock(
            in_ch=512, out_ch=512, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer7 = ConvolutionBlock(
            in_ch=512, out_ch=512, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer8 = ConvolutionBlock(
            in_ch=512, out_ch=512, upsample=False, dropout=False, batchnorm=True, leaky=True
        )

        # "CD512-CD512-CD512-C512-C256-C128-C64"
        self.layer9 = ConvolutionBlock(
            in_ch=512, out_ch=512, upsample=True, dropout=True, batchnorm=True, leaky=False
        )
        self.layer10 = ConvolutionBlock(
            in_ch=1024, out_ch=512, upsample=True, dropout=True, batchnorm=True, leaky=False
        )
        self.layer11 = ConvolutionBlock(
            in_ch=1024, out_ch=512, upsample=True, dropout=True, batchnorm=True, leaky=False
        )
        self.layer12 = ConvolutionBlock(
            in_ch=1024, out_ch=512, upsample=True, dropout=False, batchnorm=True, leaky=False
        )
        self.layer13 = ConvolutionBlock(
            in_ch=1024, out_ch=256, upsample=True, dropout=False, batchnorm=True, leaky=False
        )
        self.layer14 = ConvolutionBlock(
            in_ch=512, out_ch=128, upsample=True, dropout=False, batchnorm=True, leaky=False
        )
        self.layer15 = ConvolutionBlock(
            in_ch=256, out_ch=64, upsample=True, dropout=False, batchnorm=False, leaky=False
        )

        self.last_conv = nn.ConvTranspose2d(128, out_ch, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.layer1(x) # (64, 128, 128)
        x2 = self.layer2(x1) # (128, 64, 64)
        x3 = self.layer3(x2) # (256, 32, 32)
        x4 = self.layer4(x3) # (512, 16, 16)
        x5 = self.layer5(x4) # (512, 8, 8)
        x6 = self.layer6(x5) # (512, 4, 4)
        x7 = self.layer7(x6) # (512, 2, 2)
        x8 = self.layer8(x7) # (512, 1, 1)

        x9 = self.layer9(x8) # (512, 2, 2)
        x10 = self.layer10(torch.cat([x7, x9], axis=1)) # (512, 4, 4)
        x11 = self.layer11(torch.cat([x6, x10], axis=1)) # (512, 8, 8)
        x12 = self.layer12(torch.cat([x5, x11], axis=1)) # (512, 16, 16)
        x13 = self.layer13(torch.cat([x4, x12], axis=1)) # (256, 32, 32)
        x14 = self.layer14(torch.cat([x3, x13], axis=1)) # (128, 64, 64)
        x15 = self.layer15(torch.cat([x2, x14], axis=1)) # (64, 128, 128)

        x = torch.cat([x1, x15], axis=1) # (128, 128, 128)
        x = self.last_conv(x)
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        # "C64-C128-C256-C512"
        self.layer1 = ConvolutionBlock(
            in_ch=in_ch, out_ch=64, upsample=False, dropout=False, batchnorm=False, leaky=True
        )
        self.layer2 = ConvolutionBlock(
            in_ch=64, out_ch=128, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer3 = ConvolutionBlock(
            in_ch=128, out_ch=256, upsample=False, dropout=False, batchnorm=True, leaky=True
        )
        self.layer4 = ConvolutionBlock(
            in_ch=256, out_ch=512, upsample=False, dropout=False, batchnorm=True, leaky=True
        )

        self.patch = nn.Conv2d(512, 1, kernel_size=1, stride=1)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)

        x = self.patch(x)
        print(x.shape)
        x = F.sigmoid(x)
        return x
