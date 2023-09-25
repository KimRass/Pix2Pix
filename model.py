# References:
    # https://neptune.ai/blog/pix2pix-key-model-architecture-decisions
    # https://www.tensorflow.org/tutorials/generative/pix2pix
    # https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/

import torch
import torch.nn as nn
import torch.nn.functional as F


# Let 'Ck' denote a Convolution-norm-ReLU layer with $k$ filters. 'CDk' denotes a
# Convolution-normDropout-ReLUlayer with a dropout rate of 50%. All convolutions are $4 \times 4$
# spatial filters applied with stride $2$. Convolutions in the encoder, and in the discriminator,
# downsample by a factor of $2$, whereas in the decoder they upsample by a factor of $$."
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        upsample=False,
        drop=True,
        normalize=True,
        leaky=False,
    ):
        super().__init__()

        self.stride = stride
        self.upsample = upsample
        self.drop = drop
        self.normalize = normalize
        self.leaky = leaky

        if not upsample:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False if normalize else True,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False if normalize else True,
            )
        if normalize:
            # "At inference time, we run the generator net in exactly the same manner as during
            # the training phase. This differs from the usual protocol in that we apply dropout
            # at test time, and we apply batch normalization using the statistics of the test batch,
            # rather than aggregated statistics of the training batch. This approach to batch
            # normalization, when the batch size is set to $1$, has been termed 'instance normalization'
            # and has been demonstrated to be effective at image generation tasks."
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        if drop:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        if self.drop:
            x = self.dropout(x)
        if self.leaky:
            x = F.leaky_relu(x, 0.2)
        else:
            x = torch.relu(x)
        return x


# "Weights were initialized from a Gaussian distribution with mean $0$ and standard deviation $0.02$."
def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data.normal_(0, 0.02)


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # "'C64-C128-C256-C512-C512-C512-C512-C512'"
        # (Comment: 'k'는 `out_channels`의 값을 의미합니다.)
        # "'BatchNorm' is not applied to the first 'C64' layer in the encoder."
        # "All ReLUs in the encoder are leaky, with slope $0.2$, while ReLUs in the decoder are not leaky."
        self.layer1 = ConvBlock(
            in_channels, 64, upsample=False, drop=False, normalize=False, leaky=True,
        )
        self.layer2 = ConvBlock(64, 128, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer3 = ConvBlock(128, 256, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer4 = ConvBlock(256, 512, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer5 = ConvBlock(512, 512, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer6 = ConvBlock(512, 512, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer7 = ConvBlock(512, 512, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer8 = ConvBlock(512, 512, upsample=False, drop=False, normalize=False, leaky=True)

        # 'CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128'"
        # (Comment: 'k'는 `in_channels`의 값을 의미합니다.)
        self.layer9 = ConvBlock(512, 512, upsample=True, drop=True, normalize=True, leaky=False)
        self.layer10 = ConvBlock(1024, 512, upsample=True, drop=True, normalize=True, leaky=False)
        self.layer11 = ConvBlock(1024, 512, upsample=True, drop=True, normalize=True, leaky=False)
        self.layer12 = ConvBlock(1024, 512, upsample=True, drop=False, normalize=True, leaky=False)
        self.layer13 = ConvBlock(1024, 256, upsample=True, drop=False, normalize=True, leaky=False)
        self.layer14 = ConvBlock(512, 128, upsample=True, drop=False, normalize=True, leaky=False)
        self.layer15 = ConvBlock(256, 64, upsample=True, drop=False, normalize=False, leaky=False)
        # "After the last layer in the decoder, a convolution is applied to map to the number of
        # output channels ($3$ in general, except in colorization, where it is $2$), followed by
        # a $Tanh$ function."
        self.layer16 = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)

        _init_weights(self)

    def forward(self, x):
        x1 = self.layer1(x) # `(b, 64, 128, 128)`
        x2 = self.layer2(x1) # `(b, 128, 64, 64)`
        x3 = self.layer3(x2) # `(b, 256, 32, 32)`
        x4 = self.layer4(x3) # `(b, 512, 16, 16)`
        x5 = self.layer5(x4) # `(b, 512, 8, 8)`
        x6 = self.layer6(x5) # `(b, 512, 4, 4)`
        x7 = self.layer7(x6) # `(b, 512, 2, 2)`
        x8 = self.layer8(x7) # `(b, 512, 1, 1)`

        x = self.layer9(x8) # `(b, 512, 2, 2)`
        x = self.layer10(torch.cat([x7, x], dim=1)) # `(b, 512, 4, 4)`
        x = self.layer11(torch.cat([x6, x], dim=1)) # `(b, 512, 8, 8)`
        x = self.layer12(torch.cat([x5, x], dim=1)) # `(b, 512, 16, 16)`
        x = self.layer13(torch.cat([x4, x], dim=1)) # `(b, 256, 32, 32)`
        x = self.layer14(torch.cat([x3, x], dim=1)) # `(b, 128, 64, 64)`
        x = self.layer15(torch.cat([x2, x], dim=1)) # `(b, 64, 128, 128)`
        x = self.layer16(torch.cat([x1, x], dim=1)) # `(b, 3, 256, 256)`
        x = torch.tanh(x)
        return x


def get_receptive_field(out_channels, kernel_size, stride):
    return (out_channels - 1) * stride + kernel_size


class Discriminator(nn.Module): # "$70 \times 70$ 'PatchhGAN'"
    def __init__(self, in_channels):
        super().__init__()

        # "C64-C128-C256-C512"
        # "All ReLUs are leaky, with slope $0.2$."
        # "'BatchNormorm' is not applied to the first 'C64' layer."
        self.layer1 = ConvBlock(
            in_channels, 64, upsample=False, drop=False, normalize=False, leaky=True,
        )
        self.layer2 = ConvBlock(64, 128, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer3 = ConvBlock(128, 256, upsample=False, drop=False, normalize=True, leaky=True)
        self.layer4 = ConvBlock(
            256, 512, stride=1, upsample=False, drop=False, normalize=True, leaky=True,
        )
        # "After the last layer, a convolution is applied to map to a 1-dimensional output,
        # followed by a Sigmoid function."
        self.layer5 = nn.Conv2d(512, 1, kernel_size=1)

        _init_weights(self)

    def forward(self, input_image, output_image):
        x = torch.cat([input_image, output_image], dim=1)

        x = self.layer1(x) # `(b, 64, 128, 128)`
        x = self.layer2(x) # `(b, 128, 64, 64)`
        x = self.layer3(x) # `(b, 256, 32, 32)`
        x = self.layer4(x) # `(b, 512, 31, 31)`
        x = self.layer5(x) # `(b, 1, 31, 31)`
        x = torch.sigmoid(x)
        # "We run the discriminator convolutionally across the image, averaging all responses
        # to provide the ultimate output of $D$."
        x = x.mean(dim=(2, 3))
        return x


if __name__ == "__main__":
    gen = Generator(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 256, 256)
    gen(x).shape

    rf = get_receptive_field(out_channels=1, kernel_size=4, stride=1)
    rf = get_receptive_field(out_channels=rf, kernel_size=4, stride=1)
    rf = get_receptive_field(out_channels=rf, kernel_size=4, stride=2)
    rf = get_receptive_field(out_channels=rf, kernel_size=4, stride=2)
    rf = get_receptive_field(out_channels=rf, kernel_size=4, stride=2)
    print(rf) # `70`

    disc = Discriminator(in_channels=6)
    x = y = torch.randn(4, 3, 256, 256)
    disc(x, y).shape