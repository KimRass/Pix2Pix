# References:
    # https://deep-learning-study.tistory.com/646,
    # https://www.tensorflow.org/tutorials/generative/pix2pix

import torch
import torch.nn as nn

from model import Generator, Discriminator


class Pix2PixLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.cgan_crit = nn.BCELoss()
        # "Using L1 distance rather than L2 as L1 encourages less blurring."
        self.l1_crit = nn.L1Loss()

    def forward(self, real_output_image, fake_output_image, real_pred, fake_pred):
        # "$\mathbb{E}_{x, y}[\log D(x, y)]$"
        real_loss = self.cgan_crit(
            real_pred, torch.ones_like(real_pred, device=real_pred.device),
        )
        # "$\mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$"
        fake_loss = self.cgan_crit(
            fake_pred, torch.zeros_like(fake_pred, device=real_pred.device),
        )
        cgan_loss = real_loss + fake_loss # "$\mathcal{L}_{cGAN}(G, D)$"
        cgan_loss *= 0.5

        # "$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\lVert y - G(x, z) \rVert_{1}]$"
        l1_loss = self.l1_crit(fake_output_image, real_output_image)
        return cgan_loss, l1_loss


if __name__ == "__main__":
    BATCH_SIZE = 16
    x = torch.randn((BATCH_SIZE, 3, 256, 256))
    y = torch.randn((BATCH_SIZE, 3, 256, 256))

    gen = Generator(in_ch=3, out_ch=3)
    gen_output = gen(x)

    disc = Discriminator(in_ch=6)
    disc_output = disc(x, y)
    disc_output.shape
    torch.ones_like(disc_output).dtype
    disc_output.dtype

    crit = Pix2PixLoss()
    crit(gen_output=gen_output, disc_output=disc_output, y=y)
