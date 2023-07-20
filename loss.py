# References: https://deep-learning-study.tistory.com/646, https://www.tensorflow.org/tutorials/generative/pix2pix

import torch
import torch.nn as nn

from model import Generator, Discriminator


class Pix2PixLoss(nn.Module):
    def __init__(self, lamb=100):
        super().__init__()

        self.lamb = lamb

        self.cgan_crit = nn.BCELoss()
        self.l1_crit = nn.L1Loss()

    def forward(self, gen_output, disc_output, y):
        cgan_loss = self.cgan_crit(torch.ones_like(disc_output), disc_output)
        l1_loss = self.l1_crit(gen_output, y)
        loss = cgan_loss + self.lamb * l1_loss
        return loss


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
