import torch
from torch.optim import Adam
from pathlib import Path
import argparse

import config
from model import Generator, Discriminator
from loss import Pix2PixLoss
from torch_utils import get_device, denormalize, save_parameters
from facades import get_facades_dataloader
from image_utils import save_image, batched_image_to_grid


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator(in_channels=3, out_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=6).to(DEVICE)

    disc_optim = Adam(params=disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    gen_optim = Adam(params=gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    train_dl = get_facades_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        split="train",
    )

    crit = Pix2PixLoss()
    losses = list()
    for epoch in range(1, config.N_EPOCHS + 1):
        for step, (label, real_photo) in enumerate(train_dl, start=1):
            label = label.to(DEVICE)
            real_photo = real_photo.to(DEVICE)

            disc_optim.zero_grad()
            gen_optim.zero_grad()

            fake_photo = gen(label)
            real_pred = disc(label, real_photo)
            fake_pred = disc(label, fake_photo)
            cgan_loss, l1_loss = crit(
                real_photo=real_photo, fake_photo=fake_photo, real_pred=real_pred, fake_pred=fake_pred
            )
            loss = cgan_loss + config.LAMB * l1_loss
            loss.backward()

            disc_optim.step()
            gen_optim.step()

            if step == len(train_dl):
                print(f"[ {epoch}/{str(config.N_EPOCHS)} ][ {step}/{len(train_dl)} ], enc=""")
                print(f"[ CGAN loss: {cgan_loss.item(): .4f} ][ L1 loss: {l1_loss.item(): .4f} ]")

        if epoch % 2 == 0:
            label = label.detach().cpu()
            real_photo = real_photo.detach().cpu()
            fake_photo = fake_photo.detach().cpu()

            label = denormalize(label, mean=(0.222, 0.299, 0.745), std=(0.346, 0.286, 0.336))
            real_photo = denormalize(real_photo, mean=(0.478, 0.453, 0.417), std=(0.243, 0.235, 0.236))
            fake_photo = denormalize(fake_photo, mean=(0.478, 0.453, 0.417), std=(0.243, 0.235, 0.236))

            image = torch.cat([label, real_photo, fake_photo], dim=0)
            grid = batched_image_to_grid(image, n_cols=3)
            save_image(grid, path=f"""{Path(__file__).parent}/generated_images/epoch_{epoch}.jpg""")

        if epoch % 10 == 0:
            save_parameters(
                model=gen,
                save_path=f"""{Path(__file__).parent}/pretrained/epoch_{epoch}.pth"""
            )
