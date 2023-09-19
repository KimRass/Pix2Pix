import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import argparse

import config
from model import Generator, Discriminator
from loss import Pix2PixLoss
from torch_utils import get_device, denormalize, freeze_model, unfreeze_model
from facades import FacadesDataset
from image_utils import save_image, batched_image_to_grid


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


def save_checkpoint(epoch, disc, gen, disc_optim, gen_optim, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "G": gen.state_dict(),
        "D": disc.state_dict(),
        "D_optimizer": disc_optim.state_dict(),
        "G_optimizer": gen_optim.state_dict(),

    }
    torch.save(ckpt, str(save_path))


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator(in_channels=3, out_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=6).to(DEVICE)

    disc_optim = Adam(params=disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
    gen_optim = Adam(params=gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))

    train_ds = FacadesDataset(
        data_dir=args.data_dir,
        label_mean=config.FACADES_LABEL_MEAN,
        label_std=config.FACADES_LABEL_STD,
        photo_mean=config.FACADES_PHOTO_MEAN,
        photo_std=config.FACADES_PHOTO_STD,
        split="train",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    # crit = Pix2PixLoss()
    cgan_crit = nn.BCELoss()
    # "Using L1 distance rather than L2 as L1 encourages less blurring."
    l1_crit = nn.L1Loss()

    disc_accum_loss = 0
    gen_accum_loss = 0
    l1_accum_loss = 0
    for epoch in range(1, args.n_epochs + 1):
        for step, (label, real_image) in enumerate(train_dl, start=1):
            label = label.to(DEVICE)
            real_image = real_image.to(DEVICE)

            ### Optimize D.
            real_pred = disc(label, real_image)
            fake_image = gen(label)
            fake_pred = disc(label, fake_image.detach())

            # "$\mathbb{E}_{x, y}[\log D(x, y)]$"
            real_loss = cgan_crit(real_pred, torch.ones_like(real_pred, device=real_pred.device))
            # "$\mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$"
            fake_loss = cgan_crit(fake_pred, torch.zeros_like(fake_pred, device=real_pred.device))
            disc_loss = real_loss + fake_loss # "$\mathcal{L}_{cGAN}(G, D)$"

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            ### Optimize G.
            freeze_model(disc)

            fake_image = gen(label)
            fake_pred = disc(label, fake_image)

            fake_loss = cgan_crit(fake_pred, torch.zeros_like(fake_pred, device=real_pred.device))
            l1_loss = l1_crit(fake_image, real_image)
            gen_loss = fake_loss + config.LAMB * l1_loss

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            unfreeze_model(disc)

            disc_accum_loss += disc_loss.item()
            gen_accum_loss += gen_loss.item()
            l1_accum_loss += l1_loss.item()

            if step == len(train_dl):
                print(f"[ {epoch}/{str(args.n_epochs)} ][ {step}/{len(train_dl)} ]", end="")
                print(f"[ L1 loss: {l1_accum_loss / len(train_dl): .4f} ]", end="")
                print(f"[ D loss: {disc_accum_loss / len(train_dl): .4f} ]", end="")
                print(f"[ G loss: {gen_accum_loss / len(train_dl): .4f} ]")

                disc_accum_loss = 0
                gen_accum_loss = 0
                l1_accum_loss = 0

        if epoch % config.N_GEN_EPOCHS == 0:
            label = label.detach().cpu()
            real_image = real_image.detach().cpu()
            fake_image = fake_image.detach().cpu()

            label = denormalize(label, mean=(0.222, 0.299, 0.745), std=(0.346, 0.286, 0.336))
            real_image = denormalize(real_image, mean=(0.478, 0.453, 0.417), std=(0.243, 0.235, 0.236))
            fake_image = denormalize(fake_image, mean=(0.478, 0.453, 0.417), std=(0.243, 0.235, 0.236))

            image = torch.cat([label, real_image, fake_image], dim=0)
            grid = batched_image_to_grid(image, n_cols=3)
            save_image(grid, path=f"""{Path(__file__).parent}/generated_images/epoch_{epoch}.jpg""")

        if epoch % config.N_SAVE_EPOCHS == 0:
            save_checkpoint(
                epoch=epoch,
                disc=disc,
                gen=gen,
                disc_optim=disc_optim,
                gen_optim=gen_optim,
                save_path=f"""{Path(__file__).parent}/pretrained/epoch_{epoch}.pth""",
            )
