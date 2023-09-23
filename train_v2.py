import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
import argparse
import math

import config
from model import Generator, Discriminator
from loss import Pix2PixLoss
from torch_utils import get_device
from facades import FacadesDataset
from googlemaps import GoogleMapsDataset
from image_utils import save_image, images_to_grid


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=True) # "Trained for $200$ epochs."
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--lamb", type=int, required=False, default=100)
    parser.add_argument("--resume_from", type=str, required=False)

    args = parser.parse_args()
    return args


def save_checkpoint(epoch, disc, gen, disc_optim, gen_optim, loss, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "G": gen.state_dict(),
        "D": disc.state_dict(),
        "D_optimizer": disc_optim.state_dict(),
        "G_optimizer": gen_optim.state_dict(),
        "loss": loss,
    }
    torch.save(ckpt, str(save_path))


def select_ds(args):
    if args.dataset == "facades":
        ds = FacadesDataset
        input_img_mean = config.FACADES_INPUT_IMG_MEAN
        input_img_std = config.FACADES_INPUT_IMG_STD
        output_img_mean = config.FACADES_OUTPUT_IMG_MEAN
        output_img_std = config.FACADES_OUTPUT_IMG_STD
    elif args.dataset == "google_maps":
        ds = GoogleMapsDataset
        input_img_mean = config.GOOGLEMAPS_INPUT_IMG_MEAN
        input_img_std = config.GOOGLEMAPS_INPUT_IMG_STD
        output_img_mean = config.GOOGLEMAPS_OUTPUT_IMG_MEAN
        output_img_std = config.GOOGLEMAPS_OUTPUT_IMG_STD
    return ds, input_img_mean, input_img_std, output_img_mean, output_img_std


if __name__ == "__main__":
    args = get_args()

    # 논문에서는 batch size를 1로 했는데, 그보다 큰 값으로 할 경우 Batch size를 제곱한 값에 비례하여
    # learning rate를 크게 만들겠습니다.
    lr = config.LR * ((args.batch_size) ** 0.5)
    print(f"Learning rate: {lr}")

    DEVICE = get_device()
    gen = Generator(in_channels=3, out_channels=3).to(DEVICE)
    disc = Discriminator(in_channels=6).to(DEVICE)

    disc_optim = Adam(
        params=disc.parameters(), lr=lr, betas=(config.BETA1, config.BETA2),
    )
    gen_optim = Adam(
        params=gen.parameters(), lr=lr, betas=(config.BETA1, config.BETA2),
    )

    ds, input_img_mean, input_img_std, output_img_mean, output_img_std = select_ds(args)
    train_ds = ds(
        data_dir=args.data_dir,
        input_img_mean=input_img_mean,
        input_img_std=input_img_std,
        output_img_mean=output_img_mean,
        output_img_std=output_img_std,
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

    cgan_crit = nn.BCELoss()
    l1_crit = nn.L1Loss()

    ### Resume
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=DEVICE)
        disc.load_state_dict(ckpt["D"])
        gen.load_state_dict(ckpt["G"])
        disc_optim.load_state_dict(ckpt["D_optimizer"])
        gen_optim.load_state_dict(ckpt["G_optimizer"])
        best_loss = ckpt["loss"]
        prev_ckpt_path = args.resume_from
        init_epoch = ckpt["epoch"]
        print(f"Resume from checkpoint '{args.resume_from}'.")
        print(f"Best loss ever: {best_loss:.2f}")
    else:
        best_loss = math.inf
        prev_ckpt_path = ".pth"
        init_epoch = 0

    for epoch in range(init_epoch + 1, args.n_epochs + 1):
        accum_disc_loss = 0
        accum_fake_gen_loss = 0
        accum_l1_loss = 0
        accum_tot_loss = 0
        for step, (input_image, real_output_image) in enumerate(train_dl, start=1):
            input_image = input_image.to(DEVICE)
            real_output_image = real_output_image.to(DEVICE)

            ### Train D.
            real_pred = disc(input_image=input_image, output_image=real_output_image)
            real_disc_loss = cgan_crit(
                real_pred, torch.ones_like(real_pred, device=real_pred.device),
            ) # "$\mathbb{E}_{x, y}[\log D(x, y)]$"

            fake_output_image = gen(input_image)
            fake_pred = disc(input_image=input_image, output_image=fake_output_image.detach())
            fake_disc_loss = cgan_crit(
                fake_pred, torch.zeros_like(fake_pred, device=real_pred.device),
            ) # "$\mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$"

            disc_loss = real_disc_loss + fake_disc_loss # "$\mathcal{L}_{cGAN}(G, D)$"
            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            ### Train G.
            fake_output_image = gen(input_image)
            fake_pred = disc(input_image=input_image, output_image=fake_output_image)
            fake_gen_loss = cgan_crit(
                fake_pred, torch.ones_like(fake_pred, device=real_pred.device),
            ) # Not in the paper

            # "$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\lVert y - G(x, z) \rVert_{1}]$"
            l1_loss = l1_crit(fake_output_image, real_output_image)

            gen_loss = fake_gen_loss + args.lamb * l1_loss
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            accum_disc_loss += disc_loss.item()
            accum_fake_gen_loss += fake_gen_loss.item()
            accum_l1_loss += l1_loss.item()
            accum_tot_loss += disc_loss.item() + gen_loss.item()

        print(f"[ {epoch}/{str(args.n_epochs)} ][ {step}/{len(train_dl)} ]", end="")
        print(f"[ D loss: {accum_disc_loss / len(train_dl): .4f} ]", end="")
        print(f"[ G cGAN loss: {accum_fake_gen_loss / len(train_dl): .2f} ]", end="")
        print(f"[ L1 loss: {accum_l1_loss / len(train_dl): .4f} ]")

        if epoch % config.N_GEN_EPOCHS == 0:
            grid = images_to_grid(
                input_image=input_image,
                real_output_image=real_output_image,
                fake_output_image=fake_output_image,
                input_img_mean=input_img_mean,
                input_img_std=input_img_std,
                output_img_mean=output_img_mean,
                output_img_std=output_img_std,
            )
            save_image(
                grid, path=f"{Path(__file__).parent}/generated_images/epoch_{epoch}.jpg",
            )

        if accum_tot_loss < best_loss:
            cur_ckpt_path = f"{Path(__file__).parent}/checkpoints/epoch_{epoch}.pth"
            save_checkpoint(
                epoch=epoch,
                disc=disc,
                gen=gen,
                disc_optim=disc_optim,
                gen_optim=gen_optim,
                loss=accum_fake_gen_loss,
                save_path=cur_ckpt_path,
            )
            Path(prev_ckpt_path).unlink(missing_ok=True)
            print(f"Saved checkpoint.")

            best_loss = accum_tot_loss
            prev_ckpt_path = cur_ckpt_path
