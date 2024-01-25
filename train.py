import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from pathlib import Path
import argparse
import math

import config
from model import Generator, Discriminator
from torch_utils import get_device
from facades import FacadesDataset
from google_maps import GoogleMapsDataset
from image_utils import save_image, images_to_grid


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, required=False, default=200) # "Trained for $200$ epochs."
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument("--lamb", type=int, required=False, default=100)
    parser.add_argument("--resume_from", type=str, required=False)

    args = parser.parse_args()
    return args


def save_checkpoint(epoch, D, G, D_optim, G_optim, scaler, loss, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "D_optimizer": D_optim.state_dict(),
        "G_optimizer": G_optim.state_dict(),
        "scaler": scaler.state_dict(),
        "loss": loss,
    }
    torch.save(ckpt, str(save_path))


def save_G(G, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), str(save_path))


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
    PARENT_DIR = Path(__file__).parent

    args = get_args()

    # 논문에서는 batch size를 1로 했는데, 그보다 큰 값으로 할 경우 Batch size를 제곱한 값에 비례하여
    # learning rate를 크게 만들겠습니다.
    lr = config.LR * ((args.batch_size) ** 0.5)
    print(f"Learning rate: {lr}")

    DEVICE = get_device()
    G = Generator(in_channels=3, out_channels=3).to(DEVICE)
    D = Discriminator(in_channels=6).to(DEVICE)

    D_optim = Adam(
        params=D.parameters(), lr=lr, betas=(config.BETA1, config.BETA2),
    )
    G_optim = Adam(
        params=G.parameters(), lr=lr, betas=(config.BETA1, config.BETA2),
    )

    scaler = GradScaler()

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

    cgan_crit = nn.BCEWithLogitsLoss()
    l1_crit = nn.L1Loss()

    ### Resume
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=DEVICE)
        D.load_state_dict(ckpt["D"])
        G.load_state_dict(ckpt["G"])
        D_optim.load_state_dict(ckpt["D_optimizer"])
        G_optim.load_state_dict(ckpt["G_optimizer"])
        best_loss = ckpt["loss"]
        prev_ckpt_path = args.resume_from
        init_epoch = ckpt["epoch"]
        print(f"Resume from checkpoint '{args.resume_from}'.")
        print(f"Best loss ever: {best_loss:.4f}")
    else:
        best_loss = math.inf
        prev_ckpt_path = ".pth"
        init_epoch = 0

    for epoch in range(init_epoch + 1, args.n_epochs + 1):
        accum_D_loss = 0
        accum_fake_G_loss = 0
        accum_l1_loss = 0
        accum_tot_loss = 0
        for step, (src_image, real_trg_image) in enumerate(train_dl, start=1):
            src_image = src_image.to(DEVICE)
            real_trg_image = real_trg_image.to(DEVICE)

            ### Train D.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                real_pred = D(src_image=src_image, trg_image=real_trg_image)
                real_D_loss = cgan_crit(
                    real_pred, torch.ones_like(real_pred, device=real_pred.device),
                ) # "$\mathbb{E}_{x, y}[\log D(x, y)]$"

                fake_trg_image = G(src_image)
                fake_pred = D(src_image=src_image, trg_image=fake_trg_image.detach())
                fake_D_loss = cgan_crit(
                    fake_pred, torch.zeros_like(fake_pred, device=real_pred.device),
                ) # "$\mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$"

                D_loss = (real_D_loss + fake_D_loss) / 2 # "$\mathcal{L}_{cGAN}(G, D)$"
            D_optim.zero_grad()
            scaler.scale(D_loss).backward()
            scaler.step(D_optim)

            ### Train G.
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=True):
                fake_trg_image = G(src_image)
                fake_pred = D(src_image=src_image, trg_image=fake_trg_image)
                fake_G_loss = cgan_crit(
                    fake_pred, torch.ones_like(fake_pred, device=real_pred.device),
                )

                # "$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\lVert y - G(x, z) \rVert_{1}]$"
                l1_loss = l1_crit(fake_trg_image, real_trg_image)
                G_loss = fake_G_loss + args.lamb * l1_loss
            G_optim.zero_grad()
            scaler.scale(G_loss).backward()
            scaler.step(G_optim)

            scaler.update()

            accum_D_loss += D_loss.item()
            accum_fake_G_loss += fake_G_loss.item()
            accum_l1_loss += l1_loss.item()
            accum_tot_loss += D_loss.item() + G_loss.item()

        print(f"[ {epoch}/{str(args.n_epochs)} ][ {step}/{len(train_dl)} ]", end="")
        print(f"[ D loss: {accum_D_loss / len(train_dl):.4f} ]", end="")
        print(f"[ G cGAN loss: {accum_fake_G_loss / len(train_dl):.4f} ]", end="")
        print(f"[ L1 loss: {accum_l1_loss / len(train_dl):.4f} ]")

        cur_ckpt_path = f"{PARENT_DIR}/pretrained/{args.dataset}_epoch_{epoch}.pth"
        save_G(G=G, save_path=cur_ckpt_path)
        Path(prev_ckpt_path).unlink(missing_ok=True)
        prev_ckpt_path = cur_ckpt_path

        if epoch % config.N_GEN_EPOCHS == 0:
            grid = images_to_grid(
                src_image=src_image,
                real_trg_image=real_trg_image,
                fake_trg_image=fake_trg_image,
                input_img_mean=input_img_mean,
                input_img_std=input_img_std,
                output_img_mean=output_img_mean,
                output_img_std=output_img_std,
            )
            save_image(
                grid,
                path=f"{PARENT_DIR}/Gerated_images/{args.dataset}/epoch_{epoch}.jpg",
            )
