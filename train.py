import torch
from torch.optim import Adam
from pathlib import Path

from model import Generator, Discriminator
from loss import Pix2PixLoss
from torch_utils import get_device, denormalize, save_parameters
from facades import get_facades_dataloader
from image_utils import save_image, batched_image_to_grid

DEVICE = get_device()
gen = Generator(in_channels=3, out_channels=3).to(DEVICE)
disc = Discriminator(in_channels=6).to(DEVICE)

# "We use minibatch SGD and apply the Adam solver, with a learning rate of $0.0002$,
# and momentum parameters $\BETA_{1} = 0.5$, $\BETA_{2} = 0.999."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
disc_optim = Adam(params=disc.parameters(), lr=LR, betas=(BETA1, BETA2))
gen_optim = Adam(params=gen.parameters(), lr=LR, betas=(BETA1, BETA2))

BATCH_SIZE = 1
N_WORKERS = 4
# N_WORKERS = 0
train_dl = get_facades_dataloader(
    # archive_dir="/Users/jongbeomkim/Documents/datasets/facades/archive",
    archive_dir="/home/ubuntu/project/facades/archive",
    batch_size=BATCH_SIZE,
    n_workers=N_WORKERS,
    split="train",
)

crit = Pix2PixLoss()
LAMB = 100
# "Trained for $200$ epochs."
N_EPOCHS = 200
losses = list()
for epoch in range(1, N_EPOCHS + 1):
    for batch, (label, real_photo) in enumerate(train_dl, start=1):
        label = label.to(DEVICE)
        real_photo = real_photo.to(DEVICE)

        disc_optim.zero_grad()
        gen_optim.zero_grad()

        fake_photo = gen(label)
        real_pred = disc(label, real_photo)
        fake_pred = disc(label, fake_photo)
        # loss = crit(real_photo=real_photo, fake_photo=fake_photo, real_pred=real_pred, fake_pred=fake_pred)
        cgan_loss, l1_loss = crit(
            real_photo=real_photo, fake_photo=fake_photo, real_pred=real_pred, fake_pred=fake_pred
        )
        loss = cgan_loss + LAMB * l1_loss
        loss.backward()

        disc_optim.step()
        gen_optim.step()

        if batch == len(train_dl):
            # print(f"""[{epoch}/{str(N_EPOCHS)}][{batch}/{len(train_dl)}] loss: {loss.item(): .4f}""")
            print(f"""[{epoch}/{str(N_EPOCHS)}][{batch}/{len(train_dl)}] CGAN Loss: {cgan_loss.item(): .4f} | L1 Loss: {l1_loss.item(): .4f}""")

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
