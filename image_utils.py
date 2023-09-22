import torch
from torchvision.utils import make_grid
from einops import rearrange
import numpy as np
from PIL import Image
from pathlib import Path

import config
from torch_utils import denorm


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def save_image(img, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _to_pil(img).save(str(path))


def batched_image_to_grid(image, n_cols):
    b, _, h, w = image.shape
    assert b % n_cols == 0,\
        "The batch size should be a multiple of `n_cols` argument"
    pad = max(2, int(max(h, w) * 0.02))
    grid = make_grid(tensor=image, nrow=n_cols, normalize=False, padding=pad)
    grid = grid.clone().permute((1, 2, 0)).detach().cpu().numpy()

    grid *= 255.0
    grid = np.clip(a=grid, a_min=0, a_max=255).astype("uint8")

    for k in range(n_cols + 1):
        grid[:, (pad + h) * k: (pad + h) * k + pad, :] = 255
    for k in range(b // n_cols + 1):
        grid[(pad + h) * k: (pad + h) * k + pad, :, :] = 255
    return grid


def images_to_grid(
    input_image,
    real_output_image,
    fake_output_image,
    input_img_mean,
    input_img_std,
    output_img_mean,
    output_img_std,
):
    input_image = input_image.detach().cpu()
    real_output_image = real_output_image.detach().cpu()
    fake_output_image = fake_output_image.detach().cpu()

    input_image = denorm(input_image, mean=input_img_mean, std=input_img_std)
    real_output_image = denorm(real_output_image, mean=output_img_mean, std=output_img_std)
    fake_output_image = denorm(fake_output_image, mean=output_img_mean, std=output_img_std)

    concat = torch.cat([input_image, real_output_image, fake_output_image], dim=0)
    gen_image = rearrange(concat, pattern="(n m) c h w -> (m n) c h w", n=3)
    grid = batched_image_to_grid(gen_image, n_cols=3)
    return grid
