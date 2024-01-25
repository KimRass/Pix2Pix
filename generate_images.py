import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm

from model import Generator
from torch_utils import get_device
from image_utils import save_image, image_to_grid
from train import select_ds


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    G = Generator(in_channels=3, out_channels=3).to(DEVICE)

    ### Load pre-trained parameters.
    ckpt = torch.load(args.ckpt_path, map_location=DEVICE)
    G.load_state_dict(ckpt)

    ds, input_img_mean, input_img_std, output_img_mean, output_img_std = select_ds(args)
    test_ds = ds(
        data_dir=args.data_dir,
        input_img_mean=input_img_mean,
        input_img_std=input_img_std,
        output_img_mean=output_img_mean,
        output_img_std=output_img_std,
        split="test",
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=False,
        drop_last=False,
    )

    ### Generate images.
    G.eval()
    with torch.no_grad():
        for idx, (input_image, real_output_image) in enumerate(tqdm(test_dl), start=1):
            input_image = input_image.to(DEVICE)
            real_output_image = real_output_image.to(DEVICE)

            output_image = G(input_image)
            grid = image_to_grid(
                input_image=input_image,
                real_output_image=real_output_image,
                fake_output_image=output_image,
                input_img_mean=input_img_mean,
                input_img_std=input_img_std,
                output_img_mean=output_img_mean,
                output_img_std=output_img_std,
            )
            save_image(grid, path=f"{Path(args.save_dir)}/{idx}.jpg")
