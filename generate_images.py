import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm

import config
from model import Generator
from torch_utils import get_device
from facades import FacadesDataset
from image_utils import save_image, facades_images_to_grid


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--n_workers", type=int, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    gen = Generator(in_channels=3, out_channels=3).to(DEVICE)

    ### Load pre-trained parameters.
    ckpt = torch.load(args.ckpt_path, map_location=DEVICE)
    gen.load_state_dict(ckpt)

    test_ds = FacadesDataset(
        data_dir=args.data_dir,
        input_img_mean=config.FACADES_INPUT_IMG_MEAN,
        input_img_std=config.FACADES_INPUT_IMG_STD,
        output_img_mean=config.FACADES_OUTPUT_IMG_MEAN,
        output_img_std=config.FACADES_OUTPUT_IMG_STD,
        split="test",
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        drop_last=True,
    )

    ### Generate images.
    gen.eval()
    with torch.no_grad():
        for idx, (input_image, real_output_image) in enumerate(tqdm(test_dl)):
            input_image = input_image.to(DEVICE)
            real_output_image = real_output_image.to(DEVICE)

            gen_output_image = gen(input_image)
            grid = facades_images_to_grid(
                input_image=input_image,
                real_output_image=real_output_image,
                fake_output_image=gen_output_image,
            )
            save_image(
                grid, path=f"""{Path(__file__).parent}/gen/{idx}.jpg""",
            )
