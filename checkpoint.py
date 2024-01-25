import argparse

from torch_utils import get_device, save_G


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--old_ckpt_path", type=str, required=True)
    parser.add_argument("--new_ckpt_path", type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    DEVICE = get_device()
    save_G(
        old_ckpt_path=args.old_ckpt_path, new_ckpt_path=args.new_ckpt_path, device=DEVICE,
    )
