import torch
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
import torchvision.transforms as T


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_image_dataset_mean_and_std(data_dir, ext="jpg"):
    data_dir = Path(data_dir)

    sum_rgb = 0
    sum_rgb_square = 0
    sum_resol = 0
    for img_path in tqdm(list(data_dir.glob(f"""**/*.{ext}"""))):
        pil_img = Image.open(img_path)
        tensor = T.ToTensor()(pil_img)
        
        sum_rgb += tensor.sum(dim=(1, 2))
        sum_rgb_square += (tensor ** 2).sum(dim=(1, 2))
        _, h, w = tensor.shape
        sum_resol += h * w
    mean = torch.round(sum_rgb / sum_resol, decimals=3)
    std = torch.round((sum_rgb_square / sum_resol - mean ** 2) ** 0.5, decimals=3)
    return mean, std


def denorm(tensor, mean, std):
    tensor *= torch.Tensor(std)[None, :, None, None]
    tensor += torch.Tensor(mean)[None, :, None, None]
    return tensor


def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True


def exclude_other_than_gen(old_ckpt_path, new_ckpt_path, device):
    ckpt = torch.load(old_ckpt_path, map_location=device)
    torch.save(ckpt["G"], str(new_ckpt_path))
