# References:
    # https://github.com/Seonghoon-Yu/AI_Paper_Review/blob/master/GAN/pix2pix(2016).ipynb
    # https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914

# random jitter and mirroring. Data were split into train and test randomly.

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

from torch_utils import get_image_dataset_mean_and_std

# get_image_dataset_mean_and_std("/Users/jongbeomkim/Documents/datasets/facades/archive/trainB", ext="jpg")
# get_image_dataset_mean_and_std("/Users/jongbeomkim/Documents/datasets/facades/archive/trainA", ext="jpg")


class FacadesDataset(Dataset):
    def __init__(self, archive_dir, split="train"):
        super().__init__()

        self.archive_dir = archive_dir
        self.split = split

        self.label_paths = list(Path(archive_dir).glob(f"""{split}B/*.jpg"""))
    
    def transform(self, label, photo):
        # "Random jitter was applied by resizing the $256 \times 256$ input images to $286 \times 286$
        # and then randomly cropping back to size $256 \times 256$."
        label = TF.resize(label, size=286)
        t, l, h, w = T.RandomCrop.get_params(label, output_size=(256, 256))
        label = TF.crop(label, top=t, left=l, height=h, width=w)
        # "Mirroring"
        p = random.random()
        if p > 0.5:
            label = TF.hflip(label)
        label = T.ToTensor()(label)
        label = T.Normalize((0.222, 0.299, 0.745), (0.346, 0.286, 0.336))(label)

        photo = TF.resize(photo, size=286)
        photo = TF.crop(photo, top=t, left=l, height=h, width=w)
        if p > 0.5:
            photo = TF.hflip(photo)
        photo = T.ToTensor()(photo)
        photo = T.Normalize((0.478, 0.453, 0.417), (0.243, 0.235, 0.236))(photo)
        return label, photo

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        photo_path = str(label_path).replace(
            "/archive/trainB/", "/archive/trainA/"
        ).replace("_B.jpg", "_A.jpg")

        label = Image.open(label_path)
        photo = Image.open(photo_path).convert("RGB")
        if self.split == "train":
            label, photo = self.transform(label, photo)
        return label, photo

    def __len__(self):
        return len(self.label_paths)


def get_facades_dataloader(archive_dir, batch_size, n_workers, split):
    ds = FacadesDataset(archive_dir, split=split)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=n_workers, drop_last=True)
    return dl
