# References:
    # https://github.com/Seonghoon-Yu/AI_Paper_Review/blob/master/GAN/pix2pix(2016).ipynb
    # https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914

# "Data were split into train and test randomly."

from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import random

from torch_utils import get_image_dataset_mean_and_std

# get_image_dataset_mean_and_std("/Users/jongbeomkim/Documents/datasets/archive/trainA", ext="jpg")
# get_image_dataset_mean_and_std("/Users/jongbeomkim/Documents/datasets/archive/trainB", ext="jpg")


class FacadesDataset(Dataset):
    def __init__(self, data_dir, label_mean, label_std, photo_mean, photo_std, split="train", ):
        super().__init__()

        self.data_dir = data_dir
        self.label_mean = label_mean
        self.label_std = label_std
        self.photo_mean = photo_mean
        self.photo_std = photo_std
        self.split = split

        self.label_paths = list(Path(data_dir).glob(f"""{split}B/*.jpg"""))
    
    def transform(self, label, image):
        # "Random jitter was applied by resizing the $256 \times 256$ input images to $286 \times 286$
        # and then randomly cropping back to size $256 \times 256$."
        label = TF.resize(label, size=286)
        image = TF.resize(image, size=286)

        t, l, h, w = T.RandomCrop.get_params(label, output_size=(256, 256))
        label = TF.crop(label, top=t, left=l, height=h, width=w)
        image = TF.crop(image, top=t, left=l, height=h, width=w)

        # "Mirroring"
        p = random.random()
        if p > 0.5:
            label = TF.hflip(label)
            image = TF.hflip(image)

        label = T.ToTensor()(label)
        label = T.Normalize(mean=self.label_mean, std=self.label_std)(label)

        image = T.ToTensor()(image)
        image = T.Normalize(mean=self.photo_mean, std=self.photo_std)(image)
        return label, image

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        label_path = self.label_paths[idx]
        image_path = str(label_path).replace("/trainB/", "/trainA/")
        image_path = image_path.replace("_B.jpg", "_A.jpg")

        label = Image.open(label_path).convert("RGB")
        image = Image.open(image_path).convert("RGB")
        if self.split == "train":
            label, image = self.transform(label, image)
        return label, image
