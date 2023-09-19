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
    def __init__(
        self,
        data_dir,
        input_img_mean,
        input_img_std,
        output_img_mean,
        output_img_std,
        split="train",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.input_img_mean = input_img_mean
        self.input_img_std = input_img_std
        self.output_img_mean = output_img_mean
        self.output_img_std = output_img_std
        self.split = split

        self.input_img_paths = list(Path(data_dir).glob(f"""{split}B/*.jpg"""))
    
    def transform(self, input_image, output_image):
        # "Random jitter was applied by resizing the $256 \times 256$ input images to
        # $286 \times 286$ and then randomly cropping back to size $256 \times 256$."
        input_image = TF.resize(input_image, size=286)
        output_image = TF.resize(output_image, size=286)

        t, l, h, w = T.RandomCrop.get_params(input_image, output_size=(256, 256))
        input_image = TF.crop(input_image, top=t, left=l, height=h, width=w)
        output_image = TF.crop(output_image, top=t, left=l, height=h, width=w)

        # "Mirroring"
        p = random.random()
        if p > 0.5:
            input_image = TF.hflip(input_image)
            output_image = TF.hflip(output_image)

        input_image = T.ToTensor()(input_image)
        input_image = T.Normalize(
            mean=self.input_img_mean, std=self.input_img_std,
        )(input_image)

        output_image = T.ToTensor()(output_image)
        output_image = T.Normalize(
            mean=self.output_img_mean, std=self.output_img_std,
        )(output_image)
        return input_image, output_image

    def __len__(self):
        return len(self.input_img_paths)

    def __getitem__(self, idx):
        input_img_path = self.input_img_paths[idx]
        output_img_path = str(input_img_path).replace("/trainB/", "/trainA/")
        output_img_path = output_img_path.replace("_B.jpg", "_A.jpg")

        input_image = Image.open(input_img_path).convert("RGB")
        output_image = Image.open(output_img_path).convert("RGB")
        if self.split == "train":
            input_image, output_image = self.transform(input_image, output_image)
        return input_image, output_image
