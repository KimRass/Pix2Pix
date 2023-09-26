# Source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master

# "For map$aerial photo, we trained on 256 256 resolution images, but exploited fully-convolutional translation (described above) to test on 512   512 images, which were then downsampled and presented to Turkers at 256   256 resolution."


import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

import config
from facades import FacadesDataset


class GoogleMapsDataset(FacadesDataset):
    def __init__(
        self,
        data_dir,
        input_img_mean,
        input_img_std,
        output_img_mean,
        output_img_std,
        split="train",
    ):
        super().__init__(
            data_dir=data_dir,
            input_img_mean=input_img_mean,
            input_img_std=input_img_std,
            output_img_mean=output_img_mean,
            output_img_std=output_img_std,
            split=split,
        )

        self.data_dir = data_dir
        self.input_img_mean = input_img_mean
        self.input_img_std = input_img_std
        self.output_img_mean = output_img_mean
        self.output_img_std = output_img_std
        self.split = split

    def transform(self, input_image, output_image):
        if self.split == "train":
            t, l, h, w = T.RandomCrop.get_params(input_image, output_size=(256, 256))
            input_image = TF.crop(input_image, top=t, left=l, height=h, width=w)
            output_image = TF.crop(output_image, top=t, left=l, height=h, width=w)

            angle = random.randint(0, 3) * 90
            input_image = TF.rotate(input_image, angle=angle)
            output_image = TF.rotate(output_image, angle=angle)

            if random.random() > 0.5:
                input_image = TF.hflip(input_image)
                output_image = TF.hflip(output_image)
        else:
            input_image = TF.center_crop(input_image, output_size=(256, 256))
            output_image = TF.center_crop(output_image, output_size=(256, 256))

        input_image = T.ToTensor()(input_image)
        input_image = T.Normalize(
            mean=self.input_img_mean, std=self.input_img_std,
        )(input_image)

        output_image = T.ToTensor()(output_image)
        output_image = T.Normalize(
            mean=self.output_img_mean, std=self.output_img_std,
        )(output_image)
        return input_image, output_image


if __name__ == "__main__":
    train_ds = GoogleMapsDataset(
        data_dir="/Users/jongbeomkim/Documents/datasets/maps/maps",
        input_img_mean=config.FACADES_INPUT_IMG_MEAN,
        input_img_std=config.FACADES_INPUT_IMG_STD,
        output_img_mean=config.FACADES_OUTPUT_IMG_MEAN,
        output_img_std=config.FACADES_OUTPUT_IMG_STD,
        split="train",
    )
    train_ds[1000]
