# Random jitter wasapplied byresizing the 256×256input images to 286 × 286 and then randomly cropping back to size 256 × 256. All networks were trained from scratch. Weights were initialized from a Gaussian distribution with mean 0 and standard deviation 0.02.
# "Cityscapes labels→photo 2975 training images from the Cityscapes training set [12], trained for 200 epochs, with random jitter and mirroring. We used the Cityscapes validation set for testing."

import PIL
from PIL import Image
import numpy as np

temp = Image.open("/Users/jongbeomkim/Downloads/gtFine/test/bielefeld/bielefeld_000000_000321_gtFine_color.png")
show_image(np.array(temp))
np.unique()