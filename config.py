from torch_utils import get_image_dataset_mean_and_std

### Data
# get_image_dataset_mean_and_std("/Users/jongbeomkim/Documents/datasets/archive/trainB", ext="jpg")
# FACADES_INPUT_IMG_MEAN = (0.222, 0.299, 0.745)
# FACADES_INPUT_IMG_STD = (0.346, 0.286, 0.336)
FACADES_INPUT_IMG_MEAN = (0.5, 0.5, 0.5)
FACADES_INPUT_IMG_STD = (0.5, 0.5, 0.5)
# get_image_dataset_mean_and_std("/Users/jongbeomkim/Documents/datasets/archive/trainA", ext="jpg")
# FACADES_OUTPUT_IMG_MEAN = (0.478, 0.453, 0.417)
# FACADES_OUTPUT_IMG_STD = (0.243, 0.235, 0.236)
FACADES_OUTPUT_IMG_MEAN = (0.5, 0.5, 0.5)
FACADES_OUTPUT_IMG_STD = (0.5, 0.5, 0.5)

### Optimizer
# "We use minibatch SGD and apply the Adam solver, with a learning rate of $0.0002$, and momentum
# parameters $\BETA_{1} = 0.5$, $\BETA_{2} = 0.999."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

### Training
N_GEN_EPOCHS = 10 # Generate images every
