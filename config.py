### Data
FACADES_INPUT_IMG_MEAN = (0.222, 0.299, 0.745)
FACADES_INPUT_IMG_STD = (0.346, 0.286, 0.336)
FACADES_OUTPUT_IMG_MEAN = (0.478, 0.453, 0.417)
FACADES_OUTPUT_IMG_STD = (0.243, 0.235, 0.236)

### Optimizer
# "We use minibatch SGD and apply the Adam solver, with a learning rate of $0.0002$, and momentum parameters
# $\BETA_{1} = 0.5$, $\BETA_{2} = 0.999."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

### Training
N_GEN_EPOCHS = 10 # Generate images every


import torch
from einops import rearrange
input1 = torch.full((3, 3), 1)
input2 = torch.full((3, 3), 2)
input3 = torch.full((3, 3), 3)

out = torch.concat((input1, input2, input3)).T.flatten()
torch.concat((input1, input2, input3)).shape, out.shape
torch.concat((input1, input2, input3))
torch.concat((input1, input2, input3)).T
out
torch.stack(torch.split(out, 3), dim=1).reshape(3,-1)


input_image = torch.full(size=(2, 3, 4, 4), fill_value=0)
real_output_image = torch.full(size=(2, 3, 4, 4), fill_value=1)
fake_output_image = torch.full(size=(2, 3, 4, 4), fill_value=2)
concat = torch.cat([input_image, real_output_image, fake_output_image], dim=0)
out = rearrange(concat, pattern="(n m) c h w -> (m n) c h w", n=3)
out.shape
out[0].sum(), out[1].sum(), out[2].sum()
out[3].sum(), out[4].sum(), out[5].sum()


image = torch.stack(torch.split(out, 3), dim=1)
