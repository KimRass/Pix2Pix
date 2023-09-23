### Data
### Facades dataset
FACADES_INPUT_IMG_MEAN = (0.5, 0.5, 0.5)
FACADES_INPUT_IMG_STD = (0.5, 0.5, 0.5)
FACADES_OUTPUT_IMG_MEAN = (0.5, 0.5, 0.5)
FACADES_OUTPUT_IMG_STD = (0.5, 0.5, 0.5)
### Google maps dataset
GOOGLEMAPS_INPUT_IMG_MEAN = (0.5, 0.5, 0.5)
GOOGLEMAPS_INPUT_IMG_STD = (0.5, 0.5, 0.5)
GOOGLEMAPS_OUTPUT_IMG_MEAN = (0.5, 0.5, 0.5)
GOOGLEMAPS_OUTPUT_IMG_STD = (0.5, 0.5, 0.5)

### Optimizer
# "We use minibatch SGD and apply the Adam solver, with a learning rate of $0.0002$, and momentum
# parameters $\BETA_{1} = 0.5$, $\BETA_{2} = 0.999."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

### Training
N_GEN_EPOCHS = 10 # Generate images every
