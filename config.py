### Data
FACADES_LABEL_MEAN = (0.222, 0.299, 0.745)
FACADES_LABEL_STD = (0.346, 0.286, 0.336)

### Optimizer
# "We use minibatch SGD and apply the Adam solver, with a learning rate of $0.0002$, and momentum parameters
# $\BETA_{1} = 0.5$, $\BETA_{2} = 0.999."
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999

### Loss
LAMB = 100

### Training
# "Trained for $200$ epochs."
N_EPOCHS = 400
N_SAVE_EPOCHS = 100 # Save checkpoint every
N_GEN_EPOCHS = 2 # Generate images every
