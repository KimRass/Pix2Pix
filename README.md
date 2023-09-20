# Paper Reading
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
## Training
### Loss
- The objective of a conditional GAN can be expressed as
$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$$
- Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance. The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:
$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\Vert y - G(x, z) \Vert_{1}]$$
- Without $z$, the net could still learn a mapping from $x$ to $y$, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise $z$ as an input to the generator, in addition to $x$ (e.g., [55]). In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise. Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time. Despite the dropout noise, we observe only minor stochasticity in the output of our nets.
## References
- [24] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [29] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [44] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
- [50] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

# Pre-trained models
- [pix2pix2_facades.pth](https://drive.google.com/file/d/1SPhUPA5ms4MDCuSj0bnM_Q4y8yUVOj82/view?usp=sharing)
    - Traind on Facades dataset for 789 epochs with $\lambda = 85$.