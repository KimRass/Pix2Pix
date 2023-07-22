# Paper Reading
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
## Methodology
- PatchGANs
    - In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each $N \times N$ patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of $D$.
    - We demonstrate that $N$ can be much smaller than the full size of the image and still produce high quality results. This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images. Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter.
    - An advantage of the PatchGAN is that a fixed-size patch discriminator can be applied to arbitrarily large images. We may also apply the generator convolutionally, on larger images than those on which it was trained.
## Training
### Loss
- The objective of a conditional GAN can be expressed as
$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$$
- Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance. The discriminator’s job remains unchanged, but the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:
$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\lVert y - G(x, z) \rVert_{1}]$$
- Without $z$, the net could still learn a mapping from $x$ to $y$, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise $z$ as an input to the generator, in addition to $x$ (e.g., [55]). In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise. Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time. Despite the dropout noise, we observe only minor stochasticity in the output of our nets.
## Experiments
- PatchGANs
    - We test the effect of varying the patch size $N$ of our discriminator receptive fields, from a $1 \times 1$ "PixelGAN" to a full $286 \times 286$ "ImageGAN". Note that elsewhere in this paper, unless specified, all experiments use $70 \times 70$ PatchGANs.
    - Figure 6
        - <img src="https://i.imgur.com/Ov8FmpA.png" width="800">
        - Uncertain regions become blurry and desaturated under L1. The 1x1 PixelGAN encourages greater color diversity but has no effect on spatial statistics. The 16x16 PatchGAN creates locally sharp results, but also leads to tiling artifacts beyond the scale it can observe. The 70×70 PatchGAN forces outputs that are sharp, even if incorrect, in both the spatial and spectral (colorfulness) dimensions. The full 286×286 ImageGAN produces results that are visually similar to the 70×70 PatchGAN, but somewhat lower quality according to our FCN-score metric.
        - The PixelGAN has no effect on spatial sharpness but does increase the colorfulness of the results. For example, the bus in Figure 6 is painted gray when the net is trained with an L1 loss, but becomes red with the PixelGAN loss. Using a 16×16 PatchGAN is sufficient to promote sharp outputs, and achieves good FCN-scores, but also leads to tiling artifacts. The 70 × 70 PatchGAN alleviates these artifacts and achieves slightly better scores. 286 × 286 ImageGAN does not appear to improve the visual quality of the results, and in fact gets a considerably lower FCN-score. This may be because the ImageGAN has many more parameters and greater depth than the 70 × 70 PatchGAN, and may be harder to train.
## References
- [24] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [29] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [44] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
- [50] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
