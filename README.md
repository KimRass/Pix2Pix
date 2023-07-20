# Paper Summary
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
- We demonstrate that this approach is effective at synthesizing photos from label maps, reconstructing objects from edge maps, and colorizing images, among other tasks.
- If we take a naive approach and ask the CNN to minimize the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring.
- It would be highly desirable if we could instead specify only a high-level goal, like "make the output indistinguishable from reality", and then automatically learn a loss function appropriate for satisfying this goal. Fortunately, this is exactly what is done by the recently proposed Generative Adversarial Networks (GANs) [24]. GANs learn a loss that tries to classify if the output image is real or fake, while simultaneously training a generative model to minimize this loss. Blurry images will not be tolerated since they look obviously fake. Because GANs learn a loss that adapts to the data, they can be applied to a multitude of tasks that traditionally would require very different kinds of loss functions.
- In this paper, we explore GANs in the conditional setting. Just as GANs learn a generative model of data, conditional GANs (cGANs) learn a conditional generative model [24]. This makes cGANs suitable for image-to-image translation tasks, where we condition on an input image and generate a corresponding output image. Our primary contribution is to demonstrate that on a wide variety of problems, conditional GANs produce reasonable results.
## Related Works
- Structured loss
    - Conditional GANs instead learn a structured loss.
## Methodology
- ***GANs are generative models that learn a mapping from random noise vector*** $z$ ***to output image*** $y$***,*** $G : z → y$ ***[24]. In contrast, conditional GANs learn a mapping from observed image*** $x$ ***and random noise vector*** $z$***, to*** $y$***,*** $G : \{x, z\} → y$***. The generator*** $G$ ***is trained to produce outputs that cannot be distinguished from "real" images by an adversarially trained discriminator,*** $D$***, which is trained to do as well as possible at detecting the generator’s "fakes".***
- PatchGANs
    - ***In order to model high-frequencies, it is sufficient to restrict our attention to the structure in local image patches. Therefore, we design a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each $N \times N$ patch in an image is real or fake. We run this discriminator convolutionally across the image, averaging all responses to provide the ultimate output of $D$.***
    - We demonstrate that $N$ can be much smaller than the full size of the image and still produce high quality results. ***This is advantageous because a smaller PatchGAN has fewer parameters, runs faster, and can be applied to arbitrarily large images.*** Such a discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter.
    - ***An advantage of the PatchGAN is that a fixed-size patch discriminator can be applied to arbitrarily large images.*** We may also apply the generator convolutionally, ***on larger images than those on which it was trained.***
## Training
### Loss
- The objective of a conditional GAN can be expressed as
$$\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log(1 − D(x, G(x, z)))]$$
- ***where*** $G$ ***tries to minimize this objective against an adversarial*** $D$ ***that tries to maximize it, i.e.*** $G^{\*} = \arg \min_{G} \max_{D} \mathcal{L}_{cGAN}(G, D)$***.***
- To test the importance of conditioning the discriminator, we also compare to an unconditional variant in which the discriminator does not observe x:
<!-- $$\mathcal{L}_{GAN}(G, D) = \mathbb{E}_{y}[\log D(y)] + \mathbb{E}_{x, z}[\log(1 − D(G(x, z)))]$$ -->
$$\mathcal{L}_{GAN}(G, D) = \mathbb{E}_{y}[\log D(y)] + \mathbb{E}_{z}[\log(1 − D(G(z)))]$$
- Previous approaches have found it beneficial to mix the GAN objective with a more traditional loss, such as L2 distance. The discriminator’s job remains unchanged, but ***the generator is tasked to not only fool the discriminator but also to be near the ground truth output in an L2 sense. We also explore this option, using L1 distance rather than L2 as L1 encourages less blurring:***
$$\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\lVert y - G(x, z) \rVert_{1}]$$
- ***Our final objective is***
$$G^{\*} = \arg \min_{G} \max_{D} \mathcal{L}_{cGAN}(G, D) + \lambda\mathcal{L}_{L1}(G)$$
- Without $z$, the net could still learn a mapping from $x$ to $y$, but would produce deterministic outputs, and therefore fail to match any distribution other than a delta function. Past conditional GANs have acknowledged this and provided Gaussian noise $z$ as an input to the generator, in addition to $x$ (e.g., [55]). In initial experiments, we did not find this strategy effective – the generator simply learned to ignore the noise. Instead, for our final models, we provide noise only in the form of dropout, applied on several layers of our generator at both training and test time. Despite the dropout noise, we observe only minor stochasticity in the output of our nets.
- Figure 4
    - <img src="https://i.imgur.com/kCR21Zy.png" width="800">
    - L1 alone leads to reasonable but blurry results. The cGAN alone (setting $λ = 0$) gives much sharper results but introduces visual artifacts on certain applications. Adding both terms together (with $λ = 100$) reduces these artifacts.
- We also test the effect of removing conditioning from the discriminator (labeled as GAN). In this case, the loss does not penalize mismatch between the input and output; it only cares that the output look realistic. This variant results in poor performance; examining the results reveals that the generator collapsed into producing nearly the exact same output regardless of input photograph.
## Architecture
- We adapt our generator and discriminator architectures from those in [44]. Both generator and discriminator use modules of the form convolution-BatchNorm-ReLu [29].
- Let 'Ck' denote a Convolution-BatchNorm-ReLU layer with k filters. 'CDk' denotes a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of 50%. All convolutions are 4 × 4 spatial filters applied with stride 2. Convolutions in the encoder, and in the discriminator, downsample by a factor of 2, whereas in the decoder they upsample by a factor of 2. (Comment: Stride와 동일한 크기의 Padding을 주어야 합니다.)
- Generator
    - ***Encoder: C64-C128-C256-C512-C512-C512-C512-C512***
    - ***Decoder: CD512-CD512-CD512-C512-C256-C128-C64***
    - ***After the last layer in the decoder, a convolution is applied to map to the number of output channels (3 in general, except in colorization, where it is 2), followed by a Tanh function. As an exception to the above notation, BatchNorm is not applied to the first C64 layer in the encoder. All ReLUs in the encoder are leaky, with slope 0.2, while ReLUs in the decoder are not leaky.***
- Discriminator
    - $70 \times 70$***: C64-C128-C256-C512***
    - ***After the last layer, a convolution is applied to map to a 1-dimensional output, followed by a Sigmoid function. As an exception to the above notation, BatchNorm is not applied to the first C64 layer. All ReLUs are leaky, with slope 0.2***
## Experiments
- PatchGANs
    - We test the effect of varying the patch size $N$ of our discriminator receptive fields, from a $1 \times 1$ "PixelGAN" to a full $286 \times 286$ "ImageGAN". Note that elsewhere in this paper, unless specified, all experiments use $70 \times 70$ PatchGANs.
    - Figure 6
        - <img src="https://i.imgur.com/Ov8FmpA.png" width="800">
        - Uncertain regions become blurry and desaturated under L1. The 1x1 PixelGAN encourages greater color diversity but has no effect on spatial statistics. The 16x16 PatchGAN creates locally sharp results, but also leads to tiling artifacts beyond the scale it can observe. The 70×70 PatchGAN forces outputs that are sharp, even if incorrect, in both the spatial and spectral (colorfulness) dimensions. The full 286×286 ImageGAN produces results that are visually similar to the 70×70 PatchGAN, but somewhat lower quality according to our FCN-score metric.
        - ***The PixelGAN has no effect on spatial sharpness but does increase the colorfulness of the results.*** For example, the bus in Figure 6 is painted gray when the net is trained with an L1 loss, but becomes red with the PixelGAN loss. Using a 16×16 PatchGAN is sufficient to promote sharp outputs, and achieves good FCN-scores, but also leads to tiling artifacts. The 70 × 70 PatchGAN alleviates these artifacts and achieves slightly better scores. 286 × 286 ImageGAN does not appear to improve the visual quality of the results, and in fact gets a considerably lower FCN-score. This may be because the ImageGAN has many more parameters and greater depth than the 70 × 70 PatchGAN, and may be harder to train.
## References
- [24] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [29] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [44] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
- [50] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)