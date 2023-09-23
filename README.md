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

# Pre-trained Model
- [pix2pix2_facades.pth](https://drive.google.com/file/d/1SPhUPA5ms4MDCuSj0bnM_Q4y8yUVOj82/view?usp=sharing)
    - Trained on Facades dataset for 789 epochs with $\lambda = 85$.

# Generated Images

# Researches
## Convergence
- 논문에서는 Facades dataset에 대해서 200 epochs만을 학습시켰지만, 그것만으로는 모델이 충분히 수렴하지 않았습니다. 특히 G.T. output image의 다양한 색깔을 제대로 구현하지 못했습니다.
<!-- ## Image Mean and Standard Deviation
- Facades dataset의 train set에 대해 input images와 output images 각각에 대해 mean과 standard deviation을 계산하면 다음과 같습니다.
    ```python
    FACADES_INPUT_IMG_MEAN = (0.222, 0.299, 0.745)
    FACADES_INPUT_IMG_STD = (0.346, 0.286, 0.336)
    FACADES_OUTPUT_IMG_MEAN = (0.478, 0.453, 0.417)
    FACADES_OUTPUT_IMG_STD = (0.243, 0.235, 0.236)
    ```
- 반면 다음과 같이 설정하면 모델에 입력되는 모든 tensors의 값이 $[-1, 1]$의 값을 갖게 됩니다.
    ```python
    FACADES_INPUT_IMG_MEAN = (0.5, 0.5, 0.5)
    FACADES_INPUT_IMG_STD = (0.5, 0.5, 0.5)
    FACADES_OUTPUT_IMG_MEAN = (0.5, 0.5, 0.5)
    FACADES_OUTPUT_IMG_STD = (0.5, 0.5, 0.5)
    ```
- 두 가지 settings를 가지고 실험을 해 본 결과, 후자의 학습 속도가 전자보다 훨씬 빨랐으며 loss의 크기도 더 작았습니다. -->
