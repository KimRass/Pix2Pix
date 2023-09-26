# Paper Reading
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
## References
- [24] [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [29] [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
- [44] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)
- [50] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

# Pre-trained Models
## Trained on Facades Dataset for 200 epochs
- [pix2pix_facades.pth](https://drive.google.com/file/d/1sSro8prPTV5MddkFohaiIqdznreAnAyU/view?usp=sharing)
## Trained on Google Maps Dataset for 400 epochs
- [pix2pix_google_maps.pth](https://drive.google.com/file/d/1_mt4K-0Z2x1DxA0f2om9VaAEFamMfROU/view?usp=sharing)

# Generated Images
- [Test set of Facades dataset](https://github.com/KimRass/pix2pix_from_scratch/blob/main/generated_images/facades_test_set/)
- [Test set of Google maps dataset](https://github.com/KimRass/pix2pix_from_scratch/blob/main/generated_images/google_maps_test_set/)

# Researches
## Image Mean and Standard Deviation
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
- 두 가지 settings를 가지고 실험을 해 본 결과, 후자의 학습 속도가 전자보다 빨랐습니다.
