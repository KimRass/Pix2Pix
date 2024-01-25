# 1. Introduction
- 이미지를 하나의 Representation에서 다른 Representation으로 번역하는 Task인 Image-to-image translation을 제안합니다.
- L2 distance만을 최소화하도록 $G$를 학습시키면, $G$의 출력 전부에 대해서 L2 distance의 평균이 최소화되므로, 출력되는 이미지가 Blurry한 문제가 있습니다. 따라서 좋은 품질의 이미지를 생성하기 위한 Loss function을 설계하는 것은 일반적으로 전문 지식을 요구합니다.
- GANs를 사용하면 Blurry한 $G$의 출력은 Discriminator에게 명백히 가짜로 보이므로 Blurry 이미지가 생성될 수 없습니다. GANs를 사용함으로써 기존 방식으로는 매우 어려운 수준의 Loss function을 사용하지 않고도 많은 Image-to-image translation task에 적용이 가능합니다.
- cGANs (Conditional GANs)를 사용합니다.

# 2. Related Work
- Generator에는 U-Net-based architecture를, Discriminator에는 PatchGAN을 사용한다는 점에서 기존 연구들과 차이점이 있습니다.

# 3. Method
## 3.1) Objective
- Generator $G$, Noise vector $z$, Ground truth image $y$에 대해서 GANs는
$$G: z \rightarrow y$$
- 그러나 cGANs는 Observed image $x$에 대해서
$$G: \{x, z\} \rightarrow y$$
- 그러나 이 논문에서는
$$G: x \rightarrow y$$
- 즉 $G$가 $z$를 입력으로 받지 않습니다. Stochasticity를 위해 Dropout layer를 $G$에 여러 번 집어넣음으로써 Noise를 주입했지만 Stochasticity가 매우 적었습니다. 따라서 높은 Stochasticity를 실현하는 것이 향후 과제로 남아있습니다.
- $G$는 Discriminator $D$를 속이는 한편 $G$의 출력과 $y$ 사이의 L1 distance 또한 작도록 학습이 이루어집니다.
## 3.2) Network Architecture
- 'DCGAN'을 참고하여 Conv → Batch norm → ReLU 형태의 모듈을 사용했습니다.
## (3.2.1) Generator with Skips
- Image-to-image translation task에서는 쌍을 이루는 두 이미지가 low-level features를 서로 공유하는 경우가 많으므로 이 Features를 $G$의 입력에서 출력으로 바로 전달하는 것이 바람직합니다. 따라서 Skip connection을 사용한 U-Net architecture를 사용합니다.
## (3.2.2) Markovian Discriminator (PatchGAN)
- L1 loss와 L2 loss는 Blurry 이미지를 생성하도롥 함이 알려져 있습니다. 이 말은 이 두 Loss functions가 High-level이 아닌 Low-level features만을 캡쳐함을 의미합니다. 따라서 $D$가 High-level features를 캡쳐하도록만 강제하면 되는데, 이를 위해서는 $D$가 Local image patches의 구조에만 집중하도록 하면 됩니다. PatchGAN은 각각의 Patches가 진짜인지 가짜인지만을 분류하고자 작동합니다.
- Comment: Receptive field 4 → 7 → 16 → 34 → 70
## 3.3) Optimization and Inference
- $D$를 업데이트할 때 $D$의 Loss를 2로 나누어 $D$가 학습하는 속도를 늦춥니다.
- 예측 시에도 $G$에 Dropout를 적용합니다.
- 학습 시와 예측 시에 Batch norm을 동일한 방식으로 사용합니다. 즉 예측 시에도 Mini batch의 통계량을 사용합니다. (`nn.BatchNorm2d(track_running_stats=False)`)

# 4. Experiments
## 4.1) Evaluation Metrics
- For Google Maps dataset:
    - 사람이 직접 사진을 보고 진짜인지 가짜인지 판별하도록 하는 방식으로 $G$를 평가했습니다.
    - 256 × 256 해상도로 $G$를 학습시킨 후 512 × 512 해상도의 이미지를 생성시킨 후 다시 256 × 256으로 Downsample했습니다.
- For image colorization task:
    - Google Maps dataset과 동일한 방식으로 $G$를 평가했습니다.
    - 학습과 예측 시에 모두 256 × 256 해상도의 이미지를 사용했습니다.
- For Cityscapes dataset:
    - 우리의 $G$이 생성한 이미지가 Realistic하다면, 실제 이미지에 대해서 학습된 다른 $G$이 이 이미지에 대해서도 잘 작동할 것이라는 직관을 바탕으로 합니다.
    - 'FCN-8s'이라는 Semantic segmentation을 위한 $G$를 Cityscapes dataaset의 실제 이미지에 대해서 학습시킨 후, 우리의 $G$이 생성한 이미지에 대해서 예측을 시킴으로써 우리의 $G$를 평가했습니다.
## 4.2) Analysis of The Objective Function
- For Facades dataset:
    - L1 loss만 사용 시 Blurry 이미지를 생성합니다.
    - cGAN loss만을 사용 시 Blurry하지 않지만 Artifacts가 생성되기도 합니다.
    - 두 가지 Losses를 모두 사용 시 Artifacts가 사라집니다.
- For Cityscapes dataset:
    - $D$에게 $G$ 생성한 이미지만을 입력하고 $G$가 입력으로 받은 이미지는 입력하지 않은 경우, $D$는 $G$의 실제와 같은지만을 판단하게 되므로 입력에 관계 없이 $G$는 실제처럼 보이는 하나의 이미지만을 생성했습니다 (Mode colapse).
- Colorfuless
    - cGANs의 경우 $G$가 Source image에 없는 공간적인 구조까지도 생성하는 Hallucination이 발생합니다.
    - L1 loss는 $G$가 평균적이고 무채색에 가까운 이미지를 생성하도록 하고 cGANs loss는 Ground truth에 가까운 이미지를 생성하도록 합니다.
## 4.3) Analysis of the generator architecture
- Skip connection을 사용한 $G$가 좀 더 사실적인 이미지를 생성합니다.
## 4.4) FromPixelGANs to PatchGANs to ImageGANs
- 1 × 1 'PixelGAN': Blurry하며 Colorful합니다.
- 16 × 16 PatchGAN: Tiling artifacts가 발생합니다.
- 286 × 286 'ImageGAN':
    - 정성적으로 볼 때도 70 × 70 PatchGAN보다 나은 점이 없으며 정량적으로도 성능이 낮습니다.
    - Parameters의 수도 많고 Architecture가 더 깊으므로 학습시키기 어렵기 때문으로 추정합니다.
- $D$는 PatchGAN이므로 그리고 $G$는 Fully-convolutional이므로 어떤 해상도의 이미지에도 적용 가능합니다.
## 4.5) Perceptual Validation
- Map → Aerial photograph의 경우보다 Aerial photograph → Map의 경우에 $G$가 생성한 이미지를 평가자들이 더 적은 횟수로 진짜라고 생각했습니다. Map 이미지가 직선 등 기하하적인 시각 요소가 더 두드러지게 보여 평가자들이 판단을 내리기 더 쉬웠을 것으로 추정합니다.
