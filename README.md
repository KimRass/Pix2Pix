- [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/KimRass/Pix2Pix/blob/main/papers/Pix2Pix)

## 1. Pre-trained Models
|||
|-|-|
| Trained on Facades for 200 epochs | [pix2pix_facades.pth](https://drive.google.com/file/d/1sSro8prPTV5MddkFohaiIqdznreAnAyU/view?usp=sharing) |
| Trained on Google Maps for 400 epochs | [pix2pix_google_maps.pth](https://drive.google.com/file/d/1_mt4K-0Z2x1DxA0f2om9VaAEFamMfROU/view?usp=sharing) |

# 2. Sampling
- [Test set of Facades dataset](https://github.com/KimRass/pix2pix_from_scratch/blob/main/generated_images/facades_test_set/)
    - <img src="https://github.com/KimRass/Pix2Pix/assets/105417680/9ec992f1-46c6-4c1f-bdb0-6ca728a9a053" width="400">
- [Test set of Google maps dataset](https://github.com/KimRass/pix2pix_from_scratch/blob/main/generated_images/google_maps_test_set/)
    - <img src="https://github.com/KimRass/Pix2Pix/assets/105417680/46470114-54ec-4652-aac1-1986c4d6cc18" width="400">

# 3. Implementation Details
## 1) Image Mean and STD
- Facades dataset의 Training set에 대해 Input image와 Output image 각각에 대해 Mean과 STD를 계산하면 다음과 같습니다.
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
- 두 가지 Settings를 가지고 실험을 해 본 결과, 후자의 학습 속도가 전자보다 빨랐습니다.
## 2) Architecture
- `self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)`로 설정 시 다음과 같이 모델이 생성한 이미지가 다음과 같이 Blurry했습니다.
    - <img src="https://github.com/KimRass/Pix2Pix/assets/105417680/bf2e3871-2c73-4b2c-9c99-23b67113c588" width="400">
- `self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)`로 수정하자 이런 현상이 없어졌습니다.
- Instance norm은 원래 기본적으로 `track_running_stats=False`을 사용합니다.
<!-- - 그 이유는 학습 시 Batch size를 1로 하므로 제대로 된 Mini batch의 통계량을 학습하지 못했기 때문이라고 생각합니다.
- 논문에서도 에측 시에 학습 시와 동이하게 Mini batch에 대해서 Normalize한다고 했으므로 이와 합치합니다. -->
