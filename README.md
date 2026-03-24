# pixel-NeRF를 활용한 차량 이미지 증강 및 모델 최적화

> Research on 3D Reconstruction and View Synthesis in Google Colab

[](https://colab.research.google.com/github/JY-maru/pixelNeRF-SSU/blob/main/infer.ipynb)

본 프로젝트는 Pixel-NeRF (Yu et al., 2021) 아키텍처를 기반으로, 제한된 컴퓨팅 환경인 Google Colab 내에서 다중 뷰 피처 병합 전략과 3D 공간 기하학 분석을 통한 데이터 파이프라인 최적화를 수행한 모델 고도화 프로젝트입니다.

기존 모델이 요구하는 방대한 연산량과 렌더링 병목을 수학적 기법과 구조 개선으로 해결하였으며, 자율주행 및 CV 분야의 차량 데이터 증강 파이프라인으로 실제 활용될 수 있도록 객체 복원 디테일과 학습 효율을 높이고자 했습니다.

<br>

## Demo Results

<p align="center">
<img src="https://raw.githubusercontent.com/JY-maru/pixelNeRF-SSU/main/images/rendering_demo.gif" alt="Demo Results" width="800">
</p>

<br>

-----

## Context & Optimization Strategy

본 프로젝트의 핵심 목표는 자율주행 및 컴퓨터 비전 모델의 학습 데이터 확보를 위한 고품질 차량 데이터 증강입니다. 현실 세계에서 차량의 모든 각도 데이터를 수집하는 것은 막대한 비용이 소요되므로, 소수의 차량 이미지(3-views)만으로도 정교한 3D 형상을 복원하고 다양한 각도의 새로운 시점 데이터를 생성하는 것을 목표로 합니다. 이를 위해 제한된 클라우드 자원 하에서도 최상의 기하학적 퀄리티를 확보하고자 다음과 같은 트러블슈팅 및 최적화 전략을 수행했습니다.

### 1.  Geometric Data Filtering : 좌표계 분석 및 정제
- 데이터의 카메라 좌표계를 고려치 않고 모델 파인 튜닝 시 소스 뷰에 대한 타겟 뷰가 생성되지 않는 문제를 겪었습니다. 또한 학습 데이터인 ShapeNet 원본 데이터에는 실제 차량 인식 환경에서 불필요한 바닥면이나 수직 하강 뷰가 다수 포함되어 있습니다. 적은 수의 이미지로 다중 차량 이미지를 증강한다는 목표를 달성하기에는 이러한 노이즈 데이터가 모델 수렴을 심각하게 지연시켰습니다.
- **해결** : openGL 방식으로 진행 시 카메라의 뒤를 방향으로 바라보고 학습하게 되고, openCV방식은 카메라 전방 방면으로 학습을 진행합니다. 카메라 좌표계가 OpenCV 기준인지 분석하고 모델에 맞게 변환 행렬을 수정했습니다. 이후 카메라의 고도각을 동적으로 계산하여, 사용자 시나리오에 부합하는 0도에서 60도 사이의 데이터만 전처리 및 학습에 사용되도록 필터링 로직을 구축했습니다.

### 2.  Early Fusion with Variance : 학습 안정화 및 피처 병합 최적화 
- Colab 환경의 한계로 인해 배치 크기와 레이어 수를 줄여야 했고, 네트워크 매 층마다 피처를 전달하는 기존 방식은 차량 이미지의 형태 수렴보다는 배경과 객체의 구분을 모호하게 만들어 차량 객체의 경계가 희미해지는 현상이 지속되는 등 학습 초반의 불안정성을 키웠습니다.
- **해결** : 학습 전체 시간을 줄이고 적은 연산량으로도 빠른 수렴을 유도하기 위해 Average Pooling과 분산 기반의 Early Fusion 전략을 도입했습니다. 분산값을 통해 특정 각도에서 가려진 물체에 낮은 가중치를 부여하게 만들어, 초기 5만 스텝 기준 13.65 PSNR에서 18.61 PSNR로 개선된 수렴 속도를 확보했습니다.

### 3.  Feature Pyramid Network : 해상도 한계 극복 및 디테일 보존
- Coarse/Fine 네트워크를 거치며 차량 이미지에 대한 피처, 특히 RGB 값이 끝까지 전달되지 않아 렌더링 결과물이 흑백으로 나오는 현상이 지속되었습니다. 또한 128x128 저해상도 학습으로 인해 차량 문, 바퀴, 사이드미러 등 핵심 디테일이 소실되었습니다.
- **해결** : 단순 피처 전달을 넘어 FPN 구조를 도입하여 다중 해상도의 피처맵을 추출하고 MLP 네트워크 깊은 곳까지 1차원 벡터 형태로 전달되도록 입력 벡터의 차원을 늘려 학습을 진행합니다. 밀도($\sigma$)와 RGB가 MLP 입력에 명확히 전달되도록 Coarse/Fine 각 네트워크에서 입력 차원을 늘려 학습한 결과 초기 학습 시 흑백 현상이 해결되었고, 2번의 빠른 차량 형태 수렴 성과와 함께 밀도 + 색상 학습을 보다 안정적으로 수행하게 하였습니다. 그 결과 적은 파라미터 수로도 미세한 디테일을 살려내며 최종 21.16 PSNR을 달성했습니다.

| 구분 | Original Pixel-NeRF | Optimized Model (Ours) |
| :--- | :--- | :--- |
| Objective | General 3D Reconstruction | Robust Vehicle Data Generation for CV |
| Input Views | 1 \~ 3 Views (Sparse) | 3 Views (Geometric Constraints) |
| Environment | Heavy Workstation | Google Colab (Resource Constrained) |
| Feature Fusion | Average Pooling | Early Fusion (Average + Variance) & FPN |
| Data Control | All Angles | Elevation Filtered (0도\~60도) |

<br>

-----

## Project Structure

```text
pixelNeRF-SSU/
├── config/               # Model 파라미터 설정 파일 (.yaml)
├── data/                 # 데이터 로더 및 기하학적 필터링 로직
├── model/                # PixelNeRF, encoder, Early Fusion 모듈 소스 코드
├── utils/                # Projection, 렌더링 관련 유틸리티 함수
├── train.py              # 학습용 소스코드 
├── inference.py          # novel-view 생성 관련 추론용 소스코드 
└── fetch2local.sh        # 데이터 다운로드 스크립트
```

## Dataset Details

본 프로젝트는 3D 객체 인식 및 복원 분야의 표준 벤치마크인 ShapeNet Core V2 데이터셋을 기반으로 합니다. 자율주행 환경 시뮬레이션이라는 목적에 맞춰, Cars 카테고리를 선별하여 학습을 진행합니다.

  * Dataset Source: ShapeNet Core V2 (Cars Category)
  * Target Object: Vehicles
  * Data Format: 각 3D 객체에 대해 사전 렌더링된 다각도 RGB 이미지와 Camera Pose 정보

<br>

-----

## Model Architecture & Pipeline

제한된 환경에서 다중 뷰 정보를 효과적으로 처리하기 위해 모델 아키텍처를 FPN + Stereo Matching 원리에 입각하여 고도화했습니다. 전체 파이프라인은 아래의 순서로 진행됩니다.

### 1. Multi-Scale Feature Extraction (FPN)
기존 ResNet의 단일 레이어 특징맵만 사용할 경우 발생하는 정보 손실을 막기 위해 FPN을 도입했습니다.

<p align="center">
<img src="https://raw.githubusercontent.com/JY-maru/pixelNeRF-SSU/main/images/1-2.encoder(FPN).jpg" alt="FPN structure" width="800">
</p>

- 구조: ResNet Backbone을 통해 4가지 해상도의 특징맵과 원본 RGB를 추출합니다.
- 효과: Global Shape와 Fine Detail을 동시에 학습하여 디테일한 복원이 가능합니다.

### 2. World-to-Pixel Projection & Feature Fetching
타겟 뷰의 픽셀에 대응하는 3D 좌표를 소스 뷰로 투영하여 특징을 추출하는 과정입니다.

<p align="center">
<img src="https://raw.githubusercontent.com/JY-maru/pixelNeRF-SSU/main/images/1-1.world2source.jpg" alt="World to Source Projection" width="800">
</p>

- Target Ray 위의 3D 샘플 포인트들을 Source View의 2D 평면으로 투영합니다. 투영된 위치에서 FPN으로 추출한 Multi-scale Feature를 가져옵니다.

### 3. Early Fusion with Variance
여러 뷰에서 가져온 특징들을 합치는 과정에서, 단순 평균 뿐만 아니라 분산 정보를 추가하고 손실을 막기 위해 Feature를 재 주입합니다.

<p align="center">
<img src="https://raw.githubusercontent.com/JY-maru/pixelNeRF-SSU/main/images/1-3.mean%2Bvar.jpg" alt="Mean and Variance Fusion" width="800">
</p>

분산은 여러 카메라가 동일한 색상 및 특징을 보고 있는가를 나타내는 Stereo Cue입니다.
- Low Variance: 실제 물체 표면일 확률이 높음.
- High Variance: 허공이거나 다른 객체에 의해 가려진 영역으로 판단하여 가중치 저하.

### 4. Volume Rendering Pipeline
추출된 특징들은 Coarse/Fine MLP를 거쳐 밀도와 색상으로 변환되며, 이를 Volume Rendering 적분을 통해 최종 픽셀 색상으로 합성합니다.

<p align="center">
<img src="https://raw.githubusercontent.com/JY-maru/pixelNeRF-SSU/main/images/1-4.volume_rendering.jpg" alt="Volume Rendering Pipeline" width="800">
</p>

최종 픽셀의 색상은 광선 상의 모든 샘플 포인트의 기여도를 합산하여 계산됩니다.

$$
\hat{C} = \sum_{i} T_i \cdot (1 - e^{-\sigma_i \delta_i}) \cdot c_i
$$

- $T_i$ (도달 확률): 광선이 i번째 지점까지 장애물 없이 도달할 확률.
- $(1 - e^{-\sigma_i \delta_i})$ (불투명도): 해당 구간에서 입자가 존재하여 광선이 부딪힐 확률 (여기서 밀도 기호는 NeRF 학계 표준에 따라 $\sigma$를 사용합니다).
- $c_i$ (색상): 해당 지점의 RGB 색상.

### 5. Training Strategy (Coarse-to-Fine)
렌더링된 이미지는 Ground Truth 이미지와 비교되어 학습됩니다.

<p align="center">
<img src="https://raw.githubusercontent.com/JY-maru/pixelNeRF-SSU/main/images/1-5.pred_GT.jpg" alt="Training and Loss" width="800">
</p>

- Coarse Pass: 전체 영역을 균일하게 샘플링하여 대략적인 형상을 파악.
- Fine Pass: Coarse 단계에서 물체가 있을 확률이 높은 곳을 집중적으로 샘플링하여 디테일 보정.
- Loss: Coarse와 Fine 출력 모두에 대해 MSE Loss를 계산하여 최적화를 수행합니다.

<br>

-----

## Model Performance

본 모델은 제한된 클라우드 컴퓨팅 환경 내에서 위와 같은 파이프라인 최적화를 거쳐 30시간의 학습으로 아래와 같은 정량적 수치를 기록했습니다. 기하학적 필터링과 FPN + Stereo Matching 아키텍처를 결합한 결과, 128x128 해상도에서 평균 PSNR 21.30, 256x256 해상도에서 평균 PSNR 24.50 dB (SSIM 0.9649)를 기록했습니다. 이는 기존 베이스라인 대비 차량 휠, 측면 도어 등 세부 형상의 렌더링 손실을 줄여 Demo결과처럼 다양한 각도 View의 객체 디테일을 보존하게 되었습니다.

| Input Views | Resolution | Training Steps | Feature Fusion | PSNR (dB) | SSIM |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 4 Views | 128x128 | 120,000 | Early Fusion + FPN | 21.30 | 0.9523 |
| 4 Views | 256x256 | 120,000 | Early Fusion + FPN | 24.50 | 0.9649 |

<br>

-----

## Getting Started

이 코드는 Google Colab 환경에 최적화되어 있습니다. 데이터 로드부터 학습까지 간편하게 실행 가능합니다.

### 1\. Data Loading

  - Google Drive 공유 폴더 활용

<!-- end list -->

1.  ShapeNet Cars Dataset 폴더에 접속합니다.
2.  자신의 드라이브에 추가한 뒤, 데이터 경로를 저장된 경로로 지정하여 사용하면 됩니다. 필터링이 자동으로 적용됩니다.

### 2\. Training

최초 실행 시 기하학적 필터링을 위한 캐시 생성으로 인해 시작에 약 5에서 10분이 소요될 수 있습니다.

> Argv Guide

  - config/default\_config.yaml 파일에서 주요 학습 파라미터를 수정할 수 있습니다.
  - \-- resume : 설정한 가중치부터 학습 재개
  - \-- config: 지정한 config파일로 학습

<!-- end list -->

```bash
 python train.py --config config/default_config.yaml
```

### 3\. Inference (Video Generation)

> Argv Guide

  - \--mode : views 또는 video
  - \--size: 모델 해상도 지정
  - \--num\_frames: 생성 이미지 수

<!-- end list -->

```bash
!python -u inference.py --input_folder <INPUT_ROOT> \
                        --mode video \					
                        --output_dir ./outputs \
                        --size 256 \
                        --num_frames 120 \
                        --obj_id <selected_obj_id> \
                        --n_fine 256
```

<br>

-----

## Acknowledgement & Citation

This project builds upon the official implementation of [PixelNeRF](https://github.com/sxyu/pixel-nerf). We optimized it for constrained environments by introducing FPN encoders, Variance-based Early Fusion, and Geometric data pruning.

If you use this code for your research, please cite the original Pixel-NeRF paper:

```bibtex
@inproceedings{yu2021pixelnerf,
  title={pixelNeRF: Neural Radiance Fields from One or Few Images},
  author={Yu, Alex and Ye, Vickie and Tancik, Matthew and Kanazawa, Angjoo},
  booktitle={CVPR},
  year={2021}
}
```

> License
> This project is released under the MIT License.
