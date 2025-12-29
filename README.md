# Pixel-NeRF: Multi-View Stereo Enhanced (in Colab)

본 프로젝트는 **Pixel-NeRF** (Yu et al., 2021)의 아키텍처를 기반으로, **Multi-View Stereo (MVS)** 개념(Variance 기반 정합성 판단)과 **FPN(Feature Pyramid Network)** 을 도입하여 학습 효율을 극대화한 구현체입니다.

Colab 환경에서도 **약 30시간** 의 학습만으로 ShapeNet 차량 데이터셋에 대해 준수한 3D 형상을 복원할 수 있도록 설계되었습니다.

<br>

## 🎥 Demo Results
/rendering_demo.gif

<br>

---

## ⚡ Context & Optimization Strategy

프로젝트의 목표는 GPU 환경에 손쉽게 접근할 수 있는 Colab 환경 내에서 pixel-NeRF를 활용하고, 그에 따른 최상의 기하학적 품질을 얻는 것입니다. 제한된 GPU 사용량을 고려하여 기존의 Few-shot(1~2뷰) 학습 방식 대신, **멀티 뷰(6-views)** 를 사용하여 학습 시간을 줄이고 빠르게 학습될 수 있게 하였습니다. 

| 구분 | Original Pixel-NeRF | **Ours (Colab Optimized)** |
| :--- | :--- | :--- |
| **Environment** | Heavy Workstation | **Google Colab (Ready-to-Run)** |
| **Input Views** | 1 ~ 2 Views (Sparse) | **6 Views (Geometric Constraints)** |
| **Training Time** | 6 Days+ (V100) | **30 Hours (A100)** |
| **Steps** | 400k+ Steps | **100k Steps (Early Convergence)** |
| **Dataset** | ShapeNet (Cars) | ShapeNet (Cars) |

<br>

---

## 🚀 Getting Started (Colab Friendly)

이 코드는 **Google Colab** 환경에 최적화되어 있습니다. 별도의 복잡한 환경 설정 없이 바로 실행 가능합니다.

### 1. Environment Setup
Colab 노트북에서 별도의 가상환경 설정 없이, 필요한 라이브러리만 설치하면 즉시 작동합니다.
```bash
# Colab 셀에서 실행
!pip install imageio tqdm matplotlib configargparse

```

### 2. Data Loading (One-Line Command)

복잡한 데이터 다운로드 과정 없이, 아래 스크립트를 통해 학습에 필요한 ShapeNet 데이터를 즉시 로드할 수 있습니다.

```bash
# Colab 셀에서 실행
!bash fetch2local.sh -from nerf-data-ssu/shapeNetV2_cars

```

* *데이터는 자동으로 현재 환경에 맞게 구성됩니다.*

### 3. Training

```bash
# 기본 설정(6 views, 100k steps)으로 학습 시작
python train.py --config config/default_config.yaml

```

* *Tip: 최초 실행 시 기하학적 필터링을 위한 캐시(`.pt`) 생성으로 인해 시작에 약 5~10분이 소요될 수 있습니다.*

### 4. Inference (Video Generation)

```bash
python inference.py --input_folder ./data/cars_test/object_id \
                    --checkpoint checkpoints/best_model.pth \
                    --mode video \
                    --num_frames 90

```

---

## 🏗️ Technical Enhancements: How it works?

단순히 뷰 개수만 늘린 것이 아니라, 늘어난 정보를 효과적으로 처리하기 위해 모델 아키텍처를 **Stereo Matching** 에 적합한 구조로 고도화했습니다.

### 1. Multi-Scale Feature Extraction (FPN)

* **기존:** ResNet의 단일 레이어 특징맵만 사용 (정보 손실 발생).
* **개선:** **FPN (Feature Pyramid Network)** 을 결합하여, **4가지 해상도( ~ )의 특징맵** 과 **원본 RGB** 를 모두 추출하여 리스트 형태로 NeRF 헤드에 전달합니다. 이를 통해 디테일과 전체 형상을 동시에 학습합니다.

### 2. Early Fusion with Variance Injection

* **기존:** 여러 뷰의 특징을 단순히 평균(Average)내어 MLP에 전달. 뷰 간의 차이(불일치) 정보가 사라짐.
* **개선:** MVS(Multi-View Stereo)의 핵심인 **분산(Variance)**을 함께 계산하여 MLP 입력단에 주입(**Early Fusion**)했습니다.
* **Variance의 역할:** "이 지점에서 6개의 카메라가 같은 색상을 보고 있는가?"를 판단합니다. 분산이 낮다면 물체 표면일 확률이 높다는 강력한 신호(Stereo Cue)가 됩니다.



### 3. Smart Data Filtering

* **Geometric Pruning:** 학습에 방해가 되는 '바닥(Floor)' 뷰나 '정수리(Top-down)' 뷰를 카메라 파라미터 기반으로 수학적으로 계산하여 사전에 제거했습니다.

---

## Acknowledgement

This project builds upon [Pixel-NeRF](https://github.com/sxyu/pixel-nerf). We optimized it for constrained environments by introducing **FPN encoders**, **Variance-based feature aggregation**, and **Geometric data pruning**.


---
