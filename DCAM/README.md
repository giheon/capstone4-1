# DCAM (Deep Clustering with Associative Memories) — PyTorch implementation

**Deep Clustering with Associative Memories (DCAM)** 논문의 핵심 아이디어를
**PyTorch + Lightning Fabric + Hydra** 스타일로 구현한 코드베이스

## 구현 목표

- encoder `e`, decoder `d`, prototype `rho`, inverse temperature `beta`, attractor step `T`, step size `tau`
- 이미지 / 벡터(텍스트 TF-IDF, tabular) 데이터 모두 대응
- 확장 가능한 AE backbone:
  - `cae`: convolutional autoencoder
  - `rae`: residual autoencoder
  - `eae`: MLP autoencoder
- 논문처럼 **AE pretraining → rho initialization → DCAM training → inference/eval** 파이프라인 제공

## 프로젝트 구조

```text
.
├── configs/
│   ├── config.yaml
│   ├── data/
│   ├── model/
│   └── trainer/
├── data/
│   ├── raw/
│   └── processed/
├── src/dcam/
│   ├── data/
│   ├── engine/
│   ├── metrics/
│   ├── models/
│   ├── utils/
│   ├── infer.py
│   └── train.py
└── pyproject.toml
```

## 추천 실행 환경

상위 `capstone4-1` 폴더에서 공통 가상환경을 만든 뒤 `requirements.txt`로 설치하는 방식을 권장합니다.
DCAM은 `Python >=3.10,<3.12`를 요구하므로 `python3.10` 환경을 사용하세요.

```bash
cd ..
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

위 설치에는 `-e ./DCAM`, `-e ./VaDE`가 포함되어 있어서 `dcam-train`, `dcam-infer` 명령이 바로 등록되고,
DCAM에서 VaDE 벤치마크 데이터셋도 직접 불러올 수 있습니다.

## 데이터 넣는 위치

### 1 이미지 데이터

```text
data/raw/my_images/
├── sample_0001.png
├── sample_0002.jpg
└── ...
```

또는 하위 폴더 구조도 허용합니다.

```text
data/raw/my_images/
├── any_folder_a/
│   ├── img1.png
│   └── img2.png
└── any_folder_b/
    └── img3.png
```

> 클러스터링은 비지도라서 폴더명이 라벨일 필요는 없습니다.
> 다만 evaluation용 정답 라벨이 있으면 별도 CSV/NPY로 넣을 수 있게 해두었습니다.

### 2 벡터 / 텍스트 / tabular 데이터

지원 파일 형식:
- `.npy`
- `.npz`
- `.csv`
- `.pt`

예시:

```text
data/raw/reuters10k/features.npy
```

shape 예시:
- `[N, F]`
  - `N`: 샘플 수
  - `F`: feature dimension

### 3 VaDE 벤치마크 데이터셋 직접 사용

DCAM은 VaDE에서 쓰는 아래 데이터셋을 파일 변환 없이 바로 읽을 수 있습니다.

- `mnist`
- `reuters10k`
- `har`
- `reuters_all`

Hydra data config는 `data=vade_benchmark`를 사용합니다.
이 경로는 내부적으로 `vade.data.datasets.load_data()`를 호출하므로, VaDE 쪽 전처리를 그대로 재사용합니다.
기본값은 `normalize=false`라서 VaDE 로더가 만든 입력 스케일을 유지합니다.

## 가장 기본 실행 예시

### 이미지 + CAE

```bash
dcam-train \
  data=image_folder \
  model=dcam_cae \
  data.root=data/raw/my_images \
  data.image_size=28 \
  model.latent_dim=10 \
  model.num_clusters=10
```

### 벡터 + EAE

```bash
dcam-train \
  data=array_features \
  model=dcam_eae \
  data.root=data/raw/reuters10k/features.npy \
  model.latent_dim=46 \
  model.num_clusters=46
```

### VaDE 데이터셋 + EAE

MNIST:

```bash
dcam-train \
  data=vade_benchmark \
  model=dcam_eae \
  data.dataset_name=mnist \
  model.latent_dim=10 \
  model.num_clusters=10
```

Reuters10k:

```bash
dcam-train \
  data=vade_benchmark \
  model=dcam_eae \
  data.dataset_name=reuters10k \
  model.latent_dim=15 \
  model.num_clusters=4
```

HAR:

```bash
dcam-train \
  data=vade_benchmark \
  model=dcam_eae \
  data.dataset_name=har \
  model.latent_dim=120 \
  model.num_clusters=6
```

Reuters-all:

```bash
dcam-train \
  data=vade_benchmark \
  model=dcam_eae \
  data.dataset_name=reuters_all \
  model.latent_dim=15 \
  model.num_clusters=4
```

VaDE 원본 데이터가 기본 위치가 아니면 `data.data_root=/path/to/VaDE/dataset`를 추가하면 됩니다.

## inference 예시

```bash
dcam-infer \
  checkpoint_path=outputs/latest/checkpoints/best.pt \
  data=image_folder \
  model=dcam_cae \
  data.root=data/raw/my_images \
  model.latent_dim=10 \
  model.num_clusters=10
```

결과물:
- `cluster_assignments.csv`
- `latent_before.npy`
- `latent_after.npy`
- `rho.npy`
- `decoded_prototypes.npy`
- 이미지 데이터인 경우 `decoded_prototypes.png`

## 논문 대응 관계

논문 핵심 식:

- latent encoding: `v = e(x)`
- attractor dynamics: `v' = A^T_rho(v)`
- reconstruction from moved latent: `x_hat = d(v')`
- DCAM loss: `||x - d(A^T_rho(e(x)))||^2`

코드에서는 다음과 같이 대응됩니다:

- `DCAMModel.encode()` → `e(x)`
- `AssociativeMemory.forward()` → `A^T_rho(v)`
- `DCAMModel.decode()` → `d(v')`
- `dcam_reconstruction_loss()` → 논문의 단일 loss

## 주의

1. 논문 본문은 **수식과 high-level architecture**는 분명하지만, 일부 세부 구현(특히 residual AE block 세부 배치)은
   축약되어 있습니다. 그래서 RAE는 논문 설명에 맞는 **합리적 residual AE**로 구현했습니다.
2. CAE는 논문이 참조한 DCEC 계열의 `conv5_32 -> conv5_64 -> conv3_128 -> FC_d` 구조를 따릅니다.
3. 논문은 TensorFlow로 실험했지만, 여기서는 현재 연구/개발에서 훨씬 널리 쓰이는 pyTorch로 옮겼습니다.

## 추천 override 패턴

```bash
# beta 조정
model.beta=0.5

# tau 조정
model.tau=1.0

# attractor step curriculum 시작값
trainer.curriculum.t_start=1

# 최대 T
trainer.curriculum.t_max=20

# rho learning rate
trainer.optim.lr_rho=1e-3
```
