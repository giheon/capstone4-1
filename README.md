# capstone4-1: DCAM + VaDE

이 폴더는 두 개의 PyTorch 기반 클러스터링 모델을 포함합니다.

- `VaDE/`: Variational Deep Embedding 구현
- `DCAM/`: Deep Clustering with Associative Memories 구현

## 공통 Python 환경

DCAM이 `Python >=3.10,<3.12`를 요구하므로 이 저장소는 `python3.10` 기반 가상환경을 권장합니다.

```bash
cd capstone4-1
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd capstone4-1
python3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt`에는 VaDE와 DCAM이 함께 사용하는 라이브러리 버전이 고정되어 있으며, `-e ./DCAM`이 포함되어 있어 설치 후 `dcam-train`, `dcam-infer` 명령도 바로 사용할 수 있습니다.

## VaDE

주요 파일:

- `VaDE/VaDE.py`: 학습 진입점
- `VaDE/vade_core.py`: 모델, 데이터 로딩, 손실 함수, 학습 유틸리티
- `VaDE/VaDE_test_mnist.py`: MNIST 평가 및 샘플 이미지 생성
- `VaDE/VaDE_test_reuters_all.py`: Reuters-all 평가

학습 예시:

```bash
python VaDE/VaDE.py mnist
python VaDE/VaDE.py reuters10k
python VaDE/VaDE.py har
python VaDE/VaDE.py reuters_all
```

추가 옵션 예시:

```bash
python VaDE/VaDE.py mnist --epochs 200 --lr-nn 0.001 --lr-gmm 0.0005
python VaDE/VaDE.py mnist --pretrain
python VaDE/VaDE.py mnist --no-pretrain
python VaDE/VaDE.py mnist --no-pretrain --pretrain-epochs 120
```

평가:

```bash
python VaDE/VaDE_test_mnist.py --show
python VaDE/VaDE_test_reuters_all.py
```

Reuters 원본 파일이 없으면 아래 스크립트로 받을 수 있습니다.

```bash
cd VaDE/dataset/reuters
./get_data.sh
```

## DCAM

DCAM은 `DCAM/` 하위 패키지로 관리됩니다. 공통 가상환경 설치가 끝나면 아래처럼 실행할 수 있습니다.

```bash
dcam-train \
  data=image_folder \
  model=dcam_cae \
  data.root=DCAM/data/raw/my_images \
  data.image_size=28 \
  model.latent_dim=10 \
  model.num_clusters=10
```

```bash
dcam-infer \
  checkpoint_path=DCAM/outputs/latest/checkpoints/best.pt \
  data=image_folder \
  model=dcam_cae \
  data.root=DCAM/data/raw/my_images \
  model.latent_dim=10 \
  model.num_clusters=10
```

세부 설정과 데이터 구조는 `DCAM/README.md`를 참고하면 됩니다.
