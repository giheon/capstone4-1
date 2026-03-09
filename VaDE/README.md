# VaDE (Variational Deep Embedding) — PyTorch implementation

`data / models / engine / metrics / utils / train / infer`

## 현재 코드 구조

- `src/vade/models/vade.py`
  - VaDE 모델 정의와 핵심 수식 구현
- `src/vade/engine/train.py`
  - AE 초기화, GMM 초기화, 본학습 루프, epoch 평가, history/W&B 기록
- `src/vade/engine/diagnostics.py`
  - assignment 누적 저장, GMM 진단, heatmap/t-SNE 시각화, diagnostics 아티팩트 저장
- `src/vade/engine/pretrain.py`
  - 저자 AE 가중치 로드 또는 scratch AE pretraining
- `src/vade/engine/evaluate.py`
  - dataset 단위 latent encoding / gamma prediction / clustering evaluation
- `src/vade/train.py`
  - CLI entrypoint와 실험 설정 연결

즉 현재 기준으로는:

- 학습 핵심 코드: `engine/train.py`
- 학습 중/후 확인용 저장과 시각화: `engine/diagnostics.py`
- 모델 정의: `models/vade.py`

## 실행 예시

```bash
cd VaDE
pip install -e .

vade-train mnist
vade-infer mnist
vade-infer reuters_all
```

## W&B 연동

`--wandb`를 주면 학습 step/epoch metric을 Weights & Biases로 같이 보냅니다.
기본 설정으로 batch 계열 metric은 step 축을 유지하고, epoch 계열 metric(`train/loss`, `eval/acc`, `lr/*`)은 W&B에서 `train/epoch` 축으로 보이도록 설정됩니다.

```bash
wandb login
vade-train mnist --wandb --wandb-project vade --wandb-name mnist-run-01
```

오프라인 저장은:

```bash
vade-train mnist --wandb --wandb-mode offline
```

## Diagnostics

GMM 초기화 직후, 학습 중 매 `N` epoch, 학습 종료 후에 아래 진단 아티팩트를 저장하고 W&B에도 올릴 수 있습니다.

- `theta(pi)` bar plot
- `argmax(gamma)` cluster count bar plot
- `lambda` 요약 통계
- `mu_c` pairwise distance 통계
- cluster × label weight heatmap
- label → predicted cluster count heatmap
- cluster purity / label purity scalar
- label-colored `t-SNE` with centroid overlay
- predicted-cluster-colored `t-SNE` with centroid overlay
- final 시점의 `init vs final` 비교 그림

예시:

```bash
vade-train mnist \
  --pretrain \
  --epochs 1000 \
  --device mps \
  --wandb \
  --wandb-project vade \
  --diagnostics \
  --diagnostics-interval 50 \
  --tsne-interval 100
```

이전 diagnostics 결과를 지우고 새로 시작하려면:

```bash
vade-train mnist \
  --diagnostics \
  --clean-diagnostics
```

기본 저장 위치:

```text
VaDE/checkpoints/<checkpoint_stem>_diagnostics/
```

기본 동작은 기존 diagnostics 폴더를 유지한 채 같은 파일명만 덮어쓰는 방식입니다.
`--clean-diagnostics`를 주면 실행 시작 전에 diagnostics 폴더 전체를 삭제하고 새로 생성합니다.

기본 주기:

- diagnostics scalar / bar plot 저장: `50` epoch마다
- diagnostics `t-SNE` 저장 및 W&B 업로드: `100` epoch마다
- cluster × label weight heatmap 저장 및 W&B 업로드: `100` epoch마다
- label → predicted cluster count heatmap 저장 및 W&B 업로드: `100` epoch마다
- cluster purity / label purity 기록: 매 epoch
- `init`과 `final` 시점의 `t-SNE`는 항상 저장됩니다.

## 학습 중 보이는 값과 저장되는 값

VaDE는 학습 정보를 네 군데로 나눠서 다룹니다.

1. 콘솔에만 출력되는 값
2. 체크포인트 내부 `history`에 저장되는 값
3. diagnostics 폴더에 별도 파일로 저장되는 값
4. W&B에 기록되는 값

### 1. 콘솔에 보이는 값

학습 중 콘솔에서 볼 수 있는 값은 다음과 같습니다.

- 시작 정보
  - `training on: <dataset>`
  - `device: <device>`
- 저자 AE 가중치 사용 시
  - `author pretrain weights loaded: ...`
- AE를 scratch로 pretrain할 때만
  - `[pretrain] epoch=... loss=...`
- learning rate decay 시점
  - `lr_nn: ... | lr_gmm: ...`
- 매 epoch
  - `epoch 0001/1000 | loss=... | acc_p_c_z=... | lr_nn=... | lr_gmm=...`
- diagnostics 시점
  - `[diagnostics:init] ...`
  - `[diagnostics:epoch_0050] ...`
  - `[diagnostics:final] ...`
- 종료 시
  - `final acc_p_c_z: ... | pred shape=...`
  - `checkpoint saved: ...`

주의:

- scratch pretraining의 `loss`는 현재 콘솔에는 보이지만 `history`에는 저장되지 않습니다.
- diagnostics 한 줄 요약도 콘솔에는 보이지만 `history`에는 저장되지 않습니다.

### 2. 체크포인트 내부 `history`

체크포인트는 기본적으로 아래 경로에 저장됩니다.

```text
VaDE/checkpoints/vade_<dataset>.pt
```

이 파일 안에는 아래 payload가 들어갑니다.

- `model_state`
- `model_kwargs`
- `train_config`
- `history`
- `seed`

학습 완료 후 생성되는 기본 출력물은 다음과 같습니다.

- 항상 생성
  - `VaDE/checkpoints/vade_<dataset>.pt`
- `--diagnostics`를 켠 경우 추가 생성
  - `VaDE/checkpoints/vade_<dataset>_diagnostics/`

즉 diagnostics는 `.pt` 파일이 아니라 별도 폴더입니다.

`history`는 학습 scalar 추적용입니다. 원본 배열이나 그림은 들어가지 않습니다.

50 global step마다 저장되는 `history` 항목:

- `step`
- `epoch`
- `step_in_epoch`
- `batch_gamma_entropy`
- `batch_theta_entropy`
- `batch_cluster_usage_entropy`
- `batch_cluster_top1_ratio`
- `batch_recon`
- `batch_kld_like`

매 epoch마다 저장되는 `history` 항목:

- `loss`
- `recon`
- `kld_like`
- `z_entropy`
- `cat_prior`
- `cat_entropy`
- `acc`
- `cluster_purity`
- `label_purity`
- `lr_nn`
- `lr_gmm`

`history`에 저장되지 않는 대표 항목:

- `theta`, `mu_c`, `lambda_c` 원본 배열
- `hard_assignments`
- `t-SNE` 좌표
- diagnostics 그림 파일
- `gamma` 전체 행렬

### 3. Diagnostics 폴더에 저장되는 값

diagnostics를 켜면 아래 구조로 아티팩트가 저장됩니다.

```text
VaDE/checkpoints/<checkpoint_stem>_diagnostics/
├── cluster_label_weight.npy
├── label_cluster_counts.npy
├── assignment_metrics.json
├── init/
├── epoch_0050/
├── epoch_0100/
├── ...
└── final/
```

시점 설명:

- `init/`: GMM initialization 직후 1회
- `epoch_0050/`, `epoch_0100/`, ... : `--diagnostics-interval` 기준 GMM diagnostics 저장
- `final/`: 학습 종료 직후

diagnostics 루트에 누적 저장되는 assignment 파일:

- `cluster_label_weight.npy`
  - run당 파일 1개만 생성되고, epoch가 진행될 때마다 앞 축(`T`)에 누적됩니다.
  - shape: `[T, cluster, label]`
  - 각 epoch의 `weight[cluster, label]`
- `label_cluster_counts.npy`
  - run당 파일 1개만 생성되고, epoch가 진행될 때마다 앞 축(`T`)에 누적됩니다.
  - shape: `[T, label, cluster]`
  - 각 epoch의 `counts[label, cluster]`
- `assignment_metrics.json`
  - run당 파일 1개만 생성되고, epoch별 scalar가 리스트로 누적됩니다.
  - `epochs`
  - `cluster_purity`
  - `label_purity`
  - 위 배열들은 같은 index 기준으로 대응됩니다.
  - 예: `epochs[4] == 5`이면 `cluster_label_weight.npy[4]`, `label_cluster_counts.npy[4]`도 epoch 5 결과입니다.

각 stage 폴더에 저장되는 파일:

- `theta.npy`
  - `theta = softmax(pi_logits)`
- `mu_c.npy`
  - centroid
- `lambda_c.npy`
  - diagonal variance
- `labels.npy`
  - 정답 label
- `hard_assignments.npy`
  - `argmax(gamma)`
- `cluster_counts.npy`
  - cluster별 hard assignment count
- `mu_c_pairwise_distance.npy`
  - centroid 간 pairwise distance matrix
- `tsne_points_2d.npy`
  - 전체 데이터의 t-SNE 2D 좌표
- `tsne_centroids_2d.npy`
  - centroid의 t-SNE 2D 좌표
  - 두 파일 모두 `init`, `final`, 그리고 `--tsne-interval` 배수 epoch에서만 생성
- `stats.json`
  - 아래 요약 통계 포함
    - `theta`: `entropy`, `min`, `max`, `mean`
    - `cluster_counts`: `entropy`, `min`, `max`, `mean`
    - `assignment`: `cluster_purity`, `label_purity`
    - `lambda`: `global_min`, `global_max`, `global_mean`, `per_cluster_mean`, `per_cluster_min`, `per_cluster_max`
    - `mu_c_pairwise_distance`: `min`, `max`, `mean`, `matrix_shape`

각 stage 폴더의 그림 파일:

- `theta_bar.png`
  - 그림 제목: `<stage> theta`
- `cluster_count_bar.png`
  - 그림 제목: `<stage> cluster counts`
- `label_cluster_distribution.png`
  - 그림 제목: `<stage> label -> predicted cluster counts`
  - `init`, `final`, 그리고 `--tsne-interval` 배수 epoch에서만 생성
- `cluster_label_weight_heatmap.png`
  - 그림 제목: `<stage> cluster x label weight`
  - `init`, `final`, 그리고 `--tsne-interval` 배수 epoch에서만 생성
- `tsne_label_centroid.png`
  - 그림 제목: `<stage> t-SNE (label-colored)`
- `tsne_pred_cluster_centroid.png`
  - 그림 제목: `<stage> t-SNE (predicted cluster-colored)`
  - 두 t-SNE 그림 모두 `init`, `final`, 그리고 `--tsne-interval` 배수 epoch에서만 생성

추가 final 비교 그림:

- `final/tsne_init_vs_final.png`
  - 왼쪽 패널 제목: `GMM Init`
  - 오른쪽 패널 제목: `Final`

기본적으로 diagnostics에서는 `gamma.npy` 전체는 저장하지 않습니다.
이유는 크기가 커서 실험 반복 시 부담이 커지기 때문입니다.

### 4. W&B에 기록되는 값

step 기준 W&B scalar:

- `train/batch_gamma_entropy`
- `train/batch_theta_entropy`
- `train/batch_cluster_usage_entropy`
- `train/batch_cluster_top1_ratio`
- `train/batch_recon`
- `train/batch_kld_like`
- `train/epoch`
- `train/step_in_epoch`

epoch 기준 W&B scalar:

- `train/loss`
- `train/recon`
- `train/kld_like`
- `train/z_entropy`
- `train/cat_prior`
- `train/cat_entropy`
- `eval/acc`
- `eval/cluster_purity`
- `eval/label_purity`
- `lr/nn`
- `lr/gmm`
- `train/epoch`

diagnostics 기준 W&B scalar:

- `diagnostics/<stage>/theta_entropy`
- `diagnostics/<stage>/theta_min`
- `diagnostics/<stage>/theta_max`
- `diagnostics/<stage>/cluster_count_entropy`
- `diagnostics/<stage>/cluster_count_min`
- `diagnostics/<stage>/cluster_count_max`
- `diagnostics/<stage>/cluster_purity`
- `diagnostics/<stage>/label_purity`
- `diagnostics/<stage>/lambda_global_min`
- `diagnostics/<stage>/lambda_global_max`
- `diagnostics/<stage>/lambda_global_mean`
- `diagnostics/<stage>/mu_distance_min`
- `diagnostics/<stage>/mu_distance_max`
- `diagnostics/<stage>/mu_distance_mean`

diagnostics 기준 W&B image:

- `diagnostics/<stage>/theta_bar`
- `diagnostics/<stage>/cluster_count_bar`
- `diagnostics/<stage>/label_cluster_distribution`
- `diagnostics/<stage>/cluster_label_weight_heatmap`
- `diagnostics/<stage>/tsne_label`
- `diagnostics/<stage>/tsne_pred_cluster`
- `diagnostics/final/tsne_init_vs_final`

위 `t-SNE` 이미지 키들은 `init`, `final`, 그리고 `--tsne-interval` 배수 epoch에서만 올라갑니다.
`diagnostics/<stage>/label_cluster_distribution`와 `diagnostics/<stage>/cluster_label_weight_heatmap`도 `init`, `final`, 그리고 `--tsne-interval` 배수 epoch에서만 올라갑니다.

W&B summary:

- `final_acc`
- `checkpoint_path`
- `diagnostics_dir`
- `num_predictions`

### 요약

- `history`는 학습 scalar 기록용입니다.
- diagnostics 폴더는 GMM 상태, t-SNE 좌표, 그림 같은 snapshot 저장용입니다.
- W&B는 `history` 성격의 scalar와 diagnostics 그림/통계를 같이 보여줍니다.
- 콘솔 출력은 일부만 저장되고, 일부는 화면에만 보입니다.
