import copy
import gzip
import math
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


EPS = 1e-10

# ============================================================================
# Config
# ============================================================================
@dataclass
class TrainConfig:
    dataset: str
    input_dim: int
    epochs: int
    n_centroid: int
    lr_nn: float
    lr_gmm: float
    decay_n: int
    decay_nn: float
    decay_gmm: float
    alpha: float
    reconstruction: str
    batch_size: int = 100
    latent_dim: int = 10
    hidden_dims: Tuple[int, int, int] = (500, 500, 2000)
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3


def get_default_config(dataset: str) -> TrainConfig:
    if dataset == "mnist":
        return TrainConfig("mnist", 784, 3000, 10, 0.002, 0.002, 10, 0.9, 0.9, 1.0, "sigmoid")
    if dataset == "reuters10k":
        return TrainConfig("reuters10k", 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1.0, "linear")
    if dataset == "har":
        return TrainConfig("har", 561, 120, 6, 0.002, 0.00002, 10, 0.9, 0.9, 5.0, "linear")
    if dataset == "reuters_all":
        return TrainConfig("reuters_all", 2000, 15, 4, 0.002, 0.002, 5, 0.5, 0.5, 1.0, "linear")
    raise ValueError(f"Unsupported dataset: {dataset}")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================================
# Evaluation
# ============================================================================
def cluster_acc(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError("y_pred and y_true size mismatch")
    y_pred = y_pred.astype(np.int64)
    y_true = y_true.astype(np.int64)

    dim = int(max(y_pred.max(), y_true.max()) + 1)
    weight = np.zeros((dim, dim), dtype=np.int64) # weight[a, b] : 클러스터 a로 예측된 샘플 중 정답이 b인 개수

    for i in range(y_pred.size):
        weight[y_pred[i], y_true[i]] += 1 

    row_ind, col_ind = linear_sum_assignment(weight.max() - weight) # 클러스터와 정답라벨 매칭

    acc = float(weight[row_ind, col_ind].sum()) / float(y_pred.size) # 정확도 계산
    assignment = np.stack([row_ind, col_ind], axis=1) # 매칭쌍 만들기
    return acc, assignment, weight

# ============================================================================
# Data
# ============================================================================
def load_mnist(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = data_root / "mnist" / "mnist.pkl.gz"
    with gzip.open(path, "rb") as f:
        (x_train, y_train), (x_test, y_test) = pickle.load(f, encoding="bytes")
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)
    x = np.concatenate([x_train, x_test], axis=0).astype(np.float32)
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int64)
    return x, y


def load_reuters10k(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = scio.loadmat(data_root / "reuters10k" / "reuters10k.mat")
    x = data["X"].astype(np.float32)
    y = data["Y"].squeeze().astype(np.int64)
    return x, y


def load_har(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = scio.loadmat(data_root / "har" / "HAR.mat")
    x = data["X"].astype(np.float32)[:10200]
    y = (data["Y"].squeeze().astype(np.int64) - 1)[:10200]
    return x, y


def load_reuters_all(data_root: Path, max_features: int = 2000, max_samples: int = 685000) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    reuters_root = data_root / "reuters"
    did_to_cat: Dict[int, List[str]] = {}
    cat_set = {"CCAT", "GCAT", "MCAT", "ECAT"}

    with (reuters_root / "rcv1-v2.topics.qrels").open("r", encoding="utf-8") as f:
        for line in f:
            cat, did_raw, _ = line.strip().split(" ")
            did = int(did_raw)
            if cat in cat_set:
                did_to_cat.setdefault(did, []).append(cat)

    for did in list(did_to_cat.keys()):
        if len(did_to_cat[did]) > 1:
            del did_to_cat[did]

    dat_list = [
        "lyrl2004_tokens_test_pt0.dat",
        "lyrl2004_tokens_test_pt1.dat",
        "lyrl2004_tokens_test_pt2.dat",
        "lyrl2004_tokens_test_pt3.dat",
        "lyrl2004_tokens_train.dat",
    ]

    data: List[str] = []
    target: List[int] = []
    cat_to_id = {"CCAT": 0, "GCAT": 1, "MCAT": 2, "ECAT": 3}

    did: Optional[int] = None
    doc = ""
    for dat_name in dat_list:
        with (reuters_root / dat_name).open("r", encoding="latin-1") as f:
            for line in f:
                if line.startswith(".I"):
                    if did is not None and doc and did in did_to_cat:
                        data.append(doc)
                        target.append(cat_to_id[did_to_cat[did][0]])
                    did = int(line.strip().split(" ")[1])
                    doc = ""
                elif line.startswith(".W"):
                    continue
                else:
                    doc += line

    if did is not None and doc and did in did_to_cat:
        data.append(doc)
        target.append(cat_to_id[did_to_cat[did][0]])

    x = CountVectorizer(dtype=np.float64, max_features=max_features).fit_transform(data)
    y = np.asarray(target, dtype=np.int64)
    x = TfidfTransformer(norm="l2", sublinear_tf=True).fit_transform(x)
    x = np.asarray(x.todense()) * np.sqrt(x.shape[1])
    x = preprocessing.normalize(x, norm="l2") * 200.0
    x = x.astype(np.float32)
    return x[:max_samples], y[:max_samples]


def load_data(dataset: str, data_root: str = "dataset") -> Tuple[np.ndarray, np.ndarray]:
    root = Path(data_root)
    if dataset == "mnist":
        return load_mnist(root)
    if dataset == "reuters10k":
        return load_reuters10k(root)
    if dataset == "har":
        return load_har(root)
    if dataset == "reuters_all":
        return load_reuters_all(root)
    raise ValueError(f"Unsupported dataset: {dataset}")

#---------------------------------------------------------------------------------------------------------#

def get_author_pretrain_weight_path(dataset: str) -> Path:
    mapped_dataset = "reuters10k" if dataset == "reuters_all" else dataset
    return Path("pretrain_weights") / f"ae_{mapped_dataset}.pt"


def load_author_pretrained_autoencoder(config: TrainConfig, device: torch.device) -> StackedAutoEncoder:
    weight_path = get_author_pretrain_weight_path(config.dataset)
    if not weight_path.exists():
        raise FileNotFoundError(f"author pretrain weights not found: {weight_path}")

    ae = StackedAutoEncoder(config.input_dim, config.latent_dim, config.hidden_dims, config.reconstruction).to(device)

    payload = torch.load(weight_path, map_location=device)
    if hasattr(payload, "state_dict"):
        state = payload.state_dict()
    elif isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state = payload["state_dict"]
    elif isinstance(payload, dict) and "model_state" in payload and isinstance(payload["model_state"], dict):
        state = payload["model_state"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise TypeError(f"Unsupported pretrain payload type: {type(payload)} at {weight_path}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    expected_keys = set(ae.state_dict().keys())
    state_keys = set(state.keys())
    if expected_keys != state_keys:
        if expected_keys.issubset(state_keys):
            state = {k: state[k] for k in ae.state_dict().keys()}
        else:
            missing = sorted(expected_keys - state_keys)
            extra = sorted(state_keys - expected_keys)
            raise KeyError(
                f"pretrain state dict mismatch for {weight_path}. "
                f"missing={missing[:5]} extra={extra[:5]}"
            )

    ae.load_state_dict(state, strict=True)

    return ae

# ============================================================================
# Model
# ============================================================================
class StackedAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: Sequence[int], reconstruction: str) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims # (500, 500, 2000)
        self.reconstruction = reconstruction # sigmoid or linear

        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.enc4 = nn.Linear(h3, latent_dim)

        self.dec1 = nn.Linear(latent_dim, h3)
        self.dec2 = nn.Linear(h3, h2)
        self.dec3 = nn.Linear(h2, h1)
        self.dec4 = nn.Linear(h1, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        return self.enc4(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        x_hat = self.dec4(h)
        if self.reconstruction == "sigmoid":
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class VaDE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_centroid: int,
        hidden_dims: Sequence[int],
        reconstruction: str,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_centroid = n_centroid
        self.hidden_dims = tuple(hidden_dims)
        self.reconstruction = reconstruction

        # 모델이 학습해야 하는 encoder 선형변환 레이어, 파라미터(=가중치)를 선언
        self.enc1 = nn.Linear(input_dim, h1)
        self.enc2 = nn.Linear(h1, h2)
        self.enc3 = nn.Linear(h2, h3)
        self.z_mean = nn.Linear(h3, latent_dim)
        self.z_log_var = nn.Linear(h3, latent_dim)

        # 모델이 학습해야 하는 decoder 선형변환 레이어, 파라미터(=가중치)를 선언
        self.dec1 = nn.Linear(latent_dim, h3)
        self.dec2 = nn.Linear(h3, h2)
        self.dec3 = nn.Linear(h2, h1)
        self.x_bar = nn.Linear(h1, input_dim)

        # 모델이 학습해야 하는 GMM 파라미터(=가중치)를 선언
        self.pi_logits = nn.Parameter(torch.zeros(n_centroid))
        self.mu_c = nn.Parameter(torch.zeros(n_centroid, latent_dim))
        self.log_var_c = nn.Parameter(torch.zeros(n_centroid, latent_dim))

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        return self.z_mean(h), self.z_log_var(h)

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        h = F.relu(self.dec3(h))
        x_hat = self.x_bar(h)
        if self.reconstruction == "sigmoid":
            x_hat = torch.sigmoid(x_hat)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encode(x) 
        z = self.reparameterize(z_mean, z_log_var)
        x_hat = self.decode(z)
        return x_hat, z_mean, z_log_var, z
    # input x -> z -> x_hat

    def mixture_parameters(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = torch.softmax(self.pi_logits, dim=0) # theta shape: (K,)
        lambda_c = torch.exp(self.log_var_c) # lambda_c shape: (K, J)
        return theta, self.mu_c, lambda_c # mu_c shape: (K, J)

    # latent variable z로 gamma 계산
    def compute_gamma(self, z: torch.Tensor) -> torch.Tensor:
        theta, mu_c, lambda_c = self.mixture_parameters()
        z_expand = z.unsqueeze(1) # z: (B, J) -> z_expand: (B, 1, J)
        mu_expand = mu_c.unsqueeze(0) # mu_c: (K, J) -> mu_expand: (1, K, J)
        lambda_expand = lambda_c.unsqueeze(0) # lambda_c: (K, J) -> lambda_expand: (1, K, J)

        # log p(c|z) ∝ log pi_c + sum_j log N(z_j | mu_cj, lambda_cj)
        log_theta = torch.log(theta + EPS).view(1, self.n_centroid) # (1, K)
        gaussian_log_prob = -0.5 * torch.log(2.0 * math.pi * lambda_expand + EPS)
        gaussian_log_prob = gaussian_log_prob - ((z_expand - mu_expand) ** 2) / (2.0 * lambda_expand + EPS) # (B, K, J)
        gaussian_log_prob = gaussian_log_prob.sum(dim=2) # (B, K)
        log_prob = log_theta + gaussian_log_prob 
        return torch.softmax(log_prob, dim=1) # gamma shape: (B, K)

    def vade_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        z_mean: torch.Tensor,
        z_log_var: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        theta, mu_c, lambda_c = self.mixture_parameters()
        gamma = self.compute_gamma(z)

        if self.reconstruction == "sigmoid":
            recon = F.binary_cross_entropy(x_hat, x, reduction="none").sum(dim=1)
        else:
            recon = F.mse_loss(x_hat, x, reduction="none").sum(dim=1)
        recon = alpha * recon

        batch = x.shape[0]
        z_mean_t = z_mean.unsqueeze(2).expand(batch, self.latent_dim, self.n_centroid)
        z_log_var_t = z_log_var.unsqueeze(2).expand(batch, self.latent_dim, self.n_centroid)
        mu_t = mu_c.t().unsqueeze(0).expand(batch, self.latent_dim, self.n_centroid)
        lambda_t = lambda_c.t().unsqueeze(0).expand(batch, self.latent_dim, self.n_centroid)
        gamma_t = gamma.unsqueeze(1).expand(batch, self.latent_dim, self.n_centroid)

        term_const = self.latent_dim * math.log(math.pi * 2.0)
        kld_like = 0.5 * gamma_t * (
            term_const
            + torch.log(lambda_t + EPS)
            + torch.exp(z_log_var_t) / (lambda_t + EPS)
            + (z_mean_t - mu_t) ** 2 / (lambda_t + EPS)
        )
        kld_like = kld_like.sum(dim=(1, 2))

        z_entropy = -0.5 * torch.sum(z_log_var + 1.0, dim=1)
        cat_prior = -torch.sum(torch.log(theta + EPS).view(1, self.n_centroid) * gamma, dim=1)
        cat_entropy = torch.sum(torch.log(gamma + EPS) * gamma, dim=1)

        total = recon + kld_like + z_entropy + cat_prior + cat_entropy
        loss = total.mean()

        gamma_entropy = -(gamma * torch.log(gamma + EPS)).sum(dim=1).mean()
        theta_entropy = -(theta * torch.log(theta + EPS)).sum()
        cluster_usage = torch.bincount(torch.argmax(gamma, dim=1), minlength=self.n_centroid)
        cluster_usage_ratio = cluster_usage.float() / cluster_usage.sum().clamp_min(1).float()
        cluster_usage_entropy = -(cluster_usage_ratio * torch.log(cluster_usage_ratio + EPS)).sum()
        cluster_top1_ratio = cluster_usage_ratio.max()

        logs = {
            "loss": loss.detach(),
            "recon": recon.mean().detach(),
            "kld_like": kld_like.mean().detach(),
            "z_entropy": z_entropy.mean().detach(),
            "cat_prior": cat_prior.mean().detach(),
            "cat_entropy": cat_entropy.mean().detach(),
            "gamma_entropy": gamma_entropy.detach(),
            "theta_entropy": theta_entropy.detach(),
            "cluster_usage_entropy": cluster_usage_entropy.detach(),
            "cluster_top1_ratio": cluster_top1_ratio.detach(),
        }
        return loss, logs

    # 신경망 모델 파라미터
    def nn_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for module in [self.enc1, self.enc2, self.enc3, self.z_mean, self.z_log_var, self.dec1, self.dec2, self.dec3, self.x_bar]:
            params.extend(list(module.parameters()))
        return params
    # GMM 모델 파라미터
    def gmm_parameters(self) -> List[nn.Parameter]:
        return [self.pi_logits, self.mu_c, self.log_var_c]

# ============================================================================
# Training And Inference
# ============================================================================

def pretrain_autoencoder(features: np.ndarray, config: TrainConfig, device: torch.device, verbose: bool = True) -> StackedAutoEncoder:
    ae = StackedAutoEncoder(config.input_dim, config.latent_dim, config.hidden_dims, config.reconstruction).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=config.pretrain_lr, eps=1e-4)

    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=False)

    for epoch in range(config.pretrain_epochs):
        ae.train()
        running = 0.0
        n = 0
        for batch_x in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = ae(batch_x) # encode(x) -> decode(z) = recon
            if config.reconstruction == "sigmoid":
                loss = F.binary_cross_entropy(recon, batch_x, reduction="sum") / batch_x.shape[0]
            else:
                loss = F.mse_loss(recon, batch_x, reduction="sum") / batch_x.shape[0]
            loss.backward()
            optimizer.step()

            running += loss.detach().cpu().item() * batch_x.shape[0] # 연산그래프 분리 후 배치별 loss 합계 누적
            n += batch_x.shape[0] # n : epoch 내 총 샘플 수 계산

        if verbose: 
            print(f"[pretrain] epoch={epoch + 1}/{config.pretrain_epochs} loss={running / max(n, 1):.6f}") # epoch 평균 loss
    return ae


def copy_pretrain_weights(vade: VaDE, ae: StackedAutoEncoder) -> None:
    with torch.no_grad():
        vade.enc1.weight.copy_(ae.enc1.weight)
        vade.enc1.bias.copy_(ae.enc1.bias)
        vade.enc2.weight.copy_(ae.enc2.weight)
        vade.enc2.bias.copy_(ae.enc2.bias)
        vade.enc3.weight.copy_(ae.enc3.weight)
        vade.enc3.bias.copy_(ae.enc3.bias)
        vade.z_mean.weight.copy_(ae.enc4.weight)
        vade.z_mean.bias.copy_(ae.enc4.bias)

        vade.dec1.weight.copy_(ae.dec1.weight)
        vade.dec1.bias.copy_(ae.dec1.bias)
        vade.dec2.weight.copy_(ae.dec2.weight)
        vade.dec2.bias.copy_(ae.dec2.bias)
        vade.dec3.weight.copy_(ae.dec3.weight)
        vade.dec3.bias.copy_(ae.dec3.bias)
        vade.x_bar.weight.copy_(ae.dec4.weight)
        vade.x_bar.bias.copy_(ae.dec4.bias)


def encode_dataset(model: VaDE, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)

    out: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x in loader: 
            z_mean, _ = model.encode(batch_x.to(device))
            out.append(z_mean.cpu().numpy())
    return np.concatenate(out, axis=0)

# 본학습 전, GMM init
def initialize_gmm_parameters(model: VaDE, embeddings: np.ndarray, dataset: str) -> None:
    with torch.no_grad():
        model.pi_logits.fill_(0.0)
        model.mu_c.zero_()
        model.log_var_c.zero_()

    if dataset in {"mnist", "har", "reuters_all"}:
        random_state = 3 if dataset == "har" else 0

        gmm = GaussianMixture(n_components=model.n_centroid, covariance_type="diag", random_state=random_state)
        gmm.fit(embeddings) # 내부에서 EM 알고리즘 반복 -> 로그우도가 더 이상 크게 안 늘 때까지 수렴
        with torch.no_grad():
            model.mu_c.copy_(torch.from_numpy(gmm.means_.astype(np.float32)))
            model.log_var_c.copy_(torch.log(torch.from_numpy(gmm.covariances_.astype(np.float32)) + EPS))
            model.pi_logits.copy_(torch.log(torch.from_numpy(gmm.weights_.astype(np.float32)) + EPS))
    elif dataset == "reuters10k":
        kmeans = KMeans(n_clusters=model.n_centroid, random_state=0, n_init=20)
        kmeans.fit(embeddings)
        with torch.no_grad():
            model.mu_c.copy_(torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)))
    else:
        raise ValueError(f"Unsupported dataset for gmm init: {dataset}")


def predict_gamma(
    model: VaDE,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
    use_mean: bool = True,
) -> np.ndarray:
    model.eval() # evaluation mode
    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)

    out: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            z_mean, z_log_var = model.encode(batch_x)
            z = z_mean if use_mean else model.reparameterize(z_mean, z_log_var) # z sampling
            gamma = model.compute_gamma(z) # z: (B, J) -> gamma: (B, K)
            out.append(gamma.cpu().numpy()) # out list는 배치 단위로 gamma를 쪼개서 저장
    return np.concatenate(out, axis=0) # 최종 shape : (N, K)


def lr_decay_step(optimizer: torch.optim.Optimizer, dataset: str, decay_nn: float, decay_gmm: float) -> Tuple[float, float]:
    nn_lr = optimizer.param_groups[0]["lr"]
    gmm_lr = optimizer.param_groups[1]["lr"]

    if dataset == "mnist":
        nn_lr = max(nn_lr * decay_nn, 0.0002)
        gmm_lr = max(gmm_lr * decay_gmm, 0.0002)
    else:
        nn_lr = nn_lr * decay_nn
        gmm_lr = gmm_lr * decay_gmm

    optimizer.param_groups[0]["lr"] = nn_lr
    optimizer.param_groups[1]["lr"] = gmm_lr
    return nn_lr, gmm_lr


def should_record_training_step(global_step: int) -> bool:
    return global_step % 50 == 0


def append_step_history(
    history: Dict[str, List[Any]],
    global_step: int,
    epoch_index: int,
    step_in_epoch: int,
    logs: Dict[str, Any],
) -> None:
    step_metrics = torch.stack(
        [
            logs["gamma_entropy"],
            logs["theta_entropy"],
            logs["cluster_usage_entropy"],
            logs["cluster_top1_ratio"],
            logs["recon"],
            logs["kld_like"],
        ]
    ).detach().cpu().tolist()

    history["step"].append(global_step)
    history["epoch"].append(epoch_index + 1)
    history["step_in_epoch"].append(step_in_epoch)
    history["batch_gamma_entropy"].append(float(step_metrics[0]))
    history["batch_theta_entropy"].append(float(step_metrics[1]))
    history["batch_cluster_usage_entropy"].append(float(step_metrics[2]))
    history["batch_cluster_top1_ratio"].append(float(step_metrics[3]))
    history["batch_recon"].append(float(step_metrics[4]))
    history["batch_kld_like"].append(float(step_metrics[5]))

# ============================================================================
# Checkpoint
# ============================================================================

def save_checkpoint(path: str, model: VaDE, config: TrainConfig, history: Dict[str, List[Any]]) -> None:
    payload = {
        "model_state": model.state_dict(), # 학습 완료된 모든 파라미터 값 저장
        "model_kwargs": {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "n_centroid": model.n_centroid,
            "hidden_dims": list(model.hidden_dims),
            "reconstruction": model.reconstruction,
        }, # 모델을 다시 만들기 위한 설정
        "train_config": asdict(config), # 학습에 사용한 config 전체
        "history": history, # 학습과정 로그
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def load_checkpoint(path: str, device: torch.device) -> Tuple[VaDE, Dict]:
    payload = torch.load(path, map_location=device)
    model = VaDE(**payload["model_kwargs"]).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload


def posterior_numpy(
    z: np.ndarray,
    theta: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)

    n_centroid, latent_dim = mu.shape
    z_expand = np.repeat(z.reshape(1, latent_dim), n_centroid, axis=0)

    log_theta = np.log(theta + EPS)

    log_prob = (
        log_theta
        - 0.5 * np.sum(np.log(2.0 * math.pi * var + EPS), axis=1)
        - 0.5 * np.sum((z_expand - mu) ** 2 / (var + EPS), axis=1)
    )
    log_prob = log_prob - np.max(log_prob)
    prob = np.exp(log_prob)
    return prob / np.sum(prob)


def reverse_assignment(assignment: np.ndarray) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for cluster_id, label_id in assignment:
        mapping[int(label_id)] = int(cluster_id)
    return mapping

# ============================================================================
# Public API
# ============================================================================

# get_default_config에서 데이터셋마다 다른 초기화 값이 TrainConfig 클래스를 호출하여 입력되고, VaDE 클래스의 파라미터로 입력됨
def build_model_from_config(config: TrainConfig) -> VaDE:
    return VaDE(config.input_dim, config.latent_dim, config.n_centroid, config.hidden_dims, config.reconstruction)

# 기본 config + 사용자 옵션 적용
def override_config(config: TrainConfig, args: Dict[str, Optional[float]]) -> TrainConfig:
    out = copy.deepcopy(config)
    for key, value in args.items():
        if value is not None and hasattr(out, key):
            setattr(out, key, value)
    return out


def train_vade(
    model: VaDE,
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
    device: torch.device,
    load_pretrained_ae: bool = True,
    eval_use_mean: bool = False,
) -> Dict[str, List[Any]]:
    history: Dict[str, List[Any]] = {
        "step": [],
        "epoch": [],
        "step_in_epoch": [],

        "batch_gamma_entropy": [],
        "batch_theta_entropy": [],
        "batch_cluster_usage_entropy": [],
        "batch_cluster_top1_ratio": [],
        "batch_recon": [],
        "batch_kld_like": [],

        "loss": [],
        "recon": [],
        "kld_like": [],
        "z_entropy": [],
        "cat_prior": [],
        "cat_entropy": [],
        "acc": [],
        "lr_nn": [],
        "lr_gmm": [],
    }

    if load_pretrained_ae:
        ae = load_author_pretrained_autoencoder(config, device=device)
        print(f"author pretrain weights loaded: {get_author_pretrain_weight_path(config.dataset)}")
        copy_pretrain_weights(model, ae)
    else:
        ae = pretrain_autoencoder(features, config, device=device)
        copy_pretrain_weights(model, ae)

    embeddings = encode_dataset(model, features, config.batch_size, device) # model.encode(x)로부터 z_mean만 뽑아서 모으기 / shape: (N, J)
    initialize_gmm_parameters(model, embeddings, config.dataset) 

    optimizer = torch.optim.Adam(
        [
            {"params": model.nn_parameters(), "lr": config.lr_nn},
            {"params": model.gmm_parameters(), "lr": config.lr_gmm},
        ],
        eps=1e-4,
    )

    data = torch.from_numpy(features.astype(np.float32))
    loader = torch.utils.data.DataLoader(data, batch_size=config.batch_size, shuffle=True, drop_last=False)

    global_step = 0 # 총 학습 step 계산 변수

    for epoch in range(config.epochs):
        if epoch % config.decay_n == 0 and epoch != 0:
            lr_nn, lr_gmm = lr_decay_step(optimizer, config.dataset, config.decay_nn, config.decay_gmm)
            print(f"lr_nn: {lr_nn:.8f} | lr_gmm: {lr_gmm:.8f}")

        model.train()

        run_loss = torch.zeros((), device=device)
        run_recon = torch.zeros((), device=device)
        run_kld_like = torch.zeros((), device=device)
        run_z_entropy = torch.zeros((), device=device)
        run_cat_prior = torch.zeros((), device=device)
        run_cat_entropy = torch.zeros((), device=device)
        n = 0

        for step_in_epoch, batch_x in enumerate(loader, start=1): # batch_x : shape (B, D) tensor, dtype float32
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_hat, z_mean, z_log_var, z = model(batch_x) # forward 호출
            loss, loss_logs = model.vade_loss(batch_x, x_hat, z, z_mean, z_log_var, config.alpha)
            loss.backward()
            optimizer.step()
            global_step += 1

            # per 50 steps 
            if should_record_training_step(global_step):
                append_step_history(history, global_step, epoch, step_in_epoch, loss_logs)

            batch_size_now = batch_x.shape[0]
            run_loss += loss.detach() * batch_size_now
            run_recon += loss_logs["recon"] * batch_size_now
            run_kld_like += loss_logs["kld_like"] * batch_size_now
            run_z_entropy += loss_logs["z_entropy"] * batch_size_now
            run_cat_prior += loss_logs["cat_prior"] * batch_size_now
            run_cat_entropy += loss_logs["cat_entropy"] * batch_size_now
            n += batch_size_now

        # logs per epoch : gpu->cpu
        epoch_metrics = (
            torch.stack([run_loss, run_recon, run_kld_like, run_z_entropy, run_cat_prior, run_cat_entropy])
            / max(n, 1)
        ).detach().cpu().tolist() 
        epoch_loss = float(epoch_metrics[0])
        epoch_recon = float(epoch_metrics[1])
        epoch_kld_like = float(epoch_metrics[2])
        epoch_z_entropy = float(epoch_metrics[3])
        epoch_cat_prior = float(epoch_metrics[4])
        epoch_cat_entropy = float(epoch_metrics[5])

        # 매 에폭마다 정확도 계산을 위한 코드
        gamma = predict_gamma(model, features, config.batch_size, device, eval_use_mean)
        y_pred = np.argmax(gamma, axis=1) # y_pred : (N,)
        acc, _, _ = cluster_acc(y_pred, labels)

        history["loss"].append(float(epoch_loss))
        history["recon"].append(float(epoch_recon))
        history["kld_like"].append(float(epoch_kld_like))
        history["z_entropy"].append(float(epoch_z_entropy))
        history["cat_prior"].append(float(epoch_cat_prior))
        history["cat_entropy"].append(float(epoch_cat_entropy))
        history["acc"].append(float(acc))
        history["lr_nn"].append(float(optimizer.param_groups[0]["lr"]))
        history["lr_gmm"].append(float(optimizer.param_groups[1]["lr"]))

        print(
            f"epoch {epoch + 1:04d}/{config.epochs} | loss={epoch_loss:.6f} | "
            f"acc_p_c_z={acc:.6f} | lr_nn={optimizer.param_groups[0]['lr']:.8f} | "
            f"lr_gmm={optimizer.param_groups[1]['lr']:.8f}"
        )

        if epoch == 1 and config.dataset == "har" and acc < 0.77:
            raise RuntimeError("HAR dataset bad init (acc < 0.77 at epoch 2). Please run again.")

    return history
