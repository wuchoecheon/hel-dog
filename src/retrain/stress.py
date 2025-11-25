"""
CNN-GRU 기반 스트레스 분류 모델 파인튜닝 스크립트.

백엔드에서 `export_stress_retrain_npy` 로 추출한 `.npy` 데이터를
시퀀스 형태로 묶어 기존 체크포인트(`cnn_gru_phase2_best.pt`)를
미세 조정한다. 라벨이 포함되지 않은 경우 HR/HRV/ACC 기준의
휴리스틱 라벨을 자동 생성해 준다.

예시:
    python -m src.stress.retrain \
        --data-path /data/stress_202501.npy \
        --sequence-length 60 \
        --output-path /data/cnn_gru_phase2_finetuned.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger("stress.retrain")

FEATURE_NAMES: List[str] = [
    "heart_rate_bpm",
    "hrv_sdnn_ms",
    "hrv_rmssd_ms",
    "acc_x_mean",
    "acc_y_mean",
    "acc_z_mean",
    "acc_mag_mean",
    "acc_mag_std",
]
FEATURE_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(FEATURE_NAMES)}


def _default_checkpoint_path() -> Path:
    return Path(__file__).with_name("cnn_gru_phase2_best.pt")


@dataclass
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        if values.ndim != 2:
            raise ValueError(f"Feature array must be 2-D, got shape {values.shape}")

        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        mean = np.nan_to_num(mean, nan=0.0).astype(np.float32)
        std = np.nan_to_num(std, nan=1.0).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != self.mean.shape[0]:
            raise ValueError("Incompatible feature dimension during normalization")
        normalized = (values - self.mean) / self.std
        return np.nan_to_num(normalized, nan=0.0).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_checkpoint(cls, payload: Dict[str, Iterable[float]]) -> "FeatureNormalizer":
        mean = np.asarray(payload["mean"], dtype=np.float32)
        std = np.asarray(payload["std"], dtype=np.float32)
        return cls(mean=mean, std=std)


class CNNGRUStressClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cnn_channels: int = 64,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.GRU(
            input_size=cnn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        classifier_layers: List[nn.Module] = [
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, 1),
        ]
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, seq_len, feature_dim)
        x = inputs.transpose(1, 2)  # -> (batch, feature_dim, seq_len)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)  # -> (batch, seq_len, cnn_channels)
        gru_out, _ = self.temporal_encoder(x)
        pooled = gru_out[:, -1, :]
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the CNN-GRU stress classifier using backend-exported npy files."
    )
    parser.add_argument("--data-path", required=True, type=Path, help="학습에 사용할 .npy 경로")
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=_default_checkpoint_path(),
        help="기존 파라미터(.pt) 경로",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=_default_checkpoint_path().with_name("cnn_gru_phase2_finetuned.pt"),
        help="파인튜닝 후 저장할 경로",
    )
    parser.add_argument("--labels-path", type=Path, default=None, help="라벨(.npy) 경로 (선택)")
    parser.add_argument(
        "--label-column-index",
        type=int,
        default=None,
        help="데이터 내 라벨 컬럼 위치 (기본: 없음)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="시퀀스 길이(타임스텝 수)",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=5,
        help="슬라이딩 윈도 stride",
    )
    parser.add_argument(
        "--window-label-strategy",
        choices=("max", "majority", "last"),
        default="max",
        help="윈도 라벨 결정 방식",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="검증 데이터 비율")
    parser.add_argument("--batch-size", type=int, default=64, help="배치 크기")
    parser.add_argument("--epochs", type=int, default=20, help="에폭 수")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="학습률")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--max-grad-norm", type=float, default=1.5, help="gradient clipping 값")
    parser.add_argument("--patience", type=int, default=5, help="조기 종료 patience (에폭)")
    parser.add_argument("--seed", type=int, default=2024, help="난수 시드")
    parser.add_argument(
        "--device",
        default="auto",
        help="'cuda', 'cpu' 혹은 'auto'",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    parser.add_argument("--history-path", type=Path, default=None, help="학습 히스토리 JSON 저장 경로")
    parser.add_argument(
        "--freeze-cnn",
        action="store_true",
        help="Conv 블록을 고정하고 GRU/FC만 학습",
    )
    parser.add_argument(
        "--freeze-gru",
        action="store_true",
        help="GRU를 고정하고 Conv/FC만 학습",
    )
    # Heuristic label options
    parser.add_argument("--hr-threshold", type=float, default=95.0, help="휴리스틱: HR 임계값")
    parser.add_argument("--sdnn-threshold", type=float, default=30.0, help="휴리스틱: SDNN 이하")
    parser.add_argument("--rmssd-threshold", type=float, default=25.0, help="휴리스틱: RMSSD 이하")
    parser.add_argument("--acc-mean-threshold", type=float, default=0.85, help="휴리스틱: ACC MAG 평균 이상")
    parser.add_argument("--acc-std-threshold", type=float, default=0.18, help="휴리스틱: ACC MAG 표준편차 이상")
    parser.add_argument(
        "--heuristic-votes",
        type=int,
        default=2,
        help="휴리스틱 라벨에서 스트레스 판정에 필요한 vote 수",
    )

    args = parser.parse_args()
    args.data_path = args.data_path.expanduser().resolve()
    args.base_model_path = args.base_model_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    if args.labels_path:
        args.labels_path = args.labels_path.expanduser().resolve()
    if args.history_path:
        args.history_path = args.history_path.expanduser().resolve()
    if not 0.0 < args.val_split < 0.5:
        raise ValueError("val-split 은 0과 0.5 사이여야 합니다.")
    if args.sequence_length <= 0:
        raise ValueError("sequence-length 는 1 이상이어야 합니다.")
    if args.sequence_stride <= 0:
        raise ValueError("sequence-stride 는 1 이상이어야 합니다.")
    if args.heuristic_votes <= 0:
        raise ValueError("heuristic-votes 는 1 이상이어야 합니다.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def load_features(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {path}")
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.ndim == 0 and isinstance(arr.item(), dict):
        payload = arr.item()
        if "features" not in payload:
            raise ValueError("Object npy must contain 'features' key")
        features = np.asarray(payload["features"], dtype=np.float32)
        if "labels" in payload:
            payload_labels = np.asarray(payload["labels"], dtype=np.float32)
            return np.column_stack([features, payload_labels])
        return features
    if arr.ndim == 1:
        return arr.reshape(1, -1).astype(np.float32)
    return arr.astype(np.float32)


def extract_labels(
    features: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    labels: Optional[np.ndarray] = None
    feature_matrix = features

    if args.labels_path:
        labels = np.load(args.labels_path).astype(np.float32).reshape(-1)
    elif args.label_column_index is not None:
        col = args.label_column_index
        if col < 0:
            col += features.shape[1]
        if col < 0 or col >= features.shape[1]:
            raise ValueError("label-column-index 가 범위를 벗어났습니다.")
        labels = features[:, col].astype(np.float32).reshape(-1)
        feature_matrix = np.delete(features, col, axis=1)

    if labels is None:
        LOGGER.warning("라벨이 제공되지 않아 휴리스틱 기반으로 생성합니다.")
        labels = derive_labels_from_heuristics(feature_matrix, args)

    if feature_matrix.shape[1] != len(FEATURE_NAMES):
        LOGGER.warning(
            "특성 수가 예상(%d)과 다릅니다. 실제=%d",
            len(FEATURE_NAMES),
            feature_matrix.shape[1],
        )

    if feature_matrix.shape[0] != labels.shape[0]:
        raise ValueError("특성과 라벨의 샘플 수가 일치하지 않습니다.")

    return feature_matrix.astype(np.float32), labels.astype(np.float32)


def derive_labels_from_heuristics(features: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    labels = []
    for row in features:
        votes = 0
        hr = row[FEATURE_INDEX["heart_rate_bpm"]]
        sdnn = row[FEATURE_INDEX["hrv_sdnn_ms"]]
        rmssd = row[FEATURE_INDEX["hrv_rmssd_ms"]]
        acc_mean = row[FEATURE_INDEX["acc_mag_mean"]]
        acc_std = row[FEATURE_INDEX["acc_mag_std"]]

        if hr >= args.hr_threshold:
            votes += 1
        if sdnn <= args.sdnn_threshold:
            votes += 1
        if rmssd <= args.rmssd_threshold:
            votes += 1
        if acc_mean >= args.acc_mean_threshold:
            votes += 1
        if acc_std >= args.acc_std_threshold:
            votes += 1

        labels.append(1.0 if votes >= args.heuristic_votes else 0.0)

    labels_arr = np.asarray(labels, dtype=np.float32)
    pos_ratio = labels_arr.mean()
    LOGGER.info(
        "휴리스틱 라벨 생성 완료: 총 %d개, 스트레스 비율=%.3f",
        labels_arr.shape[0],
        pos_ratio,
    )
    if pos_ratio in (0.0, 1.0):
        LOGGER.warning("라벨이 한쪽 클래스에 치우쳐 있습니다. 임계값을 조정하세요.")
    return labels_arr


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    stride: int,
    label_strategy: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if features.shape[0] < seq_len:
        raise ValueError("시퀀스 길이가 전체 샘플 수보다 깁니다.")
    windows = []
    window_labels = []
    for start in range(0, features.shape[0] - seq_len + 1, stride):
        end = start + seq_len
        windows.append(features[start:end])
        window_labels.append(reduce_window_label(labels[start:end], label_strategy))
    if not windows:
        raise ValueError("생성된 윈도가 없습니다. stride/seq_len 을 확인하세요.")
    X = np.stack(windows).astype(np.float32)
    y = np.asarray(window_labels, dtype=np.float32)
    return X, y


def reduce_window_label(values: np.ndarray, strategy: str) -> float:
    if strategy == "max":
        return float(values.max())
    if strategy == "majority":
        return float((values.mean() >= 0.5))
    if strategy == "last":
        return float(values[-1])
    raise ValueError(f"Unknown window label strategy: {strategy}")


def split_datasets(
    sequences: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[TensorDataset, TensorDataset]:
    total = sequences.shape[0]
    if total < 2:
        raise ValueError("학습/검증으로 나눌 수 있을 만큼 윈도가 부족합니다.")
    val_count = max(1, int(total * val_ratio))
    train_count = total - val_count
    if train_count == 0:
        raise ValueError("훈련 샘플이 없습니다. val-split 을 줄이세요.")

    indices = np.arange(total)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    train_idx = indices[:train_count]
    val_idx = indices[train_count:]

    train_dataset = TensorDataset(
        torch.from_numpy(sequences[train_idx]),
        torch.from_numpy(labels[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(sequences[val_idx]),
        torch.from_numpy(labels[val_idx]),
    )
    return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def compute_class_weights(train_dataset: TensorDataset, device: torch.device) -> Optional[torch.Tensor]:
    labels = train_dataset.tensors[1].numpy()
    pos_count = float(labels.sum())
    neg_count = float(labels.shape[0] - pos_count)
    if pos_count == 0 or neg_count == 0:
        LOGGER.warning("단일 클래스만 존재하여 pos_weight 를 사용할 수 없습니다.")
        return None
    pos_weight = neg_count / pos_count
    return torch.tensor(pos_weight, device=device, dtype=torch.float32)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    pos_weight = compute_class_weights(train_loader.dataset, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            if args.max_grad_norm:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            total_count += batch_x.size(0)

        train_loss = total_loss / max(1, total_count)
        val_loss, metrics = evaluate(model, val_loader, criterion, device)

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(metrics["accuracy"]),
            "val_precision": float(metrics["precision"]),
            "val_recall": float(metrics["recall"]),
            "val_f1": float(metrics["f1"]),
        }
        history.append(epoch_record)

        LOGGER.info(
            "[Epoch %02d] train_loss=%.4f val_loss=%.4f acc=%.3f f1=%.3f",
            epoch,
            train_loss,
            val_loss,
            metrics["accuracy"],
            metrics["f1"],
        )

        if val_loss + 1e-4 < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if args.patience and patience_counter >= args.patience:
                LOGGER.info("조기 종료 발동 (patience=%d)", args.patience)
                break

    model.load_state_dict(best_state)
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    probs_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            total_count += batch_x.size(0)
            probs_list.append(torch.sigmoid(logits).cpu())
            targets_list.append(batch_y.cpu())
    avg_loss = total_loss / max(1, total_count)
    all_probs = torch.cat(probs_list)
    all_targets = torch.cat(targets_list)
    metrics = compute_metrics(all_probs, all_targets)
    return avg_loss, metrics


def compute_metrics(probs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    preds = (probs >= 0.5).int()
    targets_int = targets.int()
    tp = int(((preds == 1) & (targets_int == 1)).sum().item())
    tn = int(((preds == 0) & (targets_int == 0)).sum().item())
    fp = int(((preds == 1) & (targets_int == 0)).sum().item())
    fn = int(((preds == 0) & (targets_int == 1)).sum().item())
    total = max(1, tp + tn + fp + fn)
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def maybe_load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> Optional[FeatureNormalizer]:
    if not checkpoint_path.exists():
        LOGGER.warning("기존 체크포인트(%s)를 찾을 수 없습니다.", checkpoint_path)
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = None
    normalizer: Optional[FeatureNormalizer] = None

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint

        if "feature_mean" in checkpoint and "feature_std" in checkpoint:
            normalizer = FeatureNormalizer.from_checkpoint(
                {"mean": checkpoint["feature_mean"], "std": checkpoint["feature_std"]}
            )
        elif "normalizer" in checkpoint and isinstance(checkpoint["normalizer"], dict):
            normalizer = FeatureNormalizer.from_checkpoint(checkpoint["normalizer"])
    elif isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            LOGGER.warning("체크포인트에 없는 파라미터: %s", missing)
        if unexpected:
            LOGGER.warning("모델에 존재하지 않는 파라미터: %s", unexpected)
        LOGGER.info("체크포인트(%s)를 로드했습니다.", checkpoint_path.name)
    else:
        LOGGER.warning("체크포인트 로드에 실패했습니다.")

    return normalizer


def save_checkpoint(
    model: nn.Module,
    normalizer: FeatureNormalizer,
    history: List[Dict[str, float]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_mean": normalizer.mean.tolist(),
        "feature_std": normalizer.std.tolist(),
        "history": history,
        "config": {
            "sequence_length": args.sequence_length,
            "sequence_stride": args.sequence_stride,
            "window_label_strategy": args.window_label_strategy,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output_path)
    LOGGER.info("새 체크포인트를 %s 에 저장했습니다.", args.output_path)

    if args.history_path:
        args.history_path.parent.mkdir(parents=True, exist_ok=True)
        args.history_path.write_text(json.dumps(history, indent=2))
        LOGGER.info("학습 히스토리를 %s 에 저장했습니다.", args.history_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    LOGGER.info("사용 디바이스: %s", device)

    raw_features = load_features(args.data_path)
    feature_matrix, labels = extract_labels(raw_features, args)

    normalizer = FeatureNormalizer.fit(feature_matrix)
    model = CNNGRUStressClassifier(input_dim=feature_matrix.shape[1])
    checkpoint_normalizer = maybe_load_checkpoint(model, args.base_model_path, device)
    if checkpoint_normalizer:
        LOGGER.info("기존 정규화 통계를 재사용합니다.")
        normalizer = checkpoint_normalizer

    normalized_features = normalizer.transform(feature_matrix)
    sequences, window_labels = build_windows(
        normalized_features,
        labels,
        args.sequence_length,
        args.sequence_stride,
        args.window_label_strategy,
    )
    train_dataset, val_dataset = split_datasets(
        sequences,
        window_labels,
        args.val_split,
        args.seed,
    )
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )

    model.to(device)

    if args.freeze_cnn:
        freeze_module(model.feature_extractor)
    if args.freeze_gru:
        freeze_module(model.temporal_encoder)

    history = train_model(model, train_loader, val_loader, device, args)
    save_checkpoint(model, normalizer, history, args)


if __name__ == "__main__":
    main()
