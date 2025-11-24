from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

from src.stress import retrain as stress_retrain

BASE_MODEL_DIR = Path("/app/model")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SLEEP_BASE_MODEL = PROJECT_ROOT / "sleep" / "exp_conv_bilstm_base_best.keras"
STRESS_BASE_MODEL = PROJECT_ROOT / "stress" / "cnn_gru_phase2_best.pt"


def build_sleep_windows(
    raw_matrix: np.ndarray,
    *,
    window_size: int = 5,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TIMESTAMP + ACC XYZ + HR + Stage 형태의 sleep npy를 conv+BiLSTM 윈도우 데이터로 변환.
    레이블은 윈도우 내 Stage의 다수결(깊은 잠 분리 없이 Wake vs Sleep).
    """
    if raw_matrix.ndim != 2 or raw_matrix.shape[1] < 6:
        raise ValueError("sleep npy는 (N, >=6) 형태여야 합니다.")

    features = np.nan_to_num(raw_matrix[:, 1:5], nan=0.0).astype(np.float32)
    stages = raw_matrix[:, 5].astype(np.int32)

    windows: list[np.ndarray] = []
    labels: list[int] = []

    for start in range(0, len(features) - window_size + 1, stride):
        window_feat = features[start:start + window_size]
        window_stage = stages[start:start + window_size]
        if np.isnan(window_feat).any():
            continue
        windows.append(window_feat)
        wake_ratio = np.mean(window_stage == 0)
        labels.append(0 if wake_ratio >= 0.5 else 1)

    if not windows:
        raise ValueError("수면 윈도우를 생성할 수 없습니다. 입력 데이터를 확인하세요.")

    return np.stack(windows), np.asarray(labels, dtype=np.float32)


class SlidingWindowReconstructor:
    def __init__(self, window_size: int, stride: int, smooth_kernel: int = 5):
        self.window_size = window_size
        self.stride = max(1, stride)
        self.smooth_kernel = smooth_kernel if smooth_kernel > 1 else 1

    def reconstruct(self, window_probs: np.ndarray, total_length: int) -> np.ndarray:
        preds_sum = np.zeros(total_length, dtype=np.float32)
        counts = np.zeros(total_length, dtype=np.float32)

        for idx, prob in enumerate(window_probs):
            start = idx * self.stride
            end = min(start + self.window_size, total_length)
            if start >= total_length or end <= start:
                continue
            preds_sum[start:end] += prob
            counts[start:end] += 1.0

        counts[counts == 0] = 1.0
        reconstructed = preds_sum / counts

        if self.smooth_kernel > 1 and len(reconstructed) >= self.smooth_kernel:
            kernel = np.ones(self.smooth_kernel, dtype=np.float32) / self.smooth_kernel
            reconstructed = np.convolve(reconstructed, kernel, mode="same")

        return reconstructed


class SleepQualityFeatureExtractor:
    def __init__(self, minute_resolution: int = 1, min_wake_block_minutes: int = 5):
        self.minute_resolution = max(1, minute_resolution)
        self.min_wake_block = max(1, int(np.ceil(min_wake_block_minutes / self.minute_resolution)))

    def extract(self, session: Dict[str, np.ndarray]) -> Dict[str, float]:
        length_minutes = int(session["length_minutes"])
        acc_x = session["acc_x"][:length_minutes]
        acc_y = session["acc_y"][:length_minutes]
        acc_z = session["acc_z"][:length_minutes]
        hr = session["hr"][:length_minutes]
        sleep_wake = session["sleep_wake_pred"][:length_minutes]

        total_minutes = length_minutes
        sleep_minutes = int(np.sum(sleep_wake == 1) * self.minute_resolution)
        wake_minutes = max(0, total_minutes - sleep_minutes)

        features = {
            "total_time_minutes": total_minutes,
            "sleep_time_minutes": sleep_minutes,
            "wake_time_minutes": wake_minutes,
            "sleep_efficiency": self._sleep_efficiency(sleep_wake),
            "sleep_stability": self._sleep_stability(sleep_wake),
            "movement_index": self._movement_index(acc_x, acc_y, acc_z, sleep_wake),
            "waso_minutes": self._compute_waso(sleep_wake),
            "hr_mean": self._hr_mean(hr, sleep_wake),
            "hr_recovery_index": self._hr_recovery(hr, sleep_wake),
        }
        return features

    def _sleep_efficiency(self, sleep_wake: np.ndarray) -> float:
        if len(sleep_wake) == 0:
            return 0.0
        return round(float(np.mean(sleep_wake == 1) * 100), 2)

    def _sleep_stability(self, sleep_wake: np.ndarray) -> float:
        if len(sleep_wake) < 2:
            return 0.0
        transitions = np.sum(np.diff(sleep_wake) != 0)
        stability = (1 - transitions / max(1, len(sleep_wake))) * 100
        return round(float(max(0.0, stability)), 2)

    def _movement_index(self, acc_x, acc_y, acc_z, sleep_wake) -> float:
        mask = sleep_wake == 1
        if np.sum(mask) < 10:
            return 0.0
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)[mask]
        if len(acc_mag) < 10:
            return 0.0
        p90 = np.percentile(acc_mag, 90)
        p10 = np.percentile(acc_mag, 10)
        movement_range = max(0.0, p90 - p10)
        return float(np.clip(movement_range * 80, 0, 100))

    def _compute_waso(self, sleep_wake: np.ndarray) -> float:
        sleep_indices = np.where(sleep_wake == 1)[0]
        if len(sleep_indices) == 0:
            return 0.0
        sleep_onset = sleep_indices[0]
        last_sleep = sleep_indices[-1]
        sleep_period = sleep_wake[sleep_onset:last_sleep + 1]

        waso_minutes = 0
        block = 0
        for state in sleep_period:
            if state == 0:
                block += 1
            elif block:
                if block >= self.min_wake_block:
                    waso_minutes += block * self.minute_resolution
                block = 0
        if block >= self.min_wake_block:
            waso_minutes += block * self.minute_resolution
        return float(waso_minutes)

    def _hr_mean(self, hr: np.ndarray, sleep_wake: np.ndarray) -> float:
        mask = sleep_wake == 1
        if np.sum(mask) < 10:
            return 0.0
        hr_sleep = hr[mask]
        hr_sleep = hr_sleep[(hr_sleep >= 30) & (hr_sleep <= 120)]
        if len(hr_sleep) < 10:
            return 0.0
        return round(float(np.mean(hr_sleep)), 2)

    def _hr_recovery(self, hr: np.ndarray, sleep_wake: np.ndarray) -> float:
        indices = np.where(sleep_wake == 1)[0]
        if len(indices) < 30:
            return 0.0
        window = max(10, len(indices) // 4)
        start_mean = np.mean(hr[indices[:window]])
        end_mean = np.mean(hr[indices[-window:]])
        drop = max(0.0, start_mean - end_mean)
        return round(float(np.clip(drop * 8, 0, 100)), 2)


class SleepQualityCalculator:
    def __init__(self):
        self.weights = {
            "sleep_efficiency": 0.35,
            "sleep_stability": 0.2,
            "movement_index": 0.1,
            "waso_minutes": 0.1,
            "hr_mean": 0.1,
            "hr_recovery_index": 0.15,
        }

    def calculate(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {}
        scores["efficiency_score"] = float(np.clip(features["sleep_efficiency"], 0, 100))
        scores["stability_score"] = float(np.clip(features["sleep_stability"], 0, 100))
        movement = max(0.0, features["movement_index"])
        scores["movement_score"] = float(np.clip(100 - movement, 0, 100))
        waso = max(0.0, features["waso_minutes"])
        scores["waso_score"] = float(np.clip(100 - waso * 0.5, 0, 100))
        hr_mean = max(40.0, min(90.0, features["hr_mean"]))
        scores["hr_mean_score"] = float(np.interp(hr_mean, [40, 60, 80, 90], [95, 100, 70, 50]))
        scores["recovery_score"] = float(np.clip(features["hr_recovery_index"], 0, 100))

        total = (
            scores["efficiency_score"] * self.weights["sleep_efficiency"]
            + scores["stability_score"] * self.weights["sleep_stability"]
            + scores["movement_score"] * self.weights["movement_index"]
            + scores["waso_score"] * self.weights["waso_minutes"]
            + scores["hr_mean_score"] * self.weights["hr_mean"]
            + scores["recovery_score"] * self.weights["hr_recovery_index"]
        )
        total = round(float(np.clip(total, 0, 100)), 2)
        scores["total_score"] = total
        return scores


def summarize_sleep_quality(
    sleep_matrix: np.ndarray,
    window_probs: np.ndarray,
    *,
    window_size: int = 5,
    stride: int = 1,
) -> Dict[str, float]:
    features = np.nan_to_num(sleep_matrix[:, 1:5], nan=0.0).astype(np.float32)
    reconstructor = SlidingWindowReconstructor(window_size=window_size, stride=stride, smooth_kernel=5)
    minute_probs = reconstructor.reconstruct(window_probs, total_length=len(features))
    minute_binary = (minute_probs >= 0.5).astype(int)

    session = {
        "length_minutes": len(minute_binary),
        "sleep_wake_pred": minute_binary,
        "acc_x": features[:, 0],
        "acc_y": features[:, 1],
        "acc_z": features[:, 2],
        "hr": features[:, 3],
    }
    extractor = SleepQualityFeatureExtractor(minute_resolution=1)
    calculator = SleepQualityCalculator()
    feature_summary = extractor.extract(session)
    score_summary = calculator.calculate(feature_summary)
    return {**feature_summary, **score_summary}


def run_sleep_retrain(
    sleep_matrix: np.ndarray,
    *,
    year: int,
    month: int,
    base_dir: Optional[Path] = None,
    seed: int = 2024,
) -> Dict[str, object]:
    import tensorflow as tf
    from tensorflow import keras

    print("[SLEEP] conv+BiLSTM 재학습 시작")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    sequences, labels = build_sleep_windows(sleep_matrix, window_size=5, stride=1)

    num_samples = len(sequences)
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    split_idx = max(1, int(num_samples * 0.85))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:] if split_idx < num_samples else indices[-1:]

    X_train, y_train = sequences[train_idx], labels[train_idx]
    X_val, y_val = sequences[val_idx], labels[val_idx]

    n_wake = max(1, np.sum(y_train == 0))
    n_sleep = max(1, np.sum(y_train == 1))
    total = len(y_train)
    class_weight = {
        0: total / (2 * n_wake),
        1: total / (2 * n_sleep),
    }

    out_dir = base_dir or BASE_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    run_token = f"{year:04d}{month:02d}"
    fine_tuned_path = out_dir / f"exp_conv_bilstm_base_finetuned_{run_token}.keras"
    history_path = out_dir / f"exp_conv_bilstm_base_finetuned_{run_token}_history.json"

    model = keras.models.load_model(SLEEP_BASE_MODEL)
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(5e-4),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    eval_metrics = model.evaluate(X_val, y_val, verbose=0)
    preds = model.predict(sequences, batch_size=256).squeeze()
    model.save(fine_tuned_path)
    history_path.write_text(json.dumps(history.history, ensure_ascii=False, indent=2))

    quality_summary = summarize_sleep_quality(sleep_matrix, preds, window_size=5, stride=1)

    metric_names = dict(zip(model.metrics_names, eval_metrics))
    print(f"[SLEEP] 검증 성능: {metric_names}")
    print(f"[SLEEP] 수면 품질 요약: {quality_summary}")
    print(f"[SLEEP] 모델 및 히스토리 저장 완료 -> {fine_tuned_path}")

    return {
        "model_path": str(fine_tuned_path),
        "history_path": str(history_path),
        "metrics": metric_names,
        "sleep_quality": quality_summary,
    }


def run_stress_retrain(
    stress_matrix: np.ndarray,
    *,
    year: int,
    month: int,
    base_dir: Optional[Path] = None,
    seed: int = 2024,
) -> Dict[str, object]:
    run_token = f"{year:04d}{month:02d}"
    out_dir = base_dir or BASE_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        sequence_length=60,
        sequence_stride=5,
        window_label_strategy="max",
        val_split=0.2,
        batch_size=128,
        epochs=15,
        learning_rate=1e-3,
        weight_decay=1e-4,
        max_grad_norm=1.5,
        patience=5,
        num_workers=0,
        device="auto",
        seed=seed,
        freeze_cnn=False,
        freeze_gru=False,
        hr_threshold=95.0,
        sdnn_threshold=30.0,
        rmssd_threshold=25.0,
        acc_mean_threshold=0.85,
        acc_std_threshold=0.18,
        heuristic_votes=2,
        base_model_path=STRESS_BASE_MODEL,
        output_path=out_dir / f"cnn_gru_phase2_finetuned_{run_token}.pt",
        history_path=out_dir / f"cnn_gru_phase2_finetuned_{run_token}_history.json",
    )

    stress_retrain.set_seed(args.seed)
    device = stress_retrain.select_device(args.device)

    features = stress_matrix.astype(np.float32)
    labels = stress_retrain.derive_labels_from_heuristics(features, args)
    normalizer = stress_retrain.FeatureNormalizer.fit(features)

    model = stress_retrain.CNNGRUStressClassifier(input_dim=features.shape[1])
    checkpoint_normalizer = stress_retrain.maybe_load_checkpoint(
        model,
        args.base_model_path,
        device,
    )
    if checkpoint_normalizer:
        normalizer = checkpoint_normalizer

    normalized = normalizer.transform(features)
    sequences, window_labels = stress_retrain.build_windows(
        normalized,
        labels,
        args.sequence_length,
        args.sequence_stride,
        args.window_label_strategy,
    )
    train_dataset, val_dataset = stress_retrain.split_datasets(
        sequences,
        window_labels,
        args.val_split,
        args.seed,
    )
    train_loader, val_loader = stress_retrain.create_dataloaders(
        train_dataset,
        val_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )

    model.to(device)
    if args.freeze_cnn:
        stress_retrain.freeze_module(model.feature_extractor)
    if args.freeze_gru:
        stress_retrain.freeze_module(model.temporal_encoder)

    history = stress_retrain.train_model(
        model,
        train_loader,
        val_loader,
        device,
        args,
    )
    stress_retrain.save_checkpoint(model, normalizer, history, args)
    print("[STRESS] 재학습 완료")

    return {
        "model_path": str(args.output_path),
        "history_path": str(args.history_path),
        "history": history,
    }


def run_all_retrainings(
    *,
    sleep_matrix: np.ndarray,
    stress_matrix: np.ndarray,
    year: int,
    month: int,
    base_dir: Optional[Path] = None,
    seed: int = 2024,
) -> Dict[str, object]:
    """
    수면/스트레스 파이프라인을 순차적으로 실행하고 결과 요약을 반환한다.
    """
    out_dir = base_dir or BASE_MODEL_DIR
    sleep_result = run_sleep_retrain(
        sleep_matrix,
        year=year,
        month=month,
        base_dir=out_dir,
        seed=seed,
    )
    stress_result = run_stress_retrain(
        stress_matrix,
        year=year,
        month=month,
        base_dir=out_dir,
        seed=seed,
    )
    return {"sleep": sleep_result, "stress": stress_result}
