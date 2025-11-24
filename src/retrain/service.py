import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from src.retrain.models import SleepRetrainLog, StressRetrainLog

def export_sleep_retrain_npy(
    db: Session,
    user_email: str,
    year: int,
    month: int,
    out_path: Union[str, Path],
):

    out_path = Path("/app/model/")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(
        year + (1 if month == 12 else 0),
        1 if month == 12 else month + 1,
        1,
        tzinfo=timezone.utc,
    )

    logs = (
        db.query(SleepRetrainLog)
        .filter(SleepRetrainLog.user == user_email)
        .filter(SleepRetrainLog.measured_at >= start)
        .filter(SleepRetrainLog.measured_at < end)
        .order_by(SleepRetrainLog.measured_at.asc())
        .all()
    )

    if not logs:
        raise ValueError("No logs found")

    df = pd.DataFrame([
        {
            "TIMESTAMP": log.timestamp,
            "ACC_X": log.acc_x,
            "ACC_Y": log.acc_y,
            "ACC_Z": log.acc_z,
            "HR": log.hr,
            "Sleep_Stage": log.sleep_stage,
            "SAO2": log.sao2,
            "BVP": log.bvp,
        }
        for log in logs
    ])

    cols = [
        "TIMESTAMP",
        "ACC_X",
        "ACC_Y",
        "ACC_Z",
        "HR",
        "Sleep_Stage",
        "SAO2",
        "BVP",
    ]

    X = df[cols].astype("float32").values
    np.save(out_path, X)

def export_stress_retrain_npy(
    db: Session,
    user_email: str,
    year: int,
    month: int,
    out_path: Union[str, Path],
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = datetime(year, month, 1, tzinfo=timezone.utc)
    end = datetime(
        year + (1 if month == 12 else 0),
        1 if month == 12 else month + 1,
        1,
        tzinfo=timezone.utc,
    )

    logs = (
        db.query(StressRetrainLog)
        .filter(StressRetrainLog.user == user_email)
        .filter(StressRetrainLog.created_at >= start)
        .filter(StressRetrainLog.created_at < end)
        .order_by(StressRetrainLog.created_at.asc())
        .all()
    )

    if not logs:
        raise ValueError("No stress retrain logs found")

    df = pd.DataFrame(
        [
            {
                "heart_rate_bpm": log.heart_rate_bpm,
                "hrv_sdnn_ms": log.hrv_sdnn_ms,
                "hrv_rmssd_ms": log.hrv_rmssd_ms,
                "acc_x_mean": log.acc_x_mean,
                "acc_y_mean": log.acc_y_mean,
                "acc_z_mean": log.acc_z_mean,
                "acc_mag_mean": log.acc_mag_mean,
                "acc_mag_std": log.acc_mag_std,
                "wrist_temperature_c_mean": log.wrist_temperature_c_mean,
            }
            for log in logs
        ]
    )

    cols = [
        "heart_rate_bpm",
        "hrv_sdnn_ms",
        "hrv_rmssd_ms",
        "acc_x_mean",
        "acc_y_mean",
        "acc_z_mean",
        "acc_mag_mean",
        "acc_mag_std",
        "wrist_temperature_c_mean",
    ]

    X = df[cols].astype("float32").values
    np.save(out_path, X)
