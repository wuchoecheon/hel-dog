import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user

from src.retrain.models import SleepRetrainLog

def export_sleep_retrain_npy(
    db: Session,
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
    year: int,
    month: int,
    out_path: Union[str, Path],
): pass