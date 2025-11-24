from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user

from src.retrain.models import SleepRetrainLog, StressRetrainLog
from src.retrain.schemas import SleepRetrainLogSchema, StressRetrainLogSchema

from datetime import timedelta

router = APIRouter(
    prefix="/api/retrain"
)

@router.post("/sleep")
def create_sleep_retrain_log(
    payload: SleepRetrainLogSchema,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    measured_at = payload.sleep_start_time + timedelta(milliseconds=payload.timestamp)

    obj = SleepRetrainLog(
        user=user.email,
        sleep_start_time=payload.sleep_start_time,
        timestamp=payload.timestamp,
        measured_at=measured_at,
        acc_x=payload.acc_x,
        acc_y=payload.acc_y,
        acc_z=payload.acc_z,
        hr=payload.hr,
        sleep_stage=payload.sleep_stage,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    return {"response": "sleep retrain log saved"}

@router.post("/stress")
def create_stress_retrain_log(
    payload: StressRetrainLogSchema,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    obj = StressRetrainLog(
        user=user.email,
        heart_rate_bpm=payload.heart_rate_bpm,
        hrv_sdnn_ms=payload.hrv_sdnn_ms,
        hrv_rmssd_ms=payload.hrv_rmssd_ms,
        acc_x_mean=payload.acc_x_mean,
        acc_y_mean=payload.acc_y_mean,
        acc_z_mean=payload.acc_z_mean,
        acc_mag_mean=payload.acc_mag_mean,
        acc_mag_std=payload.acc_mag_std,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    return {"response": "stress retrain log saved"}
