from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user

from src.retrain.models import SleepRetrainLog, StressRetrainLog
from src.retrain.schemas import SleepRetrainLogSchema, StressRetrainLogSchema

from datetime import datetime, timedelta, timezone

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
        sao2=payload.sao2,
        bvp=payload.bvp,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    return {"response": "sleep retrain log saved"}

@router.get("/sleep")
def get_sleep_retrain_logs(
    month: int,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    #
