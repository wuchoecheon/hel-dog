from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.device.utils import map_device_to_user

from src.sleep.models import SleepLog
from src.sleep.schemas import SleepLogSchema
from src.sleep.service import summarize_sleep_last_7h
from src.sleep.schemas import SleepLogSchema, CreateSleepLogResponse, GetSleepScoreResponse
from src.sleep.utils import calc_sleep_score

from datetime import datetime, timedelta

router = APIRouter(
    prefix="/api/sleep"
)

@router.post("/{device_id}", response_model=CreateSleepLogResponse)
def create_sleep_log(
    device_id: str,
    body: SleepLogSchema,
    user: Annotated[User, Depends(map_device_to_user)],
    db: Annotated[Session, Depends(get_db)],
):
    db_sleep_log = SleepLog(user=user, label=body.label, timestamp=datetime.now())
    db.add(db_sleep_log)
    db.commit()
    db.refresh(db_sleep_log)

    return {"response": "signup success!"}


@router.get("", response_model=GetSleepScoreResponse)
def get_sleep_score(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):

    summary = summarize_sleep_last_7h(db, user.email)

    return {
        "response": "request proceed successfully",
        **summary
    }
