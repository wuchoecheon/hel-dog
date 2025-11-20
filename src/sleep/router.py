from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.device.utils import map_device_to_user

from src.sleep.models import SleepLog
from src.sleep.schemas import SleepLogSchema
from src.sleep.utils import calc_sleep_score

from datetime import datetime, timedelta

router = APIRouter(
    prefix="/api/sleep"
)

@router.post("/{device_id}")
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


@router.get("")
def get_sleep_score(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):

    sleeps = db.query(SleepLog).filter(
        SleepLog.user == user.email,
        SleepLog.timestamp >= (datetime.now() - timedelta(hours=7))
    ).all()

    return {
        "response": "request proceed successfully",
        "sleep_score": calc_sleep_score(sleeps)
    }