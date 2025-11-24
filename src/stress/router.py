from fastapi import APIRouter, Depends
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.device.utils import map_device_to_user

from src.stress.models import StressLog

from src.stress.service import (
    summarize_stress_last_12h,
    summarize_stress_today_by_hour,
)

router = APIRouter(
    prefix="/api/stress"
)


@router.post("/{device_id}")
def create_stress_log(
    device_id: str,
    user_email: Annotated[str, Depends(map_device_to_user)],
    db: Session = Depends(get_db),
):

    db_stress_log = StressLog(user=user_email)

    db.add(db_stress_log)
    db.commit()
    db.refresh(db_stress_log)

    return {"response": "stress log saved", "id": db_stress_log.id}


@router.get("")
def get_stress_status(
    user: Annotated[User, Depends(get_user)],
    db: Session = Depends(get_db),
):
    # 12시간 내에 스트레스 로그 기반으로 스트레스 o/x 상태 판단
    summary = summarize_stress_last_12h(db, user.email)

    return {
        "response": "request proceeded successfully",
        **summary,
    }


@router.get("/detail")
def get_stress_log_detail(
    user: Annotated[User, Depends(get_user)],
    db: Session = Depends(get_db),
):

    # 시간대별 스트레스 횟수
    summary = summarize_stress_today_by_hour(db, user.email)

    return {
        "response": "request proceeded successfully",
        **summary,
    }
