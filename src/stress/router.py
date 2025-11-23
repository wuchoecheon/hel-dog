# 스트레스 12시간 내 평균 횟수
# 시간대별 스트레스 횟수 전달
# 스트레스: 1시간 내 라벨 stress 몇번 이상이면 스트레스 상태라고 판단
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.device.utils import map_device_to_user

from src.stress.models import StressLog

from datetime import datetime, timedelta, timezone

router = APIRouter(
    prefix="/api/stress"
)

STRESS_THRESHOLD_12H = 3 # 12시간 내 기준치

@router.post("/{device_id}")
def create_stress_log(
    device_id: str,
    user_email: Annotated[str, Depends(map_device_to_user)],
    db: Session=Depends(get_db),
):
    db_stress_log = StressLog(
        user=user_email,
    )

    db.add(db_stress_log)
    db.commit()
    db.refresh(db_stress_log)

    return {"response": "stress log saved"}

@router.get("")
def get_stress_status(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    # 12시간 내에 스트레스 로그 기반으로 스트레스 o/x 상태 판단
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=12)

    count_last_12h = (
        db.query(func.count(StressLog.id))
        .filter(
            StressLog.user == user.email,
            StressLog.timestamp >= start,
            StressLog.timestamp < now,
        )
        .scalar()
    )

    stressed = count_last_12h >= STRESS_THRESHOLD_12H

    return {
        "response": "request proceeded successfully",
        "stressed": stressed,
        "count_last_12h": count_last_12h,
    }



@router.get("/detail")
def get_stress_log_detail(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    # 시간대별 스트레스 횟수
    hour = func.date_trunc("hour", StressLog.timestamp)

    rows = (
        db.query(
            hour.label("hour"),
            func.count(StressLog.id).label("count"),
        )
        .filter(
            StressLog.user == user.email,
            func.date(StressLog.timestamp) == func.current_date(),
        )
        .group_by("hour")
        .order_by("hour")
        .all()
    )

    stress_log = [
        {
            "hour": row.hour.isoformat(),
            "count": row.count,
        }
        for row in rows
    ]

    data = {
        "response": "request proceeded successfully",
        "stress_hourly_log": stress_log
    }

    return data
