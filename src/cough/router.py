from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.utils import map_robot_to_user
from src.cough.models import CoughLog
from src.cough.schemas import CreateCoughLogResponse, GetCoughLogResponse, GetCoughLogDetailResponse # New import

from datetime import datetime, timedelta

router = APIRouter(
    prefix="/api/cough"
)

@router.post("/{robot_id}", response_model=CreateCoughLogResponse) # Added response_model
def create_cough_log(
    robot_id: str,
    user: Annotated[User, Depends(map_robot_to_user)],
    db: Annotated[Session, Depends(get_db)],
):
    db_cough_log = CoughLog(user=user, timestamp=datetime.now())
    db.add(db_cough_log)
    db.commit()
    db.refresh(db_cough_log)

    return {"response": "signup success!"}


@router.get("", response_model=GetCoughLogResponse) # Added response_model
def get_cough_log(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):
    three_hours_ago = datetime.now() - timedelta(hours=3)

    coughs = db.query(CoughLog).filter(
        CoughLog.user == user.email,
        CoughLog.timestamp >= three_hours_ago
    ).all()

    return {
        "response": "request proceed successfully",
        "cough_num": len(coughs),
        "cough_log": coughs
    }



@router.post("/detail", response_model=GetCoughLogDetailResponse) # Added response_model
def get_cough_log_detail(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):
    now = datetime.now()
    one_week_ago = now - timedelta(days=7)

    coughs = (
        db.query(CoughLog)
        .filter(
            CoughLog.user == user,
            CoughLog.timestamp >= one_week_ago,
        )
        .all()
    )

    daily_counts = [0] * 7

    for c in coughs:
        diff = now - c.timestamp
        days_ago = int(diff.total_seconds() // 86400) + 1

        if 1 <= days_ago <= 7:
            daily_counts[days_ago - 1] += 1

    return {
        "response": "request proceed successfully",
        "daily_counts": daily_counts,
        "total_coughs": len(coughs),
    }
