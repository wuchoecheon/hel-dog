from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.utils import map_robot_to_user

from src.posture.models import PostureLog
from src.posture.schemas import PostureLogSchema
from src.posture.service import summarize_posture_last_3h, summarize_posture_last_week

from datetime import datetime, timedelta, timezone

router = APIRouter(
    prefix="/api/posture"
)

@router.post("/{robot_id}")
def create_posture_log(
    robot_id: str,
    body: PostureLogSchema,
    user_email: Annotated[str, Depends(map_robot_to_user)],
    db: Session=Depends(get_db),
):
    db_posture_log = PostureLog(
        user=user_email,
        label=body.label,
    )

    db.add(db_posture_log)
    db.commit()
    db.refresh(db_posture_log)

    return {"response": "posture log saved"}

@router.get("")
def get_posture_last_3h(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    # 최근 3시간 동안 나쁜자세 라벨별 횟수
    summary = summarize_posture_last_3h(db, user.email)

    return {
        "response": "request proceeded successfully",
        **summary
    }

@router.get("/detail")
def get_posture_weekly(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    # 최근 7일 하루 단위로 총 나쁜 자세 횟수, 라벨별 횟수
    summary = summarize_posture_last_week(db, user.email)

    return {
        "response": "request proceeded successfully",
        **summary
    }
