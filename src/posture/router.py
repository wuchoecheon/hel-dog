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
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=3)

    rows = (
        db.query(
            PostureLog.label,
            func.count(PostureLog.id).label("count"),
        )
        .filter(
            PostureLog.user == user.email,
            PostureLog.timestamp >= start,
            PostureLog.timestamp < now,
        )
        .group_by(PostureLog.label)
        .all()
    )

    counts_by_label = {1: 0, 2: 0, 3: 0}
    for label, count in rows:
        if label in counts_by_label:
            counts_by_label[label] = count

    total = sum(counts_by_label.values())

    return {
        "response": "request proceeded successfully",
        "total": total,
        "counts_by_label": counts_by_label,
    }

@router.get("/detail")
def get_posture_weekly(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    # 최근 7일 하루 단위로 총 나쁜 자세 횟수, 라벨별 횟수
    now = datetime.now(timezone.utc)
    today = now.date()
    start_date = today - timedelta(days=6)

    day_col = func.date(PostureLog.timestamp)

    rows = (
        db.query(
            day_col.label("day"),
            PostureLog.label,
            func.count(PostureLog.id).label("count"),
        )
        .filter(
            PostureLog.user == user.email,
            day_col >= start_date,
            day_col <= today,
        )
        .group_by("day", PostureLog.label)
        .order_by("day")
        .all()
    )

    day_stats = {}
    for i in range(7):
        d = start_date + timedelta(days=i)
        day_stats[d.isoformat()] = {
            "date": d.isoformat(),
            "total": 0,
            "by_label": {1: 0, 2: 0, 3: 0},
        }

    for row in rows:
        day_str = row.day.isoformat()
        label = row.label
        count = row.count

        day_stats[day_str]["total"] += count
        if label in day_stats[day_str]["by_label"]:
            day_stats[day_str]["by_label"][label] += count

    days = [day_stats[(start_date + timedelta(days=i)).isoformat()] for i in range(7)]

    return {
        "response": "request proceeded successfully",
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "days": days,
    }
