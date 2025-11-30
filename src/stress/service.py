from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.stress.models import StressLog


STRESS_THRESHOLD_12H = 3 # 12시간 내 스트레스 판단 기준치

def summarize_stress_last_12h(db: Session, user_email: str):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=12)

    count_last_12h = (
        db.query(func.count(StressLog.id))
        .filter(
            StressLog.user == user_email,
            StressLog.timestamp >= start,
            StressLog.timestamp < now,
        )
        .scalar()
    )

    stressed = count_last_12h >= STRESS_THRESHOLD_12H

    return {
        "count_last_12h": count_last_12h,
        "stressed": stressed,
    }


def summarize_stress_last_week(db: Session, user_email: str):
    now = datetime.now(timezone.utc)
    today = now.date()
    start_date = today - timedelta(days=6)

    day_col = func.date(StressLog.timestamp)

    rows = (
        db.query(
            day_col.label("day"),
            func.count(StressLog.id).label("count"),
        )
        .filter(
            StressLog.user == user_email,
            day_col >= start_date,
            day_col <= today,
        )
        .group_by("day")
        .order_by("day")
        .all()
    )

    day_stats = {}
    for i in range(7):
        d = start_date + timedelta(days=i)
        day_stats[d.isoformat()] = {
            "date": d.isoformat(),
            "total": 0,
        }

    for row in rows:
        day_str = row.day.isoformat()
        count = row.count
        day_stats[day_str]["total"] += count

    days = [
        day_stats[(start_date + timedelta(days=i)).isoformat()]
        for i in range(7)
    ]

    return {
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "days": days,
    }
