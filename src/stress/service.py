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


def summarize_stress_today_by_hour(db: Session, user_email: str):
    hour = func.date_trunc("hour", StressLog.timestamp)

    rows = (
        db.query(
            hour.label("hour"),
            func.count(StressLog.id).label("count"),
        )
        .filter(
            StressLog.user == user_email,
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

    return {
        "stress_hourly_log": stress_log
    }
