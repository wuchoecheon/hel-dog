# src/cough/service.py
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from src.cough.models import CoughLog

def summarize_cough_last_3h(
    db: Session,
    user_email: str,
) -> dict:
    three_hours_ago = datetime.utcnow() - timedelta(hours=3)

    coughs = (
        db.query(CoughLog)
        .filter(
            CoughLog.user == user_email,
            CoughLog.timestamp >= three_hours_ago,
        )
        .all()
    )

    return {
        "cough_num": len(coughs),
        "cough_log": coughs,
    }


def summarize_cough_last_week(
    db: Session,
    user_email: str,
) -> dict:
    now = datetime.utcnow()
    one_week_ago = now - timedelta(days=7)

    coughs = (
        db.query(CoughLog)
        .filter(
            CoughLog.user == user_email,
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
        "daily_counts": daily_counts,
        "total_coughs": len(coughs),
    }
