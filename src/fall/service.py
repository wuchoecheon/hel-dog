from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.fall.models import FallLog


def summarize_today_falls(db: Session, user_email: str):
    falls = (
        db.query(FallLog)
        .filter(
            FallLog.user == user_email,
            func.date(FallLog.timestamp) == func.current_date(),
        )
        .order_by(FallLog.timestamp.desc())
        .all()
    )

    fall_log = [
        {
            "timestamp": f.timestamp.isoformat() if f.timestamp else None,
        }
        for f in falls
    ]

    return {
        "fall_num": len(falls),
        "fall_log": fall_log
    }


def summarize_fall_last_24h(db: Session, user_email: str):
    now = datetime.utcnow()
    since = now - timedelta(hours=24)

    falls = (
        db.query(FallLog)
        .filter(
            FallLog.user == user_email,
            FallLog.timestamp >= since,
        )
        .order_by(FallLog.timestamp.desc())
        .all()
    )

    last_fall_at = falls[0].timestamp if falls else None

    return {
        "fall_count_24h": len(falls),
        "last_fall_at": last_fall_at
    }
