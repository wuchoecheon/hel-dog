from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from src.sleep.models import SleepLog
from src.sleep.utils import calc_sleep_score


def summarize_sleep_last_7h(db: Session, user_email: str):
    now = datetime.now()
    start = now - timedelta(hours=7)

    sleeps = (
        db.query(SleepLog)
        .filter(
            SleepLog.user == user_email,
            SleepLog.timestamp >= start,
        )
        .all()
    )

    score = calc_sleep_score(sleeps)

    return {
        "sleep_score": score
    }
