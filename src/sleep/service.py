from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

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

def summarize_sleep_last_week(db: Session, user_email: str):
    now = datetime.now(timezone.utc)
    today = now.date()
    start_date = today - timedelta(days=6)

    day_col = func.date(SleepLog.timestamp)

    rows = (
        db.query(SleepLog)
        .filter(
            SleepLog.user == user_email,
            day_col >= start_date,
            day_col <= today,
        )
        .all()
    )

    logs_by_day = {}
    for i in range(7):
        d = start_date + timedelta(days=i)
        logs_by_day[d] = []

    for log in rows:
        d = log.timestamp.date()
        if d in logs_by_day:
            logs_by_day[d].append(log)

    day_stats = []
    for i in range(7):
        d = start_date + timedelta(days=i)
        logs = logs_by_day[d]

        score = calc_sleep_score(logs)

        day_stats.append({
            "date": d.isoformat(),
            "sleep_score": score,
        })

    return {
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "days": day_stats,
    }
