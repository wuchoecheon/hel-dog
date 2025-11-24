from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.posture.models import PostureLog


def summarize_posture_last_3h(db: Session, user_email: str):
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=3)

    rows = (
        db.query(
            PostureLog.label,
            func.count(PostureLog.id).label("count"),
        )
        .filter(
            PostureLog.user == user_email,
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
        "total": total,
        "counts_by_label": counts_by_label
    }


def summarize_posture_last_week(db: Session, user_email: str):
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
            PostureLog.user == user_email,
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

    days = [
        day_stats[(start_date + timedelta(days=i)).isoformat()]
        for i in range(7)
    ]

    return {
        "start_date": start_date.isoformat(),
        "end_date": today.isoformat(),
        "days": days
    }
