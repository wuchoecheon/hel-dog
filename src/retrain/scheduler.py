from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from src.retrain.task import run_monthly_export_npy

scheduler = BackgroundScheduler()

def schedule_monthly_job():
    now = datetime.utcnow()
    year = now.year
    month = now.month - 1
    if month == 0:
        month = 12
        year -= 1

    run_monthly_export_npy(user_email="test@test.com", year=year, month=month)

def start_scheduler():
    scheduler.add_job(
        schedule_monthly_job,
        "cron",
        day=1,
        hour=0,
        minute=10,
        id="monthly_export",
        replace_existing=True,
    )
    scheduler.start()
