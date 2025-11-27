# app.py
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta

from fastapi import FastAPI
from src.database import SessionLocal
from src.fall.models import FallLog
from src.fcm.models import FCM
from src.fcm.utils import send_notification
from typing import List, Dict

async def alarm_scheduler():
    while True:
        db = SessionLocal()
        try:
            successed = 0
            threshold = datetime.now() - timedelta(minutes=5)

            pending: List[FallLog] = db.query(FallLog).filter(
                    FallLog.notified == True,
                    FallLog.responsed != False,
                    FallLog.notified_to_caregiver == False,
                    FallLog.timestamp < threshold
                ).all()

            # TO-DO
            # Now it is send back to fell people again.
            # Add caregiver and send to them
            log_dict: Dict[str, List[FallLog]] = {}
            fcm_dict: Dict[str, str] = {}
            for log in pending:
                fcm_token = db.query(FCM).filter(FCM.owner == log.user).first()
                log_dict(log.user).append(FallLog)
                fcm_dict(log.user).append(fcm_token.fcm_token)

            for user, token in fcm_dict.items():
                send_notification(
                    fcm_token=token,
                    title="보호자 알림",
                    body="피보호자가 낙상 후 미응답하였습니다.",
                )

                if send_notification:
                    for log in log_dict[user]:
                        log.notified_to_caregiver = True
                        db.commit()
                    successed += 1

            del log_dict, fcm_dict

            print(successed)
        finally:
            db.close()

        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(alarm_scheduler())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task