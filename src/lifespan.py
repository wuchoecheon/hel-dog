# app.py
import asyncio
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict

from fastapi import FastAPI
from src.database import SessionLocal
from src.fall.models import FallLog
from src.caregiver.models import Caregiver
from src.fcm.models import FCM
from src.fcm.utils import send_notification


async def alarm_scheduler():
    while True:
        db = SessionLocal()
        try:
            successed = 0
            threshold = datetime.now() - timedelta(minutes=5)

            pending: List[FallLog] = (
                db.query(FallLog)
                .filter(
                    FallLog.notified.is_(True),
                    FallLog.responsed.is_(False),
                    FallLog.notified_to_caregiver.is_(False),
                    FallLog.timestamp < threshold,
                )
                .all()
            )

            log_dict: Dict[str, List[FallLog]] = defaultdict(list)
            fcm_dict: Dict[str, List[str]] = defaultdict(list)

            for log in pending:
                caregiver = (
                    db.query(Caregiver)
                    .filter(Caregiver.ward == log.user)
                    .first()
                )
                if not caregiver:
                    log.notified_to_caregiver = True
                    continue

                caregiver_fcm_token = (
                    db.query(FCM)
                    .filter(FCM.owner == caregiver.caregiver)
                    .first()
                )
                if not caregiver_fcm_token:
                    log.notified_to_caregiver = True
                    continue

                log_dict[log.user].append(log)
                fcm_dict[log.user].append(caregiver_fcm_token.fcm_token)

            for user, tokens in fcm_dict.items():
                for token in tokens:
                    send_notification(
                        fcm_token=token,
                        title="보호자 알림",
                        body=f"피보호자({user})가 낙상 후 미응답하였습니다.",
                    )

                for log in log_dict[user]:
                    log.notified_to_caregiver = True
                successed += 1

            db.commit()

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
