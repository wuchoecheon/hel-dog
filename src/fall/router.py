from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.utils import map_robot_to_user
from src.fall.models import FallLog
from src.fall.service import summarize_today_falls
from src.fall.schemas import CreateFallLogResponse, GetFallLogResponse, FallResponseEntry
from src.fcm.models import FCM

from src.fcm.utils import send_notification

from datetime import datetime, timedelta

router = APIRouter(
    prefix="/api/fall"
)

@router.post("/{robot_id}", response_model=CreateFallLogResponse) # Added response_model
def create_fall_log(
    robot_id: str,
    user_email: Annotated[str, Depends(map_robot_to_user)],
    db: Session=Depends(get_db),
):
    db_fall_log = FallLog(user=user_email)

    fcm_token = db.query(FCM).filter(FCM.owner == user_email).first()
    if fcm_token:
        send_notification(
            fcm_token=fcm_token,
            title="Fall detected",
            body="Click to response",
            data_payload={
                "id": db_fall_log.id
            }
        )
        db_fall_log.notified = True

    db.add(db_fall_log)
    db.commit()
    db.refresh(db_fall_log)

    return {
        "response": "fall log saved",
        "notification_sent": db_fall_log.notified
    }


@router.get("", response_model=GetFallLogResponse) # Added response_model
def get_fall_log(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    summary = summarize_today_falls(db=db, user_email=user.email)

    return {
        "response": "request proceeded successfully",
        **summary
    }


@router.post("response")
def response_to_fall(
    body: FallResponseEntry,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    fall_log: FallLog | None = db.query(FallLog).get(body.id)
    
    if not fall_log:
        return HTTPException(400, "There is no fall log there")
    
    if fall_log.user != user.email:
        return HTTPException(400, "It is not your log")
    
    if fall_log.notified != user.email:
        return HTTPException(400, "It did not send... how did you respond?????")

    fall_log.responsed = True

    db.add(fall_log)
    db.commit()
    db.refresh(fall_log)

    return {
        "response": "response received successfully"
    }
