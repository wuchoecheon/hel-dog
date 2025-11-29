from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.fcm.models import FCM
from src.fcm.schemas import FCMRegisterSchema
from src.fcm.utils import send_notification

router = APIRouter(
    prefix="/api/fcm"
)

@router.post("/register")
def fcm_register(body: FCMRegisterSchema, user: Annotated[User, Depends(get_user)], db: Session=Depends(get_db)):
    fcm_token = body.fcm_token
    if db.query.get(fcm_token):
        return HTTPException(400, "FCM token already exists")


    db_fcm = FCM(
        uuid=body.uuid,
        fcm_token=body.fcm_token,
        owner=user.email
    )
    
    db.add(db_fcm)
    db.commit()
    db.refresh(db_fcm)

    return {"response": "register success!"}


@router.post("/test")
def fcm_notification_test(user: Annotated[User, Depends(get_user)], db: Session=Depends(get_db)):
    fcm_token = db.query(FCM).filter(FCM.owner == user.email).first()
    if not fcm_token:
        return HTTPException(400, "fcm token does not exists")

    response = send_notification(
        fcm_token=fcm_token,
        title="FCM test",
        body="Was it successful?"
    )

    print(response)

    return {"response": "notification sent"}