from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.utils import map_robot_to_user
from src.cough.models import CoughLog

from src.cough.schemas import CoughLogSchema

from datetime import datetime

router = APIRouter(
    prefix="/api/cough"
)

@router.post("")
def create_cough_log(
        body: CoughLogSchema,
        db: Session=Depends(get_db)
    ):
    user = map_robot_to_user(body.robot_id, db)
    db_cough_log = CoughLog(user=user, timestamp=datetime.now())

    db.add(db_cough_log)
    db.commit()
    db.refresh(db_cough_log)

    return {"response": "signup success!"}


@router.get("")
def get_cough_log(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):
    ...

@router.post("/detail")
def get_cough_log_detail(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):
    ...