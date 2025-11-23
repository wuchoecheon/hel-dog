from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.models import Robot
from src.robot.utils import map_robot_to_user
from src.fall.models import FallLog
from src.fall.service import summarize_today_falls

router = APIRouter(
    prefix="/api/fall"
)

@router.post("/{robot_id}")
def create_fall_log(
    robot_id: str,
    user_email: Annotated[str, Depends(map_robot_to_user)],
    db: Session=Depends(get_db),
):

    db_fall_log = FallLog(
        user=user_email,
    )

    db.add(db_fall_log)
    db.commit()
    db.refresh(db_fall_log)

    return {"response": "fall log saved"}


@router.get("")
def get_fall_log(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    summary = summarize_today_falls(db=db, user_email=user.email)

    return {
        "response": "request proceeded successfully",
        **summary
    }
