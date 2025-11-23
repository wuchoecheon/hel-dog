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
from src.fall.schemas import CreateFallLogResponse, GetFallLogResponse # New import

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

    db_fall_log = FallLog(
        user=user_email,
    )

    db.add(db_fall_log)
    db.commit()
    db.refresh(db_fall_log)

    return {"response": "fall log saved"}


@router.get("", response_model=GetFallLogResponse) # Added response_model
def get_fall_log(
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    falls = (
        db.query(FallLog)
        .filter(
            FallLog.user == user.email,
            func.date(FallLog.timestamp) == func.current_date(),
        )
        .order_by(FallLog.timestamp.desc())
        .all()
    )

    fall_log = [
        {
            "timestamp": f.timestamp.isoformat() if f.timestamp else None,
        }
        for f in falls
    ]

    data = {
        "response": "request proceed successfully",
        "fall_num": len(falls),
        "fall_log": fall_log
    }

    return data