from fastapi import APIRouter, Depends, HTTPException, status
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.utils import map_robot_to_user
from src.cough.models import CoughLog
from src.cough.service import summarize_cough_last_3h, summarize_cough_last_week
from src.cough.schemas import CreateCoughLogResponse, GetCoughLogResponse, GetCoughLogDetailResponse # New import

from datetime import datetime, timedelta

router = APIRouter(
    prefix="/api/cough"
)

@router.post("/{robot_id}", response_model=CreateCoughLogResponse) # Added response_model
def create_cough_log(
    robot_id: str,
    user: Annotated[User, Depends(map_robot_to_user)],
    db: Annotated[Session, Depends(get_db)],
):
    db_cough_log = CoughLog(user=user, timestamp=datetime.now())
    db.add(db_cough_log)
    db.commit()
    db.refresh(db_cough_log)

    return {"response": "signup success!"}


@router.get("", response_model=GetCoughLogResponse) # Added response_model
def get_cough_log(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):

    summary = summarize_cough_last_3h(
        db=db,
        user_email=user.email,
    )

    return {
        "response": "request proceeded successfully",
        **summary,
    }



@router.get("/detail", response_model=GetCoughLogDetailResponse) # Added response_model
def get_cough_log_detail(
        user: User=Depends(get_user),
        db: Session=Depends(get_db)
    ):
    summary = summarize_cough_last_week(
        db=db,
        user_email=user.email,
    )

    return {
        "response": "request proceeded successfully",
        **summary,
    }
