from fastapi import APIRouter, Depends, HTTPException, status, Response
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.models import Robot
from src.fall.models import Fall
from src.fall.schemas import FallLogSchema, FallCheckSchema

router = APIRouter(
    prefix="/api/fall"
)

@router.post("/log")
def record_fall_log(
    body: FallLogSchema,
    db: Session=Depends(get_db),
):
    if not db.get(Robot, body.robot_id):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail="Invalid robot id",
        )

    #if body.label == "normal":
    #   return {"response": "received normal log"}

    db_fall = Fall(
        robot_id=body.robot_id,
        label=body.label,
    )
    if body.occurred_at is not None:
        db_fall.occurred_at = body.occurred_at

    db.add(db_fall)
    db.commit()
    db.refresh(db_fall)

    return {"response": "fall log saved"}


@router.get("", response_model=list[FallCheckSchema])
def get_fall_logs(
    user:Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    robot_ids = db.scalars(
        select(Robot.robot_id).where(Robot.user_email == user.email)
    ).all()

    if not robot_ids:
        return []

    falls = db.scalars(
        select(Fall)
        .where(
            Fall.robot_id.in_(robot_ids),
            Fall.label == "fall",
            func.date(Fall.occurred_at) == func.current_date(),
        )
        .order_by(Fall.occurred_at.desc())
    ).all()

    return falls
