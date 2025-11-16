from fastapi import APIRouter, Depends, HTTPException, status, Response
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.robot.models import Robot
from src.robot.schemas import RegisterSchema

router = APIRouter(
    prefix="/api/robot"
)

@router.post("/register")
def register_robot(
    body: RegisterSchema,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    # 중복 등록 방지 (PK: robot_id)
    if db.get(Robot, body.robot_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Robot already registered",
        )

    db_robot = Robot(
        robot_id=body.robot_id,
        user_email=user.email,
    )

    db.add(db_robot)
    db.commit()
    db.refresh(db_robot)

    return {"response": "register success!"}

@router.delete("/{robot_id}")
def delete_robot(
    robot_id: str,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):

    db_robot = db.get(Robot, robot_id)
    if not db_robot:
        raise HTTPException(
            status_code=status.HTTP_400_NOT_FOUND,
            detail="Robot not found",
        )

    if db_robot.user_email != user.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not owner of this robot",
        )

    db.delete(db_robot)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
