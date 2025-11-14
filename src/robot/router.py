from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from sqlalchemy import select

from src.database import get_db
from src.auth.models import User
from src.robot.models import Robot
from src.robot.schemas import RegisterSchema, DeleteSchema
from src.robot.utils import get_current_user_email  # 현재 로그인한 사용자

router = APIRouter(
    prefix="/api/robot"
)

@router.post("/register")
def register_robot(
    body: RegisterSchema,
    db: Session=Depends(get_db),
    current_email: str = Depends(get_current_user_email),
):
    # 본인 계정만 등록 가능
    if body.user_email != current_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You can only register your own robot",
        )

    # FK 유저 존재 확인
    user_exists = db.execute(
        select(User.email).where(User.email == body.user_email)
    ).first()
    if not user_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # 중복 등록 방지 (PK: robot_id)
    if db.get(Robot, body.robot_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Robot already registered",
        )

    db_robot = Robot(robot_id=body.robot_id, user_email=body.user_email)

    db.add(db_robot)
    db.commit()
    db.refresh(db_robot)

    return {"response": "register success!"}

@router.delete("/{robot_id}")
def delete_robot(
    robot_id: str,
    body: DeleteSchema,
    db: Session=Depends(get_db),
    current_email: str = Depends(get_current_user_email),
):
    # 본인 계정만 삭제 가능
    if body.user_email != current_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You can only delete your own robot",
        )

    db_robot = db.get(Robot, robot_id)
    if not db_robot:
        raise HTTPException(
            status_code=status.HTTP_400_NOT_FOUND,
            detail="Robot not found",
        )

    if db_robot.user_email != body.user_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not owner of this robot",
        )

    db.delete(db_robot)
    db.commit()

    #return (f"deleted robot {robot_id}")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
