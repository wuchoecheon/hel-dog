from typing import Annotated

from sqlalchemy.orm import Session
from src.robot.models import Robot
from src.database import get_db 

from fastapi import Depends, HTTPException, status

def map_robot_to_user(robot_id: str, db: Annotated[Session, Depends(get_db)]):
    robot = db.query(Robot).filter(Robot.robot_id == robot_id).first()
    return robot.user_email
