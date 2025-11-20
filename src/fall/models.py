from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.sql import func

from src.database import Base

class Fall(Base):
    __tablename__ = "fall"

    id = Column(Integer, primary_key=True, autoincrement=True)
    robot_id = Column(String, ForeignKey("robot.robot_id", ondelete="CASCADE"))
    label = Column(String) # fall/normal
    occurred_at = Column(DateTime(timezone=True), server_default=func.now())
