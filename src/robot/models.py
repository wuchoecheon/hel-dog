from sqlalchemy import Column, String, ForeignKey

from src.database import Base

class Robot(Base):
    __tablename__ = "robot"

    robot_id = Column(String, primary_key=True, unique=True, index=True)
    user_email = Column(String, ForeignKey("user.email", ondelete="CASCADE"))
