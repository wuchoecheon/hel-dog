from sqlalchemy import Column, String, ForeignKey, DateTime, Integer
from sqlalchemy.sql import func

from src.database import Base

class PostureLog(Base):
    __tablename__ = "posture_log"

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    label = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
