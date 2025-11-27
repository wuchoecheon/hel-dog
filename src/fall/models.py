from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean
from sqlalchemy.sql import func

from src.database import Base

class FallLog(Base):
    __tablename__ = "fall_log"

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    responsed = Column(Boolean, nullable=False, server_default="False")
    notified = Column(Boolean, nullable=False, server_default="False")
    notified_to_caregiver = Column(Boolean, nullable=False, server_default="False")