from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.sql import func

from src.database import Base

class StressLog(Base):
    __tablename__ = "stress_log"

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
