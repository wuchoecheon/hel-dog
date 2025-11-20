from sqlalchemy import Column, String, ForeignKey, DateTime, Integer

from src.database import Base

class SleepLog(Base):
    __tablename__ = "sleep_log"

    id = Column(Integer, primary_key=True) 
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    label = Column(Integer, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)