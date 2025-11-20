from sqlalchemy import Column, String, ForeignKey, DateTime, Integer

from src.database import Base

class CoughLog(Base):
    __tablename__ = "cough_log"

    id = Column(Integer, primary_key=True) 
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)