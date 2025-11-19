from sqlalchemy import Column, String, ForeignKey, Time, Integer

from src.database import Base

class CoughLog(Base):
    __tablename__ = "cough_log"

    id = Column(Integer, primary_key=True) 
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    timestamp = Column(Time)
