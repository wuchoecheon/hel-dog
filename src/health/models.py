from sqlalchemy import Column, String, ForeignKey, Integer, Float, DateTime

from src.database import Base
from src.auth.models import User

class HealthData(Base):
    __tablename__ = "health_data"
    
    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    timestamp = Column(DateTime, nullable=False)
    heart_rate_data = Column(Integer)
    oxygen_saturation = Column(Float)
    stress_level = Column(String)