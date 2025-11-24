from sqlalchemy import Column, String, ForeignKey, DateTime, Integer, Float
from sqlalchemy.sql import func

from src.database import Base

class SleepRetrainLog(Base):
    __tablename__ = "sleep_retrain_log"

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    sleep_start_time = Column(DateTime(timezone=True), nullable=False)
    timestamp = Column(Integer, nullable=False) # offset(ms) 1 ~ 36000
    measured_at = Column(DateTime(timezone=True), nullable=False, index=True) # sleep_start_time + timestamp(ms)
    acc_x = Column(Float, nullable=False)
    acc_y = Column(Float, nullable=False)
    acc_z = Column(Float, nullable=False)
    hr = Column(Float, nullable=False)
    sleep_stage = Column(Integer, nullable=False)
    sao2 = Column(Float, nullable=False, server_default="0")
    bvp = Column(Float, nullable=False, server_default="0")

class StressRetrainLog(Base):
    __tablename__ = "stress_retrain_log"

    id = Column(Integer, primary_key=True)
    user = Column(String, ForeignKey("user.email", ondelete="CASCADE"), index=True)
    heart_rate_bpm = Column(Float, nullable=False)
    hrv_sdnn_ms = Column(Float, nullable=False)
    hrv_rmssd_ms = Column(Float, nullable=False)
    acc_x_mean = Column(Float, nullable=False)
    acc_y_mean = Column(Float, nullable=False)
    acc_z_mean = Column(Float, nullable=False)
    acc_mag_mean = Column(Float, nullable=False)
    acc_mag_std = Column(Float, nullable=False)
    wrist_temperature_c_mean = Column(Float, nullable=False, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
