from datetime import datetime
from pydantic import BaseModel

class SleepRetrainLogSchema(BaseModel):
    sleep_start_time: datetime
    timestamp: int
    acc_x: float
    acc_y: float
    acc_z: float
    hr: float
    sleep_stage: int
    sao2: float = 0.0
    bvp: float = 0.0

    class Config:
        from_attributes = True

class StressRetrainLogSchema(BaseModel):
    heart_rate_bpm: float
    hrv_sdnn_ms: float
    hrv_rmssd_ms: float
    acc_x_mean: float
    acc_y_mean: float
    acc_z_mean: float
    acc_mag_mean: float
    acc_mag_std: float
    wrist_temperature_c_mean: float = 0.0
