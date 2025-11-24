from datetime import datetime
from pydantic import BaseModel, Field

class HealthDataSchema(BaseModel):
	timestamp: datetime = Field(alias="timestamp")
	heart_rate_data: int = Field(alias="heartRateData")
	oxygen_saturation: float = Field(alias="oxygenSaturation")
	stress_level: str = Field(alias="stressLevel")

class CreateHealthDataResponse(BaseModel):
    response: str

class HealthScoreComponents(BaseModel):
    sleep_score: float
    cough_score: float
    cough_count_3h: int
    stress_score: float
    stress_count_12h: int

class HealthScoreResponse(BaseModel):
    response: str
    health_score: float
    status: str
    components: HealthScoreComponents
