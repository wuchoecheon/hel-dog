
from datetime import datetime
from pydantic import BaseModel, Field

class HealthDataSchema(BaseModel):    
	timestamp: datetime = Field(alias="timestamp")    
	heart_rate_data: int = Field(alias="heartRateData")    
	oxygen_saturation: float = Field(alias="oxygenSaturation")    
	stress_level: str = Field(alias="stressLevel")
