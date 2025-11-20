from pydantic import BaseModel, field_validator, ConfigDict
from datetime import datetime
from typing import Optional

# 로그 수집
class FallLogSchema(BaseModel):
    robot_id: str
    label: str
    occurred_at: Optional[datetime] = None

    @field_validator('robot_id', 'label')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('empty value')
        return v

# 낙상 알림
class FallCheckSchema(BaseModel):
    robot_id: str
    label: str
    occurred_at: datetime

    model_config = ConfigDict(from_attributes=True)
