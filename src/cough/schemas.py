from pydantic import BaseModel
from datetime import datetime
from typing import List

class CreateCoughLogResponse(BaseModel):
    response: str

class CoughLogBase(BaseModel):
    id: int
    user: str
    timestamp: datetime

class GetCoughLogResponse(BaseModel):
    response: str
    cough_num: int
    cough_log: List[CoughLogBase]

class GetCoughLogDetailResponse(BaseModel):
    response: str
    daily_counts: List[int]
    total_coughs: int
