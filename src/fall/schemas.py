from pydantic import BaseModel
from typing import List
from datetime import datetime

class CreateFallLogResponse(BaseModel):
    response: str

class FallLogEntry(BaseModel):
    timestamp: datetime # Corrected: not Optional

class FallResponseEntry(BaseModel):
    id: int 

class GetFallLogResponse(BaseModel):
    response: str
    fall_num: int
    fall_log: List[FallLogEntry]
