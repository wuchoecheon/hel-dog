from pydantic import BaseModel
from datetime import datetime

class CoughLogSchema(BaseModel):
    robot_id: str
