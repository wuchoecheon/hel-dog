from fastapi import APIRouter

from typing import Annotated

from fastapi import Depends

from sqlalchemy.orm import Session
from src.database import get_db
from src.auth.models import User
from src.device.utils import map_device_to_user
from src.health.schemas import HealthDataSchema, CreateHealthDataResponse
from src.health.models import HealthData

router = APIRouter(prefix="/api/health")

@router.post("/{device_id}", response_model=CreateHealthDataResponse)
def create_health_data(
    device_id: str,
    user: Annotated[User,Depends(map_device_to_user)],
    db: Annotated[Session, Depends(get_db)],    
    body: HealthDataSchema
  ):

  db_health_data = HealthData(
    user=user, 
    timestamp=body.timestamp,
    heart_rate_data=body.heart_rate_data,
    oxygen_saturation=body.oxygen_saturation,
    stress_level=body.stress_level
  )
  
  db.add(db_health_data)
  db.commit()
  db.refresh(db_health_data)
  return {"response": "receive success!"}


# @router.post("/{device_id}")
# def use_this(
#     device_id: str,
#     user: Annotated[User,Depends(map_device_to_user)],
#     db: Annotated[Session, Depends(get_db)],    
#     body: HealthDataSchema
#   ):

#   user=user, 
#   timestamp=body.timestamp,
#   heart_rate_data=body.heart_rate_data,
#   oxygen_saturation=body.oxygen_saturation,
#   stress_level=body.stress_level

#   return {"response": "receive success!"}