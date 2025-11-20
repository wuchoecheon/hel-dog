from typing import Annotated

from sqlalchemy.orm import Session
from src.device.models import Device
from src.database import get_db 

from fastapi import Depends, HTTPException, status

def map_device_to_user(device_id: str, db: Annotated[Session, Depends(get_db)]):
    device = db.query(Device).filter(Device.device_id == device_id).first()
    return device.owner
