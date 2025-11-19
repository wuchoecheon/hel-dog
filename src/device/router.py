from fastapi import APIRouter, Depends, HTTPException, status, Response
from typing import Annotated
from sqlalchemy.orm import Session

from src.database import get_db
from src.auth.models import User
from src.auth.utils import get_user
from src.device.models import Device
from src.device.schemas import RegisterSchema

router = APIRouter(
    prefix="/api/device"
)

@router.post("/register")
def register_device(
    body: RegisterSchema,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    if db.get(Device, body.device_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Device already registered",
        )

    device = Device(
        device_id=body.device_id,
        owner=user.email,
    )

    db.add(device)
    db.commit()
    db.refresh(device)

    return {"response": "register success!"}

@router.delete("/{device_id}")
def delete_device(
    device_id: str,
    user: Annotated[User, Depends(get_user)],
    db: Session=Depends(get_db),
):
    device = db.get(Device, device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_400_NOT_FOUND,
            detail="Device not found",
        )

    if device.owner != user.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not owner of this Device",
        )

    db.delete(device)
    db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)