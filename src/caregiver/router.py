from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from src.database import get_db
from src.auth.utils import get_user
from src.auth.models import User
from . import models, schemas

router = APIRouter(
    prefix="/caregiver",
    tags=["Caregiver"],
)

@router.post("/request", response_model=schemas.CaregiverResponse, status_code=status.HTTP_201_CREATED)
def create_caregiver_request(
    caregiver_request: schemas.CaregiverCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user)
):
    target_user = db.query(User).filter(User.email == caregiver_request.caregiver_email).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="Caregiver not found")

    if current_user.email == caregiver_request.caregiver_email:
        raise HTTPException(status_code=400, detail="Cannot request yourself as a caregiver")

    existing_request = db.query(models.Caregiver).filter(
        (models.Caregiver.ward == current_user.email) &
        (models.Caregiver.caregiver == caregiver_request.caregiver_email)
    ).first()
    if existing_request:
        if existing_request.accepted:
            raise HTTPException(status_code=400, detail="Already accepted as caregiver")
        else:
            raise HTTPException(status_code=400, detail="Caregiver request already pending")

    db_caregiver_request = models.Caregiver(
        ward=current_user.email,
        caregiver=caregiver_request.caregiver_email,
        accepted=False
    )
    db.add(db_caregiver_request)
    try:
        db.commit()
        db.refresh(db_caregiver_request)
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Caregiver request could not be created due to data integrity issue.")
    return db_caregiver_request


@router.get("/requests/me", response_model=List[schemas.CaregiverResponse])
def list_my_caregiver_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user)
):
    # List requests where current_user is the caregiver (receiver) and request is not accepted yet
    pending_requests = db.query(models.Caregiver).filter(
        (models.Caregiver.caregiver == current_user.email) &
        (models.Caregiver.accepted == False)
    ).all()
    return pending_requests


@router.put("/request/{request_id}/accept", response_model=schemas.CaregiverResponse)
def accept_caregiver_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user)
):
    db_request = db.query(models.Caregiver).filter(
        (models.Caregiver.id == request_id) &
        (models.Caregiver.caregiver == current_user.email) # Ensure only the intended caregiver can accept
    ).first()

    if not db_request:
        raise HTTPException(status_code=404, detail="Caregiver request not found or you are not authorized to accept it")

    if db_request.accepted:
        raise HTTPException(status_code=400, detail="Caregiver request already accepted")

    db_request.accepted = True
    db.commit()
    db.refresh(db_request)
    return db_request


@router.delete("/request/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_caregiver_request(
    request_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_user)
):
    db_request = db.query(models.Caregiver).filter(
        (models.Caregiver.id == request_id) &
        ((models.Caregiver.ward == current_user.email) | # Either the sender
         (models.Caregiver.caregiver == current_user.email)) # Or the receiver can delete
    ).first()

    if not db_request:
        raise HTTPException(status_code=404, detail="Caregiver request not found or you are not authorized to delete it")

    db.delete(db_request)
    db.commit()
    return {"message": "Caregiver request deleted successfully"}