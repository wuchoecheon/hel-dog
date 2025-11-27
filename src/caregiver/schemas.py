from pydantic import BaseModel, EmailStr


class CaregiverCreate(BaseModel):
    caregiver_email: EmailStr


class CaregiverResponse(BaseModel):
    id: int
    ward: EmailStr
    caregiver: EmailStr
    accepted: bool

    class Config:
        orm_mode = True
