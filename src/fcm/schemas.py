from pydantic import BaseModel, field_validator, EmailStr, UUID4
from pydantic_core.core_schema import ValidationInfo

class FCMRegisterSchema(BaseModel):
    uuid: UUID4
    fcm_token: str
    
    @field_validator('uuid', 'fcm_token')
    def not_empty(cls, v):
        if not v:
            raise ValueError('empty value')
        return v