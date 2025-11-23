from pydantic import BaseModel, field_validator

class RegisterSchema(BaseModel):
    device_id: str

    @field_validator('device_id')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('empty value')
        return v

class RegisterDeviceResponse(BaseModel):
    response: str
