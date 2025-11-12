from pydantic import BaseModel, field_validator, EmailStr

class RegisterSchema(BaseModel):
    robot_id: str
    user_email: EmailStr

    @field_validator('robot_id', 'user_email')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('empty value')
        return v

class DeleteSchema(BaseModel):
    user_email: EmailStr

    @field_validator('user_email')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('empty value')
        return v
