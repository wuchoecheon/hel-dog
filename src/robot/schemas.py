from pydantic import BaseModel, field_validator, EmailStr

class RegisterSchema(BaseModel):
    robot_id: str

    @field_validator('robot_id')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('empty value')
        return v

class RegisterRobotResponse(BaseModel):
    response: str