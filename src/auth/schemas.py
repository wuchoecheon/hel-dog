from pydantic import BaseModel, field_validator, EmailStr
from pydantic_core.core_schema import ValidationInfo

class SignupSchema(BaseModel):
    email: EmailStr
    password: str
    password2: str

    @field_validator('email', 'password', 'password2')
    def not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('empty value')
        return v
    
    @field_validator('password2')
    def passwords_match(cls, v, info: ValidationInfo):
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('password does not match')
        return v


class LoginSchema(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class SignupResponse(BaseModel):
    response: str


class CheckResponse(BaseModel):
    email: EmailStr
