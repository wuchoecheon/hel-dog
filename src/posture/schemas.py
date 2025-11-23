from pydantic import BaseModel, field_validator

class PostureLogSchema(BaseModel):
    label: int

    @field_validator('label')
    def not_empty(cls, v):
        if v is None:
            raise ValueError('empty value')
        return v
