from pydantic import BaseModel, field_validator

class SleepLogSchema(BaseModel):
    label: int

    @field_validator('label')
    def not_empty(cls, v):
        if v is None:
            raise ValueError('empty value')
        return v

class CreateSleepLogResponse(BaseModel):
    response: str

class GetSleepScoreResponse(BaseModel):
    response: str
    sleep_score: int
