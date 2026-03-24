from pydantic import BaseModel, Field

class UserIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class UserOut(BaseModel):
    answer: str
