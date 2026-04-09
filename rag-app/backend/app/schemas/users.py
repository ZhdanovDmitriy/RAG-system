from pydantic import BaseModel, Field
from typing import List, Optional


class UserIn(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)

class UserOut(BaseModel):
    short: List[str]
    long: List[str]


# --- Test schemas ---

class TestQuestion(BaseModel):
    id: int
    question: str

class TestResponse(BaseModel):
    questions: List[TestQuestion]

class TestAnswer(BaseModel):
    question_id: int
    answer: str  # может быть пустой строкой

class TestSubmission(BaseModel):
    answers: List[TestAnswer]

class TestResultItem(BaseModel):
    question_id: int
    question: str
    is_correct: bool
    similarity: float
    best_correct_answer: Optional[str] = None

class TestResult(BaseModel):
    score: int
    total: int
    grade: str
    results: List[TestResultItem]