from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        description="Question to ask about the indexed documents.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str):

        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot be empty."
            )

        if value.lower() == "string":
            raise ValueError(
                "Please enter an actual question."
            )

        return value


class Source(BaseModel):
    source: str
    page: int | str
    rerank_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    attempts: int


class HealthResponse(BaseModel):
    status: str