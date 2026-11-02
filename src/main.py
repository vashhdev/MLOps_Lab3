import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from predict import ModelNotFoundError, load_model, predict_proba

logger = logging.getLogger("uvicorn.error")

MIN_MESSAGE_LENGTH = 3
MAX_MESSAGE_LENGTH = 1000
MAX_BATCH_SIZE = 50

MessageText = Annotated[
    str,
    Field(
        min_length=MIN_MESSAGE_LENGTH,
        max_length=MAX_MESSAGE_LENGTH,
        description="What the caller said, as plain text.",
    ),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup. If it is missing, keep serving and return 503 on predictions."""
    try:
        load_model()
        logger.info("Urgency model loaded.")
    except ModelNotFoundError as exc:
        logger.warning(str(exc))
    yield


app = FastAPI(
    title="Call Urgency Triage API",
    description=(
        "Classifies a caller's message to a healthcare call center as **urgent**, **normal** or **low** "
        "priority, with a confidence score. Model: TF-IDF (1-2 grams) + Logistic Regression trained on "
        "synthetic call transcripts.\n\n"
        "This is a lab project. It is not a medical device and must not replace emergency services."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class CallInput(BaseModel):
    """A single caller message."""

    message: MessageText

    model_config = {"json_schema_extra": {"examples": [{"message": "my father collapsed and isn't responding"}]}}


class BatchCallInput(BaseModel):
    """Several caller messages scored in one request."""

    messages: list[MessageText] = Field(min_length=1, max_length=MAX_BATCH_SIZE)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        "my father collapsed and isn't responding",
                        "I need to refill my blood pressure prescription",
                        "what are your office hours on Saturday",
                    ]
                }
            ]
        }
    }


class UrgencyResponse(BaseModel):
    """Prediction for one message."""

    urgency: str = Field(description="Predicted label: urgent, normal or low.")
    confidence: float = Field(description="Probability of the predicted label, rounded to 3 decimals.")
    all_probabilities: dict[str, float] = Field(description="Probability of every label.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


def _clean_messages(messages, is_batch):
    """Strip whitespace and reject messages that are empty or too short once stripped (422)."""
    cleaned = []
    for i, message in enumerate(messages):
        text = message.strip()
        if len(text) < MIN_MESSAGE_LENGTH:
            where = f"messages[{i}]" if is_batch else "message"
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"{where} must contain at least {MIN_MESSAGE_LENGTH} non-whitespace characters.",
            )
        cleaned.append(text)
    return cleaned


def _score(messages, is_batch=False):
    """Run the model and map its output to response objects, translating failures to HTTP errors."""
    texts = _clean_messages(messages, is_batch)
    try:
        results = predict_proba(texts)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    responses = []
    for probabilities in results:
        label = max(probabilities, key=probabilities.get)
        responses.append(
            UrgencyResponse(
                urgency=label,
                confidence=round(probabilities[label], 3),
                all_probabilities={k: round(v, 3) for k, v in probabilities.items()},
            )
        )
    return responses


@app.get("/health", response_model=HealthResponse)
def health(response: Response):
    """Report whether the API is up and the model is loaded. Returns 503 if the model file is missing."""
    try:
        load_model()
        return HealthResponse(status="ok", model_loaded=True)
    except ModelNotFoundError:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", model_loaded=False)


# Prediction endpoints are plain `def` (not `async def`) because scikit-learn inference is
# blocking CPU work. FastAPI runs sync endpoints in a thread pool so they don't block the event loop.
@app.post("/predict", response_model=UrgencyResponse)
def predict_urgency(call: CallInput):
    """
    Predict the urgency of a single caller message.

    Example request: `{"message": "my father collapsed and isn't responding"}`
    """
    return _score([call.message])[0]


@app.post("/predict/batch", response_model=list[UrgencyResponse])
def predict_urgency_batch(batch: BatchCallInput):
    """Predict the urgency of up to 50 caller messages. Results are returned in the same order."""
    return _score(batch.messages, is_batch=True)
