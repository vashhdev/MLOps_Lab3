from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "urgency_model.pkl"

_model = None


class ModelNotFoundError(RuntimeError):
    """Raised when the trained model file does not exist yet."""


def load_model():
    """
    Load the trained pipeline from disk the first time it is needed and reuse it afterwards.
    Returns:
        Pipeline: The fitted TF-IDF + Logistic Regression pipeline.
    Raises:
        ModelNotFoundError: If the model file is missing (run train.py first).
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run `python train.py` from src/ first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_proba(texts):
    """
    Get the probability of every urgency label for each message.
    Args:
        texts (list[str]): Caller messages.
    Returns:
        list[dict[str, float]]: One {label: probability} dict per message.
    """
    model = load_model()
    probabilities = model.predict_proba(list(texts))
    labels = [str(label) for label in model.classes_]
    return [dict(zip(labels, map(float, row))) for row in probabilities]


def predict(text):
    """
    Predict the urgency label for a single message.
    Args:
        text (str): The caller's message.
    Returns:
        tuple[str, float]: The predicted label and its probability (confidence).
    """
    probabilities = predict_proba([text])[0]
    label = max(probabilities, key=probabilities.get)
    return label, probabilities[label]
