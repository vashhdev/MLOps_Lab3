from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from data import load_data, split_data

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "urgency_model.pkl"


def build_pipeline():
    """
    Build the text classification pipeline: TF-IDF features + Logistic Regression.
    Returns:
        Pipeline: An unfitted scikit-learn pipeline.
    """
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def fit_model(X_train, y_train):
    """
    Train the urgency classification pipeline.
    Args:
        X_train (pandas.Series): Training messages.
        y_train (pandas.Series): Training labels.
    Returns:
        Pipeline: The fitted pipeline.
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Print accuracy and a per-class classification report on the test set.
    Args:
        pipeline (Pipeline): The fitted pipeline.
        X_test (pandas.Series): Test messages.
        y_test (pandas.Series): Test labels.
    """
    y_pred = pipeline.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
    print(classification_report(y_test, y_pred, digits=3))


def save_model(pipeline, path=MODEL_PATH):
    """
    Save the fitted pipeline to disk with joblib.
    Args:
        pipeline (Pipeline): The fitted pipeline.
        path (str | Path): Destination file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train size: {len(X_train)}, test size: {len(X_test)}\n")
    model = fit_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
