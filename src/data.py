from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "calls.csv"


def load_data(path=DATA_PATH):
    """
    Load the call urgency dataset.
    Args:
        path (str | Path): Location of the CSV file with `text` and `label` columns.
    Returns:
        X (pandas.Series): The caller messages.
        y (pandas.Series): The urgency labels ("urgent", "normal" or "low").
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"])
    X = df["text"].str.strip()
    y = df["label"].str.strip().str.lower()
    return X, y


def split_data(X, y):
    """
    Split the data into stratified training and testing sets (80/20).
    Args:
        X (pandas.Series): The caller messages.
        y (pandas.Series): The urgency labels.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test
