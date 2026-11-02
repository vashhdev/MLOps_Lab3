# MLOps Lab 3: Call Urgency Triage API (FastAPI)

A FastAPI service that reads a caller's short message to a healthcare call center and returns an urgency label (`urgent`, `normal` or `low`) together with a confidence score.

Based on the [FastAPI lab](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/FastAPI_Labs) from the MLOps course, which trains a Decision Tree on the Iris dataset. This version swaps Iris for a text classification problem.

| Caller message | Urgency |
|---|---|
| "my father collapsed and isn't responding" | `urgent` |
| "I need to refill my blood pressure prescription" | `normal` |
| "what are your office hours on Saturday" | `low` |

Label meaning:

- **urgent**: possible medical emergency, needs immediate attention (chest pain, stroke signs, heavy bleeding, overdose, suicidal thoughts, breathing problems, infant fever, ...).
- **normal**: needs action from the care team, but not right now (refills, appointments, test results, referrals, mild symptoms).
- **low**: general or administrative questions (hours, parking, billing, insurance, portal, records).

> This is a course project trained on a small synthetic dataset. It is not a medical device and must not be used to triage real patients.

## Project structure

```
MLOps_Lab3
├── data/
│   └── calls.csv          # 155 synthetic caller messages (text, label)
├── model/
│   └── urgency_model.pkl  # trained TF-IDF + LogisticRegression pipeline
├── src/
│   ├── __init__.py
│   ├── data.py            # load CSV, stratified 80/20 split
│   ├── train.py           # build, train, evaluate and save the pipeline
│   ├── predict.py         # load the model once, predict(text) -> (label, confidence)
│   └── main.py            # FastAPI app
├── tests/
│   └── test_api.py        # pytest + TestClient
├── pytest.ini
├── requirements.txt
└── README.md
```

## Dataset

`data/calls.csv` has 155 hand-written synthetic messages (51 urgent, 52 normal, 52 low) with two columns: `text` and `label`. No real personal data is used. The messages mix formal and casual tone, short and long phrasing, and typos ("cant breathe", "wont wake up"). Some are deliberately tricky, where keywords point the wrong way:

| Message | Label | Why it's tricky |
|---|---|---|
| "not urgent but my chest has felt tight since yesterday" | urgent | says "not urgent" |
| "Its probably nothing but I passed out twice this morning" | urgent | caller downplays it |
| "URGENT: please call me back about my bill" | low | says "URGENT" |
| "this is an emergency!!! I need to change my appointment time" | normal | says "emergency" |
| "what number do I call if it's an emergency after hours" | low | mentions "emergency" |
| "my dad had a stroke last year and needs his follow-up booked" | normal | mentions "stroke" |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train the model

```bash
cd src
python train.py
```

This loads `data/calls.csv`, makes a stratified 80/20 split (`random_state=42`), fits the pipeline, prints test accuracy and a classification report, and saves the pipeline to `model/urgency_model.pkl`.

Pipeline: `TfidfVectorizer(ngram_range=(1, 2), lowercase=True)` followed by `LogisticRegression(max_iter=1000, class_weight="balanced")`.

Output:

```
Train size: 124, test size: 31

Test accuracy: 0.935

              precision    recall  f1-score   support

         low      1.000     1.000     1.000        11
      normal      0.900     0.900     0.900        10
      urgent      0.900     0.900     0.900        10

    accuracy                          0.935        31
   macro avg      0.933     0.933     0.933        31
weighted avg      0.935     0.935     0.935        31
```

The test set only has 31 messages, so a single wrong prediction moves accuracy by about 3 points. Treat the numbers as a sanity check, not a benchmark.

## Run the API

```bash
cd src
uvicorn main:app --reload
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The model is loaded once when the app starts. If `model/urgency_model.pkl` is missing, the server still starts, but `/health` and the prediction endpoints return `503` until you run `train.py`.

### Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "model_loaded": true}` |
| POST | `/predict` | `{"message": "..."}` | one prediction |
| POST | `/predict/batch` | `{"messages": ["...", "..."]}` (1 to 50 items) | list of predictions, same order |

Each message must be 3 to 1000 characters and contain at least 3 non-whitespace characters.

### Example requests

Health check:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

Single prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "not urgent but my chest has felt tight since yesterday"}'
```

```json
{
  "urgency": "urgent",
  "confidence": 0.553,
  "all_probabilities": {"low": 0.209, "normal": 0.238, "urgent": 0.553}
}
```

Batch prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["my father collapsed and is not responding", "I need to refill my blood pressure prescription", "what are your office hours on Saturday"]}'
```

```json
[
  {"urgency": "urgent", "confidence": 0.596, "all_probabilities": {"low": 0.199, "normal": 0.205, "urgent": 0.596}},
  {"urgency": "normal", "confidence": 0.655, "all_probabilities": {"low": 0.174, "normal": 0.655, "urgent": 0.171}},
  {"urgency": "low",    "confidence": 0.618, "all_probabilities": {"low": 0.618, "normal": 0.182, "urgent": 0.201}}
]
```

Whitespace-only message:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "     "}'
```

```json
{"detail": "message must contain at least 3 non-whitespace characters."}
```

(HTTP 422.) A message shorter than 3 characters, or a batch with more than 50 messages, also returns 422 with Pydantic's validation details.

On Windows PowerShell, use `curl.exe` instead of `curl`, or `Invoke-RestMethod`:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"message": "where do I park"}'
```

## Run the tests

From the repository root (the model must be trained first):

```bash
pytest -v
```

`pytest.ini` adds `src/` to the import path, so the tests import `main` and `predict` the same way uvicorn does. The tests cover `/health`, an urgent and a low message on `/predict`, the batch endpoint, rejection of too-short, whitespace-only and oversized batch input, and the 503 response when the model file is missing.

## Changes from the original lab

- **Dataset**: replaced the built-in Iris dataset with `data/calls.csv`, 155 synthetic healthcare call center messages labeled `urgent`, `normal` or `low`.
- **Text input instead of numeric**: the API takes a free-text `message` instead of four flower measurements. Input is validated with `Field(min_length=3, max_length=1000)` plus a whitespace check.
- **TF-IDF + Logistic Regression instead of a Decision Tree**: a scikit-learn `Pipeline` with `TfidfVectorizer` (unigrams and bigrams) and a class-balanced `LogisticRegression`. The whole pipeline is saved as one `.pkl`, so the API sends raw text straight to the model.
- **Evaluation**: `train.py` uses a stratified 80/20 split and prints accuracy and a classification report (the original trained without reporting any metric).
- **Confidence scores**: responses include the confidence of the predicted label and the probability of every label, from `predict_proba`.
- **Batch endpoint**: `POST /predict/batch` scores up to 50 messages in one request.
- **Health endpoint**: `GET /health` replaces `GET /` and reports whether the model is loaded (503 if not).
- **Error handling**: `HTTPException` returns 503 when the model file is missing and 422 for empty or whitespace-only messages.
- **Model loaded once**: the original reloaded the `.pkl` on every request. Now it is loaded at startup and cached.
- **Path handling**: model and data paths are resolved relative to the source files, so the code works no matter which directory it is run from (the original used `../model/...`, which only worked from `src/`).
- **Tests**: added `tests/test_api.py` using FastAPI's `TestClient` and pytest.
- **Docs**: the FastAPI app has a title, description and request examples so `/docs` is self-explanatory.
- **Dependencies**: added `pandas`, `joblib`, `pytest` and `httpx` to `requirements.txt`, and updated versions.
