import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app,model_store

#1. create a fake trained model
class FakePipeline:
    feature_names_in_ = np.array(['age', 'win_odds', 'weight_pounds'])
    def predict_proba(self, X):
        # Return predictable probabilities for testing
        probabilities = np.array(
            [0.2 + 0.1 * i for i in range(len(X))]
        )
        return np.column_stack([
            1-probabilities, probabilities
        ])

# 2. Create a reusable API client
@pytest.fixture
def client():
    # Start FastAPI without running its real model-loading lifespan.
    # Inject the fake model into the store instead.
    model_store["pipeline"] = FakePipeline()
    try:
        yield TestClient(app)
    finally:
        model_store.clear()
# 3. Create example request data
@pytest.fixture
def sample_request():
    return {
        "race_id": "race_001",
        "runners": [
            {
                "runner_id": "horse_1",
                "features": {
                    "age": 5,
                    "win_odds": 3.5,
                    "weight_pounds": 125
                }
            },
            {
                "runner_id": "horse_2",
                "features": {
                    "age": 4,
                    "win_odds": 6.0,
                    "weight_pounds": 130
                }
            }
        ]
    }

# 4. Test home endpoint
def test_home(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Horse Racing API is running"
    }
# 5. Test health endpoint
def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# 6. Test prediction endpoint
def test_predict(client, sample_request):
    # lets Python simulate that HTTP request directly.
    response = client.post("/predict", json=sample_request)

    assert response.status_code == 200

    data = response.json()

    assert data["race_id"] == "race_001"
    assert len(data["predictions"]) == 2

    probabilities = [
        runner["win_probability"]
        for runner in data["predictions"]
    ]

    assert sum(probabilities) == pytest.approx(1.0)
    assert all(0 <= p <= 1 for p in probabilities)

# 7. Test missing features
def test_missing_feature(client, sample_request):
    del sample_request["runners"][0]["features"]["age"]

    response = client.post("/predict", json=sample_request)

    assert response.status_code == 422

    data = response.json()
    assert "age" in data["detail"]["missing_features"]

def test_invalid_feature_type(client,sample_request):
    sample_request["runners"][0]["features"]["age"] = "unknown"

    response = client.post("/predict", json=sample_request)

    assert response.status_code == 422

def test_feature_order(client, sample_request,monkeypatch):
    # Shuffle the order of features in the first runner
    sample_request["runners"][0]["features"] = {
        "weight_pounds": 125,
        "win_odds": 3.5,
        "age": 5
    }

    captured = {}
    def fake_predict_proba(X):
        captured['X'] = X.copy()
        probabilities = np.array([0.2, 0.3])
        return np.column_stack([1 - probabilities, probabilities])
    # Replace the fake model's prediction method temporarily
    monkeypatch.setattr(
        model_store["pipeline"],
        "predict_proba",
        fake_predict_proba
    )

    # Act: call the API
    response = client.post(
        "/predict",
        json=sample_request
    )

    # Assert
    assert response.status_code == 200

    X = captured["X"]

    assert list(X.columns) == [
        "age",
        "win_odds",
        "weight_pounds"
    ]

    assert X.iloc[0]["age"] == 5
    assert X.iloc[0]["win_odds"] == 3.5
    assert X.iloc[0]["weight_pounds"] == 125