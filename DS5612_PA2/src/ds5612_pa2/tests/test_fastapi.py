# GiG

from io import StringIO

import pytest
from fastapi.testclient import TestClient

from ds5612_pa2.code import pipeline_configs
from ds5612_pa2.code.task2.fastapi_demo import DetailedPredictionResponse, PredictionResponse, app


client = TestClient(app)


@pytest.mark.parametrize("classifier", [v.value for v in pipeline_configs.ValidClassifierNames])
@pytest.mark.t2fastAPI()
def test_predict_v1_correct_data(classifier: str) -> None:
    # Test data
    features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Make a POST request to the predict endpoint
    response = client.post("/v1/predict", json={"classifier": classifier, "features": features})

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if the response is a list containing two floats
    d = PredictionResponse(**response.json())

    prediction = pipeline_configs.get_prediction_class(features, classifier)

    assert d.predicted_class == prediction


@pytest.mark.parametrize("classifier", [v.value for v in pipeline_configs.ValidClassifierNames])
@pytest.mark.t2fastAPI()
def test_predict_v2_correct_data(classifier: str) -> None:
    # Test data
    features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Make a POST request to the predict endpoint
    response = client.post("/v2/predict", json={"classifier": classifier, "features": features})

    # Check if the response status code is 200 (OK)
    assert response.status_code == 200

    # Check if the response is a list containing two floats
    d = DetailedPredictionResponse(**response.json())

    probabilities = pipeline_configs.get_prediction_probabilities(features, classifier)
    prediction = pipeline_configs.get_prediction_class(features, classifier)

    assert d.probabilities == probabilities
    assert d.predicted_class == prediction


@pytest.mark.t2fastAPI()
def test_predict_v2_invalid_classifier() -> None:
    # Test with an invalid classifier
    test_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    response = client.post("/v2/predict", json={"classifier": "INVALID", "features": test_data})

    # Check if the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422


@pytest.mark.t2fastAPI()
def test_predict_v2_invalid_data() -> None:
    # Test with invalid data (non-float values)
    invalid_data = ["string", 0.2, 0.3, "not a float", 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    response = client.post("/v2/predict", json={"classifier": "NB", "features": invalid_data})

    # Check if the response status code is 422 (Unprocessable Entity)
    assert response.status_code == 422


@pytest.mark.t2fastAPI()
def test_batch_predict() -> None:
    # Test data
    classifier = pipeline_configs.ValidClassifierNames.DT
    features_list = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [-1, 0, -1, 0, -1, 0, -1, 0, -1, 0],
    ]

    features_as_str = "\n".join(
        " ".join(str(item) for item in inner_list) for inner_list in features_list
    )

    test_file = StringIO(features_as_str)
    assert test_file is not None

    # Make a POST request to the batch predict endpoint
    file_to_upload = {"input_file": ("test.txt", test_file.getvalue(), "text/plain")}
    response = client.post("/batch_predict/", files=file_to_upload)

    assert response.status_code == 200

    response_json: list[dict] = response.json()
    for index in range(len(features_list)):
        features = features_list[index]
        d = DetailedPredictionResponse(**response_json[index])

        probabilities = pipeline_configs.get_prediction_probabilities(features, classifier)
        prediction = pipeline_configs.get_prediction_class(features, classifier)

        assert d.probabilities == probabilities
        assert d.predicted_class == prediction


def test_classify_upload_invalid_file() -> None:
    test_content = "a b c\nd e f\ng h i"
    test_file = StringIO(test_content)

    response = client.post(
        "/batch_predict/", files={"input_file": ("invalid.txt", test_file.getvalue(), "text/plain")}
    )

    assert response.status_code == 422  # Unprocessable Entity


def test_classify_upload_missing_file() -> None:
    response = client.post(
        "/batch_predict/",
    )

    assert response.status_code == 422  # Unprocessable Entity
