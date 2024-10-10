# GiG
# GiG

import subprocess
import time

import pytest
import requests
from gradio_client import Client

from ds5612_pa2.code import pipeline_configs, utils


def client_output_to_probabilities(client_output: tuple) -> dict:
    prediction_prob = client_output[0]["confidences"]
    return {d["label"]: d["confidence"] for d in prediction_prob}


def local_predict(classifier: str, item_str: str) -> tuple[dict[str, float], str]:
    """Predict using the specified classifier and input values."""
    class_names = ["Positive", "Negative"]
    probabilities = [0.0, 1.0]
    error_msg = "Success"
    try:
        item = [float(elem) for elem in item_str.split()]
        probabilities = pipeline_configs.get_prediction_probabilities(item, classifier)
    except Exception as e:  # noqa: BLE001
        error_msg = str(e)
    return {class_names[i]: probabilities[i] for i in range(2)}, error_msg


# Start the Gradio app in a separate process
@pytest.fixture(scope="module")
def gradio_app():  # noqa: ANN201
    GRADIO_APP_PATH = utils.get_project_root() / "src/ds5612_pa2/code/task2/gradio_demo.py"
    process = subprocess.Popen(["rye", "run", "python", GRADIO_APP_PATH])  # noqa: S603, S607
    time.sleep(5)  # Wait for the app to start
    yield process
    process.terminate()  # Stop the app after tests


@pytest.fixture(scope="module")
def client(gradio_app) -> Client:  # noqa: ANN001, ARG001
    # Create a client
    return Client("http://127.0.0.1:7860")


@pytest.mark.t2gradio()
def test_app_is_running(gradio_app) -> None:  # noqa: ANN001, ARG001
    response = requests.get("http://127.0.0.1:7860", timeout=10)
    assert response.status_code == 200


@pytest.mark.parametrize("classifier", ["DecisionTree", "NaiveBayes", "KNN"])
@pytest.mark.t2gradio()
def test_prediction(client: Client, classifier: str) -> None:
    # Test with a positive example
    features = "1 2 3 4 5 6 7 8 9 0"
    expected_probabilities = local_predict(classifier, features)[0]

    client_output = client.predict(classifier, features, api_name="/predict")
    estimated_probabilities = client_output_to_probabilities(client_output)

    assert estimated_probabilities == expected_probabilities
