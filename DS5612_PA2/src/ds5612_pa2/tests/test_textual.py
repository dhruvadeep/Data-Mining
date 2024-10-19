# GiG

import asyncio

import pytest
from textual.widgets import Button, ProgressBar, Select, TextArea

from ds5612_pa2.code import pipeline_configs
from ds5612_pa2.code.task2.textual_demo import TextualClassifier


test_features = ["1 2 3 4 5 6 7 8 9 0", "0 0 0 0 0 0 0 0 0 0", "-1 -1 -1 -1 -1 -1 -1 -1 -1 -1"]


def get_probabilities(features_str: str, classifier: str) -> tuple[float, float]:
    features = [float(f) for f in features_str.split(" ")]
    return pipeline_configs.get_prediction_probabilities(features, classifier)


@pytest.mark.t2textual()
async def test_existence_of_elements() -> None:
    app = TextualClassifier()
    async with app.run_test() as pilot:
        widget = pilot.app.query_one("#classifier", Select)
        assert widget is not None, "Select widget for classifier is not found"

        widget = pilot.app.query_one("#input-text", TextArea)
        assert widget is not None, "TextArea widget for features is not found"

        widget = pilot.app.query_one("#predict", Button)
        assert widget is not None, "Button widget for prediction is not found"

        widget = pilot.app.query_one("#positive", ProgressBar)
        assert widget is not None, "ProgressBar widget for positive_score is not found"

        widget = pilot.app.query_one("#negative", ProgressBar)
        assert widget is not None, "ProgressBar widget for negative_score is not found"


@pytest.mark.t2textual()
async def test_with_knn() -> None:
    app = TextualClassifier()
    async with app.run_test() as pilot:
        for features_str in test_features:
            classifier = pipeline_configs.ValidClassifierNames.KNN.name
            probabilities = get_probabilities(features_str, classifier)

            widget = pilot.app.query_one("#classifier", Select)
            widget.value = classifier

            widget = pilot.app.query_one("#input-text", TextArea)
            widget.text = features_str

            await pilot.click("#predict")
            # Wait for the code to do the processing
            await asyncio.sleep(2)

            widget = pilot.app.query_one("#positive", ProgressBar)
            assert widget.progress == probabilities[0]

            widget = pilot.app.query_one("#negative", ProgressBar)
            assert widget.progress == probabilities[1]


@pytest.mark.t2textual()
async def test_with_naive_bayes() -> None:
    app = TextualClassifier()
    async with app.run_test() as pilot:
        for features_str in test_features:
            classifier = pipeline_configs.ValidClassifierNames.NB.name
            probabilities = get_probabilities(features_str, classifier)

            widget = pilot.app.query_one("#classifier", Select)
            widget.value = classifier

            widget = pilot.app.query_one("#input-text", TextArea)
            widget.text = features_str

            await pilot.click("#predict")
            # Wait for the code to do the processing
            await asyncio.sleep(2)

            widget = pilot.app.query_one("#positive", ProgressBar)
            assert widget.progress == probabilities[0]

            widget = pilot.app.query_one("#negative", ProgressBar)
            assert widget.progress == probabilities[1]
