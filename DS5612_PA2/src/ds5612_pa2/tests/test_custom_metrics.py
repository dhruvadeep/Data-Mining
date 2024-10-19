# GiG

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from ds5612_pa2.code import pipeline_configs
from ds5612_pa2.code.custom_metrics import create_cost_based_scorer


def get_predicted_and_actual_y(classifier: str) -> tuple[NDArray, NDArray]:
    ml_pipeline = pipeline_configs.get_simple_ml_pipeline(classifier)
    ml_pipeline.train()

    test_X, test_y = ml_pipeline.dataset.test_X, ml_pipeline.dataset.test_y
    assert ml_pipeline.classifier is not None
    pred_y = ml_pipeline.classifier.predict(test_X)

    return test_y, pred_y


@pytest.mark.parametrize("classifier", [v.value for v in pipeline_configs.ValidClassifierNames])
@pytest.mark.t3_custom_metrics()
def test_custom_metric1(classifier: str) -> None:
    test_y, pred_y = get_predicted_and_actual_y(classifier)
    cm = confusion_matrix(test_y, pred_y)

    custom_scorer = create_cost_based_scorer(0, 1, 1, 0)
    assert custom_scorer(test_y, pred_y) == np.sum(cm * np.array([[0, 1], [1, 0]]))

    custom_scorer = create_cost_based_scorer(1, 2, 2, 0)
    assert custom_scorer(test_y, pred_y) == np.sum(cm * np.array([[1, 2], [2, 0]]))
