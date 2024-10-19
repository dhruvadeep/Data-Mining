# GiG
import pytest

from ds5612_pa2.code import utils
from ds5612_pa2.code.error_estimation import (
    ErrorEstimationAlgos,
    ErrorEstimator,
    estimate_error,
    get_production_score,
)
from ds5612_pa2.code.pipeline_configs import DatasetConfig, get_classifier_config
from ds5612_pa2.tests.conftest import compare_dicts


EXPECTED_CORRECT_ERROR_ESTIMATES = {
    "DT": [
        {
            "algorithm": "TRAIN",
            "true_err": 0.7512,
            "pred_err": 0.7595,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "TRAIN_VAL",
            "true_err": 0.7512,
            "pred_err": 0.7555,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "CV_TRAIN_VAL",
            "true_err": 0.7512,
            "pred_err": 0.757,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "BOOTSTRAP_632",
            "true_err": 0.7512,
            "pred_err": 0.7549438448836538,
            "pred_err_low": 0.7354293425653484,
            "pred_err_high": 0.7694513935789313,
        },
        {
            "algorithm": "BOOTSTRAP_OOB",
            "true_err": 0.7512,
            "pred_err": 0.7535136786133763,
            "pred_err_low": 0.7283066561320729,
            "pred_err_high": 0.7755224581945117,
        },
    ],
    "KNN": [
        {
            "algorithm": "TRAIN",
            "true_err": 0.941,
            "pred_err": 0.948,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "TRAIN_VAL",
            "true_err": 0.941,
            "pred_err": 0.9465,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "CV_TRAIN_VAL",
            "true_err": 0.941,
            "pred_err": 0.9375,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "BOOTSTRAP_632",
            "true_err": 0.941,
            "pred_err": 0.9339529504797044,
            "pred_err_low": 0.9225984742748962,
            "pred_err_high": 0.9448013155886933,
        },
        {
            "algorithm": "BOOTSTRAP_OOB",
            "true_err": 0.941,
            "pred_err": 0.927175491265355,
            "pred_err_low": 0.9114411213667842,
            "pred_err_high": 0.9422943145035935,
        },
    ],
    "NB": [
        {
            "algorithm": "TRAIN",
            "true_err": 0.8637,
            "pred_err": 0.859,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "TRAIN_VAL",
            "true_err": 0.8637,
            "pred_err": 0.859,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "CV_TRAIN_VAL",
            "true_err": 0.8637,
            "pred_err": 0.8630000000000001,
            "pred_err_low": 0.0,
            "pred_err_high": 1.0,
        },
        {
            "algorithm": "BOOTSTRAP_632",
            "true_err": 0.8637,
            "pred_err": 0.863202316139284,
            "pred_err_low": 0.8494602660187789,
            "pred_err_high": 0.8786156929401474,
        },
        {
            "algorithm": "BOOTSTRAP_OOB",
            "true_err": 0.8637,
            "pred_err": 0.8630406268026644,
            "pred_err_low": 0.8417675751541156,
            "pred_err_high": 0.8848188179432713,
        },
    ],
}


@pytest.mark.parametrize("classifier", ["DT", "KNN", "NB"])
@pytest.mark.t4_error_estimator()
def test_decision_tree_error_estimates(classifier: str) -> None:
    d = DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    d.load_data()
    ml_model_config = get_classifier_config(classifier)
    ml_model = ml_model_config.create_classifier()

    expected_error_estimates = EXPECTED_CORRECT_ERROR_ESTIMATES[classifier]
    for index, algo in enumerate(ErrorEstimationAlgos):
        error_estimator = estimate_error(d, ml_model, ErrorEstimator(algorithm=algo))
        error_estimator.true_err = get_production_score(d, ml_model_config)

        compare_dicts(dict(error_estimator), expected_error_estimates[index])
