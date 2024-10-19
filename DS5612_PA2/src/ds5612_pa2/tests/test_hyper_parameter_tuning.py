# GiG
import pytest

from ds5612_pa2.code.hp_configs import get_hp_ml_pipeline
from ds5612_pa2.tests.conftest import compare_dicts


EXPECTED_ACCURACY_AFTER_HP_TUNING = {
    "RandomizedSearch": {
        "0.0": {
            "precision": 0.910295616717635,
            "recall": 0.9206185567010309,
            "f1-score": 0.9154279856483855,
            "support": 970.0,
        },
        "1.0": {
            "precision": 0.9244357212953876,
            "recall": 0.9145631067961165,
            "f1-score": 0.9194729136163983,
            "support": 1030.0,
        },
        "accuracy": 0.9175,
        "macro avg": {
            "precision": 0.9173656690065113,
            "recall": 0.9175908317485737,
            "f1-score": 0.9174504496323919,
            "support": 2000.0,
        },
        "weighted avg": {
            "precision": 0.9175777705751776,
            "recall": 0.9175,
            "f1-score": 0.9175111235519121,
            "support": 2000.0,
        },
    },
    "GridSearch": {
        "0.0": {
            "precision": 0.8982706002034588,
            "recall": 0.9103092783505154,
            "f1-score": 0.9042498719918075,
            "support": 970.0,
        },
        "1.0": {
            "precision": 0.9144542772861357,
            "recall": 0.9029126213592233,
            "f1-score": 0.9086468001954079,
            "support": 1030.0,
        },
        "accuracy": 0.9065,
        "macro avg": {
            "precision": 0.9063624387447973,
            "recall": 0.9066109498548693,
            "f1-score": 0.9064483360936078,
            "support": 2000.0,
        },
        "weighted avg": {
            "precision": 0.9066051939010374,
            "recall": 0.9065,
            "f1-score": 0.9065142900166618,
            "support": 2000.0,
        },
    },
    "HalvingGridSearchCV": {
        "0.0": {
            "precision": 0.9142857142857143,
            "recall": 0.9237113402061856,
            "f1-score": 0.918974358974359,
            "support": 970.0,
        },
        "1.0": {
            "precision": 0.9274509803921569,
            "recall": 0.9184466019417475,
            "f1-score": 0.9229268292682927,
            "support": 1030.0,
        },
        "accuracy": 0.921,
        "macro avg": {
            "precision": 0.9208683473389356,
            "recall": 0.9210789710739666,
            "f1-score": 0.9209505941213258,
            "support": 2000.0,
        },
        "weighted avg": {
            "precision": 0.9210658263305322,
            "recall": 0.921,
            "f1-score": 0.9210098811757348,
            "support": 2000.0,
        },
    },
}


@pytest.mark.parametrize("hp_name", ["RandomizedSearch", "GridSearch", "HalvingGridSearchCV"])
@pytest.mark.t5_hp_tuner()
def test_decision_tree_error_estimates(hp_name: str) -> None:
    hp_pipeline = get_hp_ml_pipeline("DT", hp_name)
    hp_pipeline.run_pipeline(verbose=False)
    report = hp_pipeline.print_evaluation_results(verbose=False)

    compare_dicts(report, EXPECTED_ACCURACY_AFTER_HP_TUNING[hp_name])
