# GiG

from enum import Enum

import numpy as np
from mlxtend.evaluate import bootstrap_point632_score
from pydantic import BaseModel, ConfigDict
from rich import print
from rich.table import Table
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from ds5612_pa2.code import utils
from ds5612_pa2.code.pipeline_configs import (
    VALID_CLASSIFIER_TYPES,
    DatasetConfig,
    MLModelConfig,
    get_classifier_config,
)


########################Begin: Do not modify anything below########################
NUM_BOOTSTRAP_TRIALS = 200


class ErrorEstimationAlgos(str, Enum):
    """ErrorEstimationAlgos is an Enum to control which classifiers are used in ML pipeline."""

    TRAIN = "TRAIN"
    TRAIN_VAL = "TRAIN_VAL"
    CV_TRAIN_VAL = "CV_TRAIN_VAL"
    BOOTSTRAP_632 = "BOOTSTRAP_632"
    BOOTSTRAP_OOB = "BOOTSTRAP_OOB"

    # By default, Enums output the "name" (eg DT, NB and KNN)
    # Make it output the value
    def __str__(self) -> str:
        """Customize the display name of the enum variable."""
        return str(self.value)


class ErrorEstimator(BaseModel):
    """ErrorEstimator is a container class to get ML model error estimates."""

    model_config = ConfigDict(extra="forbid")

    algorithm: ErrorEstimationAlgos
    true_err: float = 0.0

    pred_err: float = 0.0
    # Some algos do not give CI bounds. In that case, set to extremal values.
    pred_err_low: float = 0.0
    pred_err_high: float = 1.0


def get_production_score(dataset_config: DatasetConfig, ml_model_config: MLModelConfig) -> float:
    """get_production_score train on entire data and test it on production data."""
    ml_model = ml_model_config.create_classifier()
    ml_model.fit(dataset_config.train_val_test_X, dataset_config.train_val_test_y)
    pred_y = ml_model.predict(dataset_config.production_X)
    return float(accuracy_score(dataset_config.production_y, pred_y))


def display_error_estimates(error_estimates: list[ErrorEstimator]) -> None:
    """display_error_estimates outputs a rich table."""
    table = Table(
        "Estimator",
        "True Error",
        "Predicted Error",
        "CI Low",
        "CI High",
        title="Error Estimates",
    )

    for elem in error_estimates:
        table.add_row(
            elem.algorithm,
            str(elem.true_err),
            str(elem.pred_err),
            str(elem.pred_err_low),
            str(elem.pred_err_high),
        )

    print(table)


########################End: Do not modify anything below########################


def estimate_error(
    dataset_config: DatasetConfig, ml_model: VALID_CLASSIFIER_TYPES, error_estimator: ErrorEstimator
) -> ErrorEstimator:
    """estimate_error takes the dataset and a classifier and estimates the generalization error."""
    match error_estimator.algorithm:
        case ErrorEstimationAlgos.TRAIN:
            '''TRAIN: This approach estimates that the accuracy on the validation data is the same as the production data. So, you have to fit the model on training data and get the accuracy on the test data.'''
            ml_model.fit(dataset_config.train_X, dataset_config.train_y)
            pred_y = ml_model.predict(dataset_config.test_X)
            error_estimator.pred_err = accuracy_score(dataset_config.test_y, pred_y)
        case ErrorEstimationAlgos.TRAIN_VAL:
            '''TRAIN_VAL: This approach estimates that the accuracy on the training+validation data is the same as the production data. So, you have to fit the model on training+validation data and get the accuracy on the test data. The dataset_config already has a variable that pools training and validation data.'''
            ml_model.fit(dataset_config.train_val_test_X, dataset_config.train_val_test_y)
            pred_y = ml_model.predict(dataset_config.test_X)
            error_estimator.pred_err = accuracy_score(dataset_config.test_y, pred_y)
        case ErrorEstimationAlgos.CV_TRAIN_VAL:
            '''CV_TRAIN_VAL: This approach is same as TRAIN_VAL except that you use cross validation. Use the scikit-learn's cross_val_score function (and set scoring=accuracy). By default, scikit-learn uses a 5-fold cross validation and will return the accuracy on each of the five fold. Just compute the mean value of this and use that as the estimate'''
            scores = cross_val_score(ml_model, dataset_config.train_val_test_X, dataset_config.train_val_test_y, scoring="accuracy")
            error_estimator.pred_err = np.mean(scores)
        case ErrorEstimationAlgos.BOOTSTRAP_632:
            '''BOOTSTRAP_632: This approach uses a more complex approach using Bootstrap. You can use the function bootstrap_point632_score from here. There are two key things to be aware. First, set n_splits to NUM_BOOTSTRAP_TRIALS and random_seed to utils.ANSWER_TO_EVERYTHING so that the result is reproducible.'''
            scores = bootstrap_point632_score(
                ml_model, dataset_config.train_val_test_X, dataset_config.train_val_test_y, n_splits=NUM_BOOTSTRAP_TRIALS, random_seed=utils.ANSWER_TO_EVERYTHING
            )
            error_estimator.pred_err = np.mean(scores)
            error_estimator.pred_err_low = np.min(scores)
            error_estimator.pred_err_high = np.max(scores)
        case ErrorEstimationAlgos.BOOTSTRAP_OOB:
            '''BOOTSTRAP_OOB: This is the same as bootstrap_point632_score. But 'oob' as the method to the bootstrap_point632_score function. Both these methods output confidence intervals (see the API documentation of bootstrap_point632_score). So, you can use that code to estimate the 95% confidence intervals'''
            scores = bootstrap_point632_score(
                ml_model, dataset_config.train_val_test_X, dataset_config.train_val_test_y, method="oob", n_splits=NUM_BOOTSTRAP_TRIALS, random_seed=utils.ANSWER_TO_EVERYTHING
            )
            error_estimator.pred_err = np.mean(scores)
            error_estimator.pred_err_low = np.percentile(scores, 2.5)
            error_estimator.pred_err_high = np.percentile(scores, 97.5)
        case _:
            raise ValueError(f"Invalid error estimation algorithm: {error_estimator.algorithm}")

    return error_estimator


##### Do not change anything below
if __name__ == "__main__":
    d = DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    d.load_data()
    ml_model_config = get_classifier_config("DT")
    ml_model = ml_model_config.create_classifier()

    error_estimates = []
    for algo in ErrorEstimationAlgos:
        error_estimator = estimate_error(d, ml_model, ErrorEstimator(algorithm=algo))
        error_estimator.true_err = get_production_score(d, ml_model_config)
        error_estimates.append(error_estimator)
    display_error_estimates(error_estimates)
