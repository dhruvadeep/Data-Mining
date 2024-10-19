# GiG

import pathlib
from typing import Literal

import annotated_types
import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from ds5612_pa2.code import utils
from ds5612_pa2.code.pipeline_configs import (
    DatasetConfig,
    DecisionTreeConfig,
    KNNConfig,
    NaiveBayesConfig,
)
from ds5612_pa2.code.utils import ANSWER_TO_EVERYTHING


def check_field_info(  # noqa: PLR0913
    model_fields: dict[str, FieldInfo],
    field_name: str,
    field_type=None,  # noqa: ANN001
    field_description=None,  # noqa: ANN001
    default=None,  # noqa: ANN001
    ge=None,  # noqa: ANN001
    le=None,  # noqa: ANN001
) -> None:
    assert field_name in model_fields
    if field_type is not None:
        assert model_fields[field_name].annotation == field_type
    if field_description is not None:
        assert model_fields[field_name].description == field_description
    if default is not None:
        assert model_fields[field_name].default == default
    if ge is not None:
        assert annotated_types.Ge(ge=ge) in model_fields[field_name].metadata
    if le is not None:
        assert annotated_types.Le(le=le) in model_fields[field_name].metadata


@pytest.mark.t1pydantic()
def test_dataset_config_specification() -> None:
    # Get the fields of the model
    model_fields = DatasetConfig.model_fields

    check_field_info(
        model_fields=model_fields,
        field_name="file_path",
        field_type=pathlib.Path,
        field_description="Path to the dataset file",
    )
    check_field_info(
        model_fields=model_fields,
        field_name="train_size",
        field_type=float,
        field_description="Proportion of data to use for training",
        default=0.3,
        ge=0.1,
        le=0.5,
    )

    check_field_info(
        model_fields=model_fields,
        field_name="validation_size",
        field_type=float,
        field_description="Proportion of data to use for validation",
        default=0.1,
        ge=0.1,
        le=0.2,
    )

    check_field_info(
        model_fields=model_fields,
        field_name="test_size",
        field_type=float,
        field_description="Proportion of data to use for testing",
        default=0.1,
        ge=0.1,
        le=0.3,
    )

    check_field_info(
        model_fields=model_fields,
        field_name="production_size",
        field_type=float,
        field_description="Proportion of data to use for Production",
        default=0.5,
        ge=0.1,
        le=0.5,
    )


@pytest.mark.t1pydantic()
def test_dataset_config_file_path() -> None:
    d = DatasetConfig(file_path=utils.DATASET_FILE_PATH)
    assert d is not None

    # test with a non existent file
    with pytest.raises(ValidationError):
        DatasetConfig(file_path=pathlib.Path("a non existent file"))


@pytest.mark.t1pydantic()
def test_dataset_config_size_validator() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(
            file_path=utils.DATASET_FILE_PATH,
            train_size=0.2,
            validation_size=0.2,
            test_size=0.2,
            production_size=0.5,
        )


@pytest.mark.t1pydantic()
def test_decision_tree_config_specification() -> None:
    # Get the fields of the model
    model_fields = DecisionTreeConfig.model_fields

    check_field_info(
        model_fields=model_fields,
        field_name="ml_model_type",
        field_type=Literal["decision_tree"],
        default="decision_tree",
    )

    check_field_info(
        model_fields=model_fields,
        field_name="random_state",
        field_type=int,
        default=ANSWER_TO_EVERYTHING,
    )

    check_field_info(
        model_fields=model_fields,
        field_name="criterion",
        field_type=Literal["gini", "entropy", "log_loss"],
        default="gini",
        field_description="Function to measure the quality of a split",
    )

    check_field_info(
        model_fields=model_fields,
        field_name="max_depth",
        field_type=int,
        default=1,
        field_description="Maximum depth of the tree",
        ge=1,
    )

    check_field_info(
        model_fields=model_fields,
        field_name="min_samples_split",
        field_type=int,
        default=2,
        field_description="Minimum number of samples required to split an internal node",
        ge=2,
    )


@pytest.mark.t1pydantic()
def test_naive_bayes_config_specification() -> None:
    # Get the fields of the model
    model_fields = NaiveBayesConfig.model_fields

    check_field_info(
        model_fields=model_fields,
        field_name="ml_model_type",
        field_type=Literal["naive_bayes"],
        default="naive_bayes",
    )

    check_field_info(
        model_fields=model_fields,
        field_name="variant",
        field_type=Literal["gaussian"],
        default="gaussian",
        field_description="Variant of Naive Bayes to use",
    )


@pytest.mark.t1pydantic()
def test_knn_config_specification() -> None:
    # Get the fields of the model
    model_fields = KNNConfig.model_fields

    check_field_info(
        model_fields=model_fields,
        field_name="ml_model_type",
        field_type=Literal["knn"],
        default="knn",
    )

    check_field_info(
        model_fields=model_fields,
        field_name="n_neighbors",
        field_type=int,
        default=5,
        ge=1,
        field_description="Number of neighbors to use",
    )

    check_field_info(
        model_fields=model_fields,
        field_name="weights",
        field_type=Literal["uniform", "distance"],
        default="uniform",
        field_description="Weight function used in prediction",
    )

    check_field_info(
        model_fields=model_fields,
        field_name="p",
        field_type=int,
        default=2,
        field_description="Power parameter for the Minkowski metric",
        ge=1,
        le=3,
    )
