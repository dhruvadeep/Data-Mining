# GiG
import pytest
from typer.testing import CliRunner

from ds5612_pa2.code.task2.typer_demo import app


# Create a test runner
runner = CliRunner()


@pytest.mark.t2typer()
def test_commands_have_help() -> None:
    for cmd in app.registered_commands:
        if cmd.name == "train":
            assert cmd.help == "Train a classifier."
        if cmd.name == "predict":
            assert cmd.help == "Make a prediction using a trained classifier."


@pytest.mark.t2typer()
def test_train_with_default_classifier() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0


@pytest.mark.t2typer()
def test_train_with_specified_classifier() -> None:
    result = runner.invoke(app, ["train", "--classifier", "KNN"])
    assert result.exit_code == 0


@pytest.mark.t2typer()
def test_train_with_invalid_classifier() -> None:
    result = runner.invoke(app, ["train", "--classifier", "INVALID"])
    assert result.exit_code != 0


@pytest.mark.t2typer()
def test_predict_with_default_classifier() -> None:
    result = runner.invoke(
        app, ["predict", "1.0", "2.0", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0"]
    )
    assert result.exit_code == 0


@pytest.mark.t2typer()
def test_predict_with_specified_classifier() -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--classifier",
            "NaiveBayes",
            "1.0",
            "2.0",
            "3.0",
            "4.0",
            "5.0",
            "6.0",
            "7.0",
            "8.0",
            "9.0",
            "10.0",
        ],
    )
    assert result.exit_code == 0


@pytest.mark.t2typer()
def test_predict_with_invalid_classifier() -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--classifier",
            "INVALID",
            "1.0",
            "2.0",
            "3.0",
            "4.0",
            "5.0",
            "6.0",
            "7.0",
            "8.0",
            "9.0",
            "10.0",
        ],
    )
    assert result.exit_code != 0


@pytest.mark.t2typer()
def test_predict_with_no_input_values() -> None:
    result = runner.invoke(app, ["predict"])
    assert result.exit_code != 0


@pytest.mark.t2typer()
def test_predict_with_non_float_input() -> None:
    result = runner.invoke(app, ["predict", "1.0", "not_a_float", "3.0"])
    assert result.exit_code != 0
