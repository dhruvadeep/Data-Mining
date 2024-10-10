# GiG


from typing import Annotated

import typer

from ds5612_pa2.code import pipeline_configs

'''
inside pipeline_configs
class ValidClassifierNames(str, Enum):
    """ValidClassifierNames is an Enum to control which classifiers are used in ML pipeline."""

    DT = "DecisionTree"
    NB = "NaiveBayes"
    KNN = "KNN"

    # By default, Enums output the "name" (eg DT, NB and KNN)
    # Make it output the value
    def __str__(self) -> str:
        """Customize the display name of the enum variable."""
        return str(self.value)

'''

# Typer has some interesting feature that allows you to
# add some completion options to your shell
# This is useful in a large setting but a distraction for toy scripts
# So disable it.
app = typer.Typer(add_completion=False)
'''
Valid classifiers

1. DecisionTree
2. NaiveBayes
3. KNN
'''


@app.command(name="train", help="Train a classifier.")
def train(
    # input string and valid values are the enum values of ValidClassifierNames
    classifier: Annotated[pipeline_configs.ValidClassifierNames, typer.Option(help="Classifier name.")] = pipeline_configs.ValidClassifierNames.DT
) -> None:
    """Train a classifier."""
    pipeline = pipeline_configs.get_simple_ml_pipeline(classifier)
    pipeline.run_pipeline()

'''
'''
@app.command(name="predict", help="Make a prediction using a trained classifier.")
def predict(
    item: Annotated[list[float], typer.Argument(help="Input values for prediction.")],
    classifier: Annotated[pipeline_configs.ValidClassifierNames, typer.Option(help="Classifier name.")] = pipeline_configs.ValidClassifierNames.DT
) -> None:
    """Predict using the specified classifier and input values."""

    # Use the item directly as it is already a list of floats
    print(pipeline_configs.get_prediction_probabilities(item, classifier))


if __name__ == "__main__":
    app()
