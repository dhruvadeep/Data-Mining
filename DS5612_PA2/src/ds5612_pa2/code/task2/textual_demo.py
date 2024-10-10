# GiG

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, ProgressBar, Select, TextArea

from ds5612_pa2.code import pipeline_configs


CLASSIFIER_OPTIONS = [(clf.value, clf.name) for clf in pipeline_configs.ValidClassifierNames]


def predict(item_str: str, classifier: str) -> tuple[float, float]:
    """Predict using the specified classifier and input values."""
    item = [float(elem) for elem in item_str.split()]
    result = pipeline_configs.get_prediction_probabilities(item, classifier)
    return result[0], result[1]


class TextualClassifier(App):
    """TextualClassifier is a simple demo for a ML classifier."""
    

    CSS = """
    Horizontal {
        height: 100%;
    }
    #input-panel, #output-panel {
        width: 50%;
        height: 100%;
        padding: 1;
    }
    #input-panel {
        border-right: solid green;
    }
    TextArea {
        height: 1fr;
    }
    ProgressBar {
        margin: 1 0;
    }
    #predict {
        margin-top: 1;
    }
    Label {
        padding-top: 1;
        padding-bottom: 1;
    }
    Select {
        width: 60;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose creates the UI."""
        yield Header()
        # Fill in the body
        with Horizontal():
            with Vertical(id="input-panel"):
                yield Label("Classifier:")
                yield Select(options=CLASSIFIER_OPTIONS, id="classifier")
                yield Label("Input Text:")
                yield TextArea(id="input-text")
                yield Button("Predict", id="predict")
            with Vertical(id="output-panel"):
                yield Label("Positive Score:", id="positive_score")
                yield ProgressBar(id="positive", total=1.0)
                yield Label("Negative Score:", id="negative_score")
                yield ProgressBar(id="negative", total=1.0)
        yield Footer()

    # on select change have label value as classifier name
    @on(Select.Changed, "#classifier")
    def on_classifier_change(self) -> None:
        """on_classifier_change updates the classifier label."""
        classifier = self.query_one("#classifier").value
        self.query_one("#classifier").label = classifier


    def on_mount(self) -> None:
        """on_mount can be used to set initial values."""
        # Set default values for progress bars
        self.query_one("#positive").update(progress=0.0)
        self.query_one("#negative").update(progress=0.0)


    @on(Button.Pressed, "#predict")
    def on_predict(self) -> None:
        """on_predict is called when the Predict button is pressed."""
        # Get the value from the classifier and features
        # do validation and then call display_prediction_results
        classifier = self.query_one("#classifier").value
        input_text = self.query_one("#input-text").text

        # validate input_text
        if not classifier or not input_text:
            self.notify("Classifier and input text must be provided!", title="Error")
            return
        
        features = input_text.split()
        if len(features) != 10:
            self.notify("Input text must have 10 values!", title="Error")
            return
        
        self.display_prediction_results(classifier, input_text)

    # Do not change this function.
    def display_prediction_results(self, classifier: str, text: str) -> None:
        """display_prediction_results calls the ML pipeline and shows o/p in UI."""

        # Given the classifier and features as text,
        # pass it to predict function and get positive and negative probabilities
        # then update positive and negative ; and positive_score, negative_score
        # do not forget to handle errors.
        try:
            positive, negative = predict(text, classifier)

            # Update the progress bars
            self.query_one("#positive").update(progress=positive)
            self.query_one("#negative").update(progress=negative)

            # Update the labels
            self.query_one("#positive_score").update(f"Positive Score: {positive}")
            self.query_one("#negative_score").update(f"Negative Score: {negative}")

        except Exception as e:
            self.notify(f"Error during prediction: {str(e)}", title="Prediction Error")
            return


if __name__ == "__main__":
    app = TextualClassifier()
    app.run()
