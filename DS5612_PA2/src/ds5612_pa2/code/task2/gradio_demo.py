# GiG

import gradio as gr

from ds5612_pa2.code import pipeline_configs


CLASSIFIER_OPTIONS = [clf.value for clf in pipeline_configs.ValidClassifierNames]

'''

Product Specification

You can see a simple UI of the code in the screenshot below. The UI should have a DropDown with label "Classifier" for the classifier (use the CLASSIFIER_OPTIONS variable for creating the values) and a textbox with label "Features".

There will be two panels on the RHS for the output. If the inputs are correct and there is no error, you should output the class (Negative/Positive) and progress bar to show the respective probabilities. If there is an error, the error will be displayed in the error textbox.

The "business" logic is written by me. So you can focus on the UI. It should not take more than 5 lines or so for the UI. You also have to change the code a bit to handle the error scenario which should another couple of lines.

You can run the application using the command below. This will start a web service rendering that can be accessed (by default) here. Remember to stop the server before running the test suite.

    rye run gradio_task

You have to use the following Gradio components: Dropdown, Textbox and Label. The Label class can accept class probabilities that my code will already give in the right format.

'''

def predict(classifier: str, item_str: str) -> tuple[dict[str, float], str]:
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


# Create Gradio interface
####################################Only change here
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(CLASSIFIER_OPTIONS, label="Classifier"),
        gr.Textbox(label="Features"),
    ],
    outputs=[
        gr.Label(label="Prediction"),
        gr.Textbox(label="Error"),
    ],
    title="ML Classifier",
    description="Choose a classifier and enter features to get prediction probabilities.",
)
####################################Only change here

if __name__ == "__main__":
    # Launch the app
    iface.launch()
