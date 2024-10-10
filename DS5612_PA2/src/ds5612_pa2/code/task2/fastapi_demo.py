# GiG


from io import StringIO

from fastapi import FastAPI, HTTPException, UploadFile, status
from pydantic import BaseModel

from ds5612_pa2.code import pipeline_configs


app = FastAPI()


'''
Create three Pydantic BaseModel classes with the names PredictRequest, PredictionResponse and DetailedPredictionResponse. The request will be an instance of PredictRequest. To make things interesting, we will do a minor demonstration of API versioning. We will implement two versions of the API. The V1 will return PredictionResponse while the V2 will return DetailedPredictionResponse.

The PredictRequest should have two fields: classifier that can take only values from pipeline_configs.ValidClassifierNames and features that is a list of float.

The PredictionResponse should have two fields: predicted_class that is an integer and ml_model_version that is a string with default value of "V1". This should allow the clients to understand that they are processing the output of V1 model.

The PredictionResponse should extend PredictionResponse and have a new field: probabilities that is a tuple with two floats (ie the probability for positive and negative class). Additionally, it should set the ml_model_version as a string with default value of "V2". This should allow the clients to understand that they are processing the output of V2 model
'''

class PredictRequest(BaseModel):
    """PredictRequest is a simple request body asking for a classifier and features."""
    classifier: pipeline_configs.ValidClassifierNames
    features: list[float]


class PredictionResponse(BaseModel):
    """PredictionResponse is the response type for API V1."""
    predicted_class: int
    ml_model_version: str = "V1"


class DetailedPredictionResponse(PredictionResponse):
    """DetailedPredictionResponse is the response type for API V2."""
    probabilities: tuple[float, float]
    ml_model_version: str = "V2"


@app.post("/v1/predict")
def predict_v1(request: PredictRequest) -> PredictionResponse:
    # pass the right params from PredictRequest
    #  and create PredictionResponse appropriately
    prediction = pipeline_configs.get_prediction_class(
        request.features,
        request.classifier
    )
    return PredictionResponse(
        predicted_class=prediction
    )


@app.post("/v2/predict")
def predict_v2(request: PredictRequest) -> DetailedPredictionResponse:
    # pass the right params from PredictRequest
    #  and create DetailedPredictionResponse appropriately
    ml_pipeline = pipeline_configs.get_simple_ml_pipeline(request.classifier)
    ml_pipeline.train()

    # pass the right params from PredictRequest
    probabilities = ml_pipeline.get_prediction_probabilities(request.features)
    prediction = ml_pipeline.get_prediction_class(request.features)
    return DetailedPredictionResponse(
        predicted_class=prediction,
        probabilities=probabilities
    )


@app.post("/batch_predict/")
def batch_predict(input_file: UploadFile) -> list[DetailedPredictionResponse]:
    # Hard code classifier as decision tree
    ml_pipeline = pipeline_configs.get_simple_ml_pipeline(pipeline_configs.ValidClassifierNames.DT)
    ml_pipeline.train()

    # Change the logic from below.
    # Parse the input_file, get predictions using the code in predict_v2 as a sample
    #  return the list of DetailedPredictionResponse
    output = []
    file_content = input_file.file.read().decode("utf-8")
    lines = file_content.strip().splitlines()

    for liine_num, line in enumerate(lines, start=1):
        try:
            features = [float(f) for f in line.strip().split()]
            if len(features) == 0:
                raise ValueError("No features provided in the line.")
            
            probabilities = ml_pipeline.get_prediction_probabilities(features)
            prediction = ml_pipeline.get_prediction_class(features)
            output.append(DetailedPredictionResponse(
                predicted_class=prediction,
                probabilities=probabilities
            ))
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Error in line {liine_num}: {str(e)}"
            )
    


    return output


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
