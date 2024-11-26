import datetime
import json
import logging
import os
import uuid

import mlflow
import pandas as pd
import smart_open
import xlsxwriter
import uvicorn

from pathlib import Path

from fastapi import FastAPI, BackgroundTasks
from typing import Dict, List, Tuple
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)
from inference.inference_main import run_inference_on_input
from model_builder.model_builder import Model
from patient_availability_model import generate_features, generate_features_sc_v2, to_pandas 
from configs.configs import *
from utils import utils


app = FastAPI()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_BASEPATH = models_base_path
RESULTS_BASEPATH = predict_results_basepath
Path(RESULTS_BASEPATH).mkdir(parents=True, exist_ok=True)
Path(MODEL_BASEPATH).mkdir(parents=True, exist_ok=True)

#Default Model Name
app.state.model_name = DEFAULT_MM_MODEL_NAME
with open("models_tracker.json", 'w') as f:
    f.write(json.dumps({'model': DEFAULT_MM_MODEL_NAME}))

logger = utils.setup_logger("server")

@app.get("/ping")
async def root():
    logger.info("API / Requested")
    return {"message": "Welcome to Patient Availability Model Prediction Server"}


#request_payload to Model
@app.post("/predict_patient_availability")
async def predict_model(request_payload: Dict, bg_tasks: BackgroundTasks):
    logger.info("API [/predict_patient_availability] Requested")
    train = request_payload['training']

    if not train:
        request_id = str(uuid.uuid4())
        bg_tasks.add_task(run_inference_on_input, request_payload, request_id)
        return {"Status": "Inference is in-progress", "request_id": request_id}
    else:
        raise NotImplementedError

@app.get("/inference_results")
async def get_inference_results(req_id: str):
    try:
        s3_path = f"s3://{s3_schema_bucket}/{s3_prefix}/inference_output/{req_id}.json"
        with smart_open.open(s3_path, "r") as s3_reader:
            result = json.loads(s3_reader.read())
    except Exception as e:
        result = {"Status": "Failed", "Error Message": str(e)}
        logger.error(f"An error inferring the model - {req_id}")
        logger.error(e, exc_info=True, stack_info=True)
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8220)






