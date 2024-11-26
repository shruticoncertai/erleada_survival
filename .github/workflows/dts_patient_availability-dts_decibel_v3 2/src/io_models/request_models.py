from pydantic import BaseModel
from typing import List 

class PAModel_Prediction_Request(BaseModel):
    cancer_indication: str
    train: bool=False
    patient_ids: List[str]
    parquet_path: str
