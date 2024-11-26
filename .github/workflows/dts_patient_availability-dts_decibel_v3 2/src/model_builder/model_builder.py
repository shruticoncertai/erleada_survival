import numpy as np
import os
import smart_open
from joblib import dump, load
from sklearn.model_selection import train_test_split
from xgbse import (
    XGBSEDebiasedBCE,
)
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)
from xgbse.metrics import concordance_index


from patient_availability_model import *
from configs.configs import *
from utils.utils import setup_logger

MODEL_BASEPATH = models_base_path

from pathlib import Path
Path(MODEL_BASEPATH).mkdir(parents=True, exist_ok=True)

OUTPUT_TIME_WINDOW_RANGE = [30,60,90,120]

logger = setup_logger("model_builder")

class Model():
    def __init__(self, name) -> None:
        logger.debug("Model Object Initialized")
        self.name = name
        self.windows = OUTPUT_TIME_WINDOW_RANGE
        self.model = None
        self.request_id = None

    def save_model(self):
        logger.debug("Saving the Model")
        if self.model is None:
            raise KeyError(f"Invalid Model Name, {self.name} ")
        
        model = self.model
        model_path_name = MODEL_BASEPATH + self.name + ".joblib"
        dump(model, model_path_name)
        logger.info("Model saved to " + model_path_name)

        s3_path = f"s3://{s3_schema_bucket}/{s3_prefix}/models/{self.name}.joblib"
        with smart_open.open(s3_path, "wb") as s3_writer:
            with open(model_path_name, 'rb') as f:
                data = f.read()
            s3_writer.write(data)
        
        logger.info("Model saved to S3 location, " + s3_path)

        return None

    def load_model(self):
        logger.debug("Loading the Model")
        model_path_name = MODEL_BASEPATH + self.name + ".joblib"
        s3_path = f"s3://{s3_schema_bucket}/{s3_prefix}/models/{self.name}.joblib"
        
        if not os.path.exists(model_path_name):
            logger.info("Downloading the model from S3 location, " + s3_path)
            with smart_open.open(s3_path, "rb") as s3_reader:
                data = s3_reader.read()
                with open(model_path_name, 'wb') as f:
                    f.write(data)
            logger.info("Downloading the model completed from S3 location, " + s3_path)
        model = load(model_path_name)
        self.model = model
        return None

    def train_and_build_model(self, feature_x, feature_y, split_ratio):

        logger.debug("Model Train Begins")        
        x_train, x_test, y_train, y_test = train_test_split(feature_x,feature_y, test_size=split_ratio,random_state=123)

        PARAMS_XGB_AFT = {
            'objective': 'survival:aft',
            'eval_metric': 'aft-nloglik',
            'aft_loss_distribution': 'normal',
            'aft_loss_distribution_scale': 1.0,
            'tree_method': 'hist', 
            'learning_rate': 0.005, 
            'max_depth': 16, 
            'booster':'dart',
            'subsample':0.8,
            'min_child_weight': 30,
            'colsample_bynode':0.8
        }
        
        logger.info("Model Training started")
        xgbse_model = XGBSEDebiasedBCE(PARAMS_XGB_AFT)
        xgbse_model.fit(x_train,y_train,time_bins=np.array(self.windows))
        logger.info("Model Training Completed")

        self.model = xgbse_model

        #Predict and Validate
        train_cindex_list = []
        test_cindex_list = []
        train_cindex_list.append(concordance_index(y_train, xgbse_model.predict(x_train)))
        test_cindex_list.append(concordance_index(y_test, xgbse_model.predict(x_test)))

        logger.info(f'Train c-index list : {train_cindex_list}')
        logger.info(f'Test c-index list: {test_cindex_list}')
        logger.info(f'Train c-index average : {np.mean(train_cindex_list)}')
        logger.info(f'Test c-index average: {np.mean(test_cindex_list)}')



        return None

    def predict(self, dataset):
        logger.debug(f"Model Inference initiated - {dataset.shape}")
        if self.model: 
            result = self.model.predict(dataset)
        else:
            self.load_model()
            result = self.model.predict(dataset)
        logger.info("Inference completed")
        return result
        
