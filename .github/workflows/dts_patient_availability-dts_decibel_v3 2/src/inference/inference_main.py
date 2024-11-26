import smart_open
import time
import matplotlib.pyplot as plt
import mlflow
import json
import polars as pl
from configs.variable_constants import *
from configs.configs import *
from configs.configurations import *
from configs.cancer_indication_config import tumor_indication_config
from configs.model_tracker_config import model_tracker_config
from rule_engine.rule_engine import IndicationRuleEngine
from utils.db_pool import redshift_engine
from utils.monitoring import generate_charts
from utils.table_details import get_schema_type
from model_builder.model_builder import Model
from utils import utils
from utils.model_selection import ModelSelection
import pandas as pd

logger = utils.setup_logger("server")
from patient_availability_model import update_concept_table

def run_inference_on_input(request_payload, request_id):
    
    #MLFlow Experiment
    try:
        study_name = request_payload['study_name']
        study_id = request_payload['study_id']
        generate_charts(study_id, study_name, redshift_engine)
    except Exception as e:
        logger.error(f"An error occurred to generate charts - {e}")
        logger.error(e, exc_info=True, stack_info=True)
    experiment_name = f"dts_PA_{request_payload['study_name']}_{request_payload['study_id']}"
    mlflow.set_experiment(experiment_name)

    request_payload["req_id"] = request_id
    req_start_time = time.time()
    
    try:
        utils.update_req_status("Inprogress", None, request_payload)
        
        logger.info(f"Inference started - {request_id}")
        update_concept_table()
        mm_configs = MultipleMyelomaConfigurations()
        pansolids_configs = PanSolidsConfigurations()
        mds_configs = MDSConfigurations()
        
        sep = request_payload["sep"] 
        cohort_json = request_payload['cohort_json']
        model_selector = ModelSelection(sep, cohort_json )
        
        df = pl.from_dicts(request_payload['patient_ids'], schema=[PERSON_ID,TUMOR_INDICATION])
        tumor_types = df[TUMOR_INDICATION].unique()
        schema_name = request_payload[SCHEMA_NAME]
        if IS_CLQ:
            if TumorIndicationConstants.MULTIPLE_MYELOMA in tumor_types:
                mm_configs.update_synonyms()
            elif TumorIndicationConstants.MDS_CANCER in tumor_types:
                mds_configs.update_synonyms()
            else:
                pansolids_configs.update_synonyms()
        output_df = pd.DataFrame()
        final_evidence_df = pd.DataFrame()
        logger.info(f"Number of indications identified - {len(tumor_types)}")
        
        mlflow.start_run()
        mlflow.log_param("job_id", request_id)
        
        for tumor_type in tumor_types:
            t_df = df.filter(pl.col(TUMOR_INDICATION) == tumor_type)
            logger.info(f"\n\nIndication - {tumor_type} - DF Size - {t_df.shape}")
            
            if tumor_type not in tumor_indication_config:
                #Unsupported Cancer Indicator/tumor_type
                default_null_probability = { 
                        RES_30: 0.0, 
                        RES_60:0.0, 
                        RES_90: 0.0, 
                        RES_120: 0.0, 
                        RES_150:0.0, 
                        RES_180:0.0, 
                        SOURCE: 'no_data', 
                        SOURCE_DATE: pd.NaT
                }
                t_df = t_df.to_pandas()
                result = t_df[[PERSON_ID]].assign(**default_null_probability)
            else:
                tumor_specific_config = tumor_indication_config.get(tumor_type).copy()
                if tumor_type == TumorIndicationConstants.MULTIPLE_MYELOMA:
                    configurations = mm_configs
                    #Update the model
                    tumor_specific_config[MODEL_NAME] = tumor_specific_config[BASE_MODEL_NAME]
                elif tumor_type == TumorIndicationConstants.MDS_CANCER:
                    configurations = mds_configs
                    tumor_specific_config[MODEL_NAME] = tumor_specific_config[BASE_MODEL_NAME]
                else:
                    configurations = pansolids_configs
                    if model_selector.is_line1_study():
                        #Update the model
                        tumor_specific_config[MODEL_NAME] = tumor_specific_config[LINE_1_MODEL_NAME]
                    else:
                        #Update the model
                        tumor_specific_config[MODEL_NAME] = tumor_specific_config[BASE_MODEL_NAME]
                result, evidence_df = run_inference_for_indication(tumor_type, t_df, tumor_specific_config, schema_name, configurations)
                final_evidence_df = pd.concat([final_evidence_df, evidence_df], ignore_index=True) if not final_evidence_df.empty else evidence_df
            #output_df = output_df.extend(result) if output_df is not None else result
            
            output_df = pd.concat([output_df, result], ignore_index=True) if not output_df.empty else result
            
        mlflow.end_run()
        #Drop duplicates:
        if "person_id" in output_df.columns and "90" in output_df.columns:
            output_df = output_df.sort_values(["person_id", "90"], ascending=False).drop_duplicates(subset=["person_id"], keep="first")
        
        utils.update_req_status("Completed", {"output": output_df.to_json(), "evidence": final_evidence_df.to_json()}, request_payload)
        req_end_time = time.time()
        logger.info("Request Processing completed successfully in "+str(round(req_end_time-req_start_time, 2))+" seconds")
        return output_df 
    
    except Exception as e:
        req_end_time = time.time()
        logger.error(f"An error occurred in inference - {request_id} - {e}")
        logger.error(e, exc_info=True, stack_info=True)
        mlflow.end_run()
        utils.update_req_status("Error", str(e), request_payload)
        logger.info("Request Processing failed. Time consumed: "+str(round(req_end_time-req_start_time, 2))+" seconds")

def run_inference_for_indication(tumor_type, df, tumor_config, schema_name, configurations):
    # Feature Generation
    feature_df, evidence_df = tumor_config[FEATURIZER](df, schema_name, configurations).compute_features(tumor_config[SHORT_CODE])
    
    feature_set = model_tracker_config[tumor_config[MODEL_NAME]]["feature_set"]
    result_set = model_tracker_config[tumor_config[MODEL_NAME]]["result_set"]
    logger.info(f"Features generated for inference - {feature_df.shape}")
    
    chart_features = list()
    try:
        for chart in cancer_specific_constants[tumor_config[GROUPING_ID]].get("charts", []):
            if chart["chart_type"] == "stacked_bar":
                chart_features += chart["feature_set"]
                for i in range(0,len(chart["feature_set"]), 10):
                    try:
                        feature_figure = feature_df[chart["feature_set"][i:i+10]].stack().groupby(level=[1]).value_counts().unstack().plot.bar(rot=0, stacked=True, figsize=(12,5)).get_figure()
                        feature_figure.savefig(tumor_config[SHORT_CODE] + f"_{chart['chart_name']}_{i}.jpg")
                        mlflow.log_artifact(f"{tumor_config[SHORT_CODE]}_{chart['chart_name']}_{i}.jpg")
                    except Exception as e:
                        logger.info(f"An error occurred computing stacked_bar chart")
                        logger.error(e, stack_info=False, exc_info=True)
                        print(feature_df[chart["feature_set"][i:i+10]])
        other_featuers = [i for i in feature_set if i not in chart_features]

        #Generate Histogram Figure on Feature set    
        feature_figure = feature_df[other_featuers].hist(bins=10, figsize=(15,15))[0][0].get_figure()
        feature_figure.savefig(tumor_config[SHORT_CODE] + "_histogram.jpg")
        mlflow.log_artifact(tumor_config[SHORT_CODE] + "_histogram.jpg")
    
        #Boxplots on feature_set
        for i in other_featuers:
            plt.clf()
            try:
                feature_df.boxplot(column=[i]).get_figure().savefig(f"{tumor_config[SHORT_CODE]}_{i}.jpg")
                mlflow.log_artifact(f"{tumor_config[SHORT_CODE]}_{i}.jpg")
            except Exception as e:
                logger.info(f"An error occurred computing boxpolot chart")
                logger.error(e, stack_info=False, exc_info=True)
                print(feature_df[i])
        plt.clf()
    except Exception as e:
        logger.info(f"An error occurred computing stats")
        logger.error(e, stack_info=False, exc_info=True)


    # Get probabilies by rule engine
    i_rule_engine = IndicationRuleEngine(tumor_config[RULE_ENGINE])
    output_df_by_rules, evidence_df_by_rules = i_rule_engine.apply_rules(feature_df, evidence_df)
    output_df_by_rules = output_df_by_rules.drop_duplicates(subset=["person_id"], keep="first")


    # Filter records inferred by rules
    filtered_data = feature_df[~feature_df[PERSON_ID].isin(output_df_by_rules[PERSON_ID])].dropna(subset=feature_set,
                                                                                             inplace=False,
                                                                                             how='all')
    
    evidence_ml_data = evidence_df[~evidence_df[PERSON_ID].isin(output_df_by_rules[PERSON_ID].to_list())]
    evidence_ml_data = evidence_ml_data.loc[~evidence_ml_data["measure"].str.startswith('Criteria', na=False)]

    filtered_data = filtered_data.reset_index()
    logger.info(f"Filtered data for targeted for inference - {filtered_data.shape}")

    default_null_probability = { RES_30: 0.0, RES_60:0.0, RES_90: 0.0, RES_120: 0.0, RES_150:0.0, RES_180:0.0, SOURCE: 'no_data', SOURCE_DATE: pd.NaT}
    
    # Infer Model
    dataset_pd = filtered_data[feature_set]
    model = Model(name = model_tracker_config[tumor_config[MODEL_NAME]][MODEL_NAME])
    if dataset_pd.empty:
        result = pd.DataFrame()
        result = result.assign(**default_null_probability).drop(columns=[SOURCE_DATE])
    else:
        result = model.predict(dataset=dataset_pd)
        result.columns = result.columns.astype(str)
        print("Prediction Result Shape:", result.shape, result.columns)
        result = 1 - result  # Prediction to Probability
        result[SOURCE] = 'ml_prediction'
    

    # Data prevalance
    missing_pd = feature_df[
        (~feature_df[PERSON_ID].isin(filtered_data[PERSON_ID])) &
        (~feature_df[PERSON_ID].isin(output_df_by_rules[PERSON_ID]))
        ][[PERSON_ID]]
    logger.info(f"Patients with null data {missing_pd.shape}")
    missing_pd = missing_pd.reset_index()
    missing_pd = missing_pd.assign(**default_null_probability)
    missing_pd = missing_pd[[PERSON_ID, SOURCE, SOURCE_DATE] + result_set]

    # Post processing
    result = pd.concat([result, filtered_data], axis=1)
    if 'latest_date' in result.columns:
        result = result.rename(columns={'latest_date': SOURCE_DATE})
    else:
        result[SOURCE_DATE] = pd.NaT
    result = result[[PERSON_ID, SOURCE, SOURCE_DATE] + result_set]
    result = pd.concat([result, missing_pd], ignore_index=True)
    logger.info(f"Result + Missing Shape: {result.shape}")
    result = pd.concat([result, output_df_by_rules], ignore_index=True)
    logger.info(f"Result + Missing Shape+latest_pr_date:{result.shape}")
    result = result.drop_duplicates(subset=['person_id'], keep='last')
    logger.info(f"Final  - {result.shape}")

    #Generate Histogram Figure on Result set
    result_figure = result[result_set].hist(bins=10)[0][0].get_figure()
    result_figure.savefig(f"{tumor_config[SHORT_CODE]}_resultset_histogram.jpg")
    mlflow.log_artifact(f"{tumor_config[SHORT_CODE]}_resultset_histogram.jpg")
    
    evidence_final = pd.concat([evidence_df_by_rules, evidence_ml_data], ignore_index=True)
    
    return result, evidence_df_by_rules
