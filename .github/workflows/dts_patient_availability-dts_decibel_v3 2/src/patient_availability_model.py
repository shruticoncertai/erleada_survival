import datetime
import json
import logging
import numpy as np
import polars as pl
import random
import sys
import traceback
import xlsxwriter
import yaml


from sklearn.model_selection import train_test_split
from xgbse import (
    XGBSEDebiasedBCE,
)
from xgbse.converters import (
    convert_data_to_xgb_format,
    convert_to_structured
)
from xgbse.metrics import concordance_index

#Data Loader Packages
from configs.configs import *
from configs.constants import *
from configs.concept_code_mapper import *
from configs.configurations import *
from utils import data_loader
from utils import table_details as tables
from utils import utils
from unit_converter.convert import *
#from PA_GE_test.rule_execution import rules_check
from datetime import timedelta

with open('unit_converter/keys_labs.yaml','r') as file:
    lab_kb = yaml.safe_load(file)

lab_kb = {c['key']:c for c in lab_kb}

with open('unit_converter/units.yaml','r') as file:
    unite_conv_map = yaml.safe_load(file)

unite_conv_map = {c['base']:c for c in unite_conv_map}



## ------------------ CONSTANTS & other Parameters --------------------- ##

LOG_FILENAME = "patient_availability_FE.log"
INPUT_TIME_WINDOW_RANGE = [10,30,90]
OUTPUT_TIME_WINDOW_RANGE = [30,60,90,120]

## ---------------------------------------------------------------------------- ##

## ----------------------- Function Definition -------------------------------- ##

def get_logger(log_level, log_filename=None):
    """
    Method to create logger that can log to console and to log_file if provided
    
    ARGUMENTS:
    ------------
    log_level: logging.log_level: Log Level to print the logs
    log_filename: str: Log Filename to log the results
    
    RETURNS:
    ------------
    rootLogger: logging.Logger
    """
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(log_level)
    
    if log_filename is not None:
        fileHandler = logging.FileHandler("{0}/{1}.log".format('.', log_filename))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    
    return rootLogger

def convert_unit(df,schema_type=None):
    """
    Method to convert the units to standard units specific to the key. 
    Lab tests for unit standardization is using the knowledge Base
    
    ARGUMENTS:
    ----------
    df: Polars.DataFrame: To convert unit standardization, provided with 'key' column
    
    RETURNS:
    ----------
    df: Polars.DataFrame: Dataframe with standard units for the lab tests that are defined in key column.
    
    """
    lab_test = df['key'].unique().drop_nulls()
    df = df.with_columns(
            original_value_as_number = pl.col("value_as_number"),
            original_unit_name = pl.col("measurement_unit_source_name")
        )
    for key in lab_test:
        if key not in lab_kb.keys():
            raise KeyError(f'Lab test {key} not registered')
        std_unit = lab_kb[key]['attributes']['units']
        unit_keys = list(unite_conv_map[std_unit]['convert'].keys())
        
        #update the 'value_as_number'
        df = df.with_columns(
                pl.struct(['value_as_number','measurement_unit_source_name','key'])
                          .apply(lambda x: eval(unite_conv_map[std_unit]['convert'][x['measurement_unit_source_name']].split('.')[-1])(x['value_as_number'])\
                                 if (x['key']==key) & (x['measurement_unit_source_name'] in (unit_keys))\
                                 else x['value_as_number']                                                      
                                ).alias('value_as_number')
        )
        
        #update units
        df = df.with_columns(
                        pl.when((pl.col('key')==key) & ((pl.col('measurement_unit_source_name')==std_unit) | (pl.col('measurement_unit_source_name').is_in(unite_conv_map[std_unit]['convert'].keys()))))
                .then(pl.lit(std_unit))
                .when(pl.col('key')==key)
                .then(None)
                .otherwise(pl.col('measurement_unit_source_name'))
                .alias('measurement_unit_source_name'))\
        .drop_nulls('measurement_unit_source_name')
    return df

def get_diagnosis_codes(cancer_indicator):
    """ Method to get ICD10 Cancer codes
        ------------------
        ARGS:
        cancer_indicator: str: cancer indicator used to fectch the diagnois codes
        ------------------
        Returns:
        diagnosis_codes: List(str): List of Diagnosis codes associated with the cancer_indicator
    """
    icd10_filename = "configs/cancer_dx_icd10.json"
    with open(icd10_filename, 'r') as f:
        icd10_code = json.loads(f.read())
    
    diagnosis_codes = [ x.upper() for x in icd10_code[cancer_indicator]]
    return diagnosis_codes

def get_patient_diagnosis(diagnosis_codes, parquet_location=s3_parquet_base_path,dx_year=start_year,schema_type=None):
    """ Fetch the patient diagnosis details 
        ---------
        ARGS:
        diagnosis_codes: list(str): diagnosis codes
        update_activity: bool: Whether to update the activity information of the patient
        parquet_location: str: S3 Base path for loading the parquet files
        ---------
        Returns:
        patient_id_df: polars.Dataframe: Patient Condition Start date

    """

    conition_test_lazy = data_loader.get_table_df("condition", parquet_base_path=parquet_location, schema=schema_type)
    condition_column_names = tables.get_column_name("condition", schema=schema_type)
    if dx_year is None:
        pat_id_df = conition_test_lazy.filter(
                             pl.col(condition_column_names["diagnosis_code_standard_code"]).is_in(diagnosis_codes)
                        )
    else:
        pat_id_df = conition_test_lazy.filter(
                       ( pl.col(condition_column_names["diagnosis_code_standard_code"]).is_in(diagnosis_codes)) & \
                        (pl.col(condition_column_names["diagnosis_date"]).dt.year() >= dx_year)
                    )
    pat_id_df = pat_id_df.unique(subset=[
                                            condition_column_names["chai_patient_id"], 
                                            condition_column_names["diagnosis_date"], 
                                            condition_column_names["diagnosis_code_standard_code"]])\
                    .collect()
    
    #Safety check from null-data
    pat_id_df = pat_id_df.with_columns(
            pl.col(condition_column_names["diagnosis_type_concept_id"]).cast(pl.Int64),
            pl.col(condition_column_names["diagnosis_date"]).cast(pl.Date)
    
    )
    pat_id_df = pat_id_df.rename({
                    condition_column_names["chai_patient_id"]: "person_id",
                    condition_column_names["diagnosis_date"]: "condition_start_date",
                    condition_column_names["diagnosis_code_standard_code"]: "concept_code",
                    condition_column_names["diagnosis_code_standard_name"]: "concept_name",
                    condition_column_names["diagnosis_type_concept_id"]: "condition_type_concept_id"
                })

    return pat_id_df

def update_concept_table():
    global concept_df
    logger.info("Updating the concept table")
    concept_df = data_loader.get_concept_lazy_df().collect()
    logger.info("Updation of concept table completed")
    logger.info("Concept DF:" + str(concept_df.columns))

def get_biomarker_data(pat_ids_list, bm_list, bm_concept_names=[], parquet_path=s3_parquet_base_path, schema_type=None):
    #bm_list : concept_code
    biomarker_df = data_loader.get_table_df("biomarker", parquet_base_path=parquet_path, schema=schema_type)
    biomarker_column_names = tables.get_column_name("biomarker", schema=schema_type)
    global concept_df
    bm_names = concept_df.filter(pl.col("concept_code").is_in(bm_list)).unique(["concept_name"]).select(["concept_code", "concept_name"])
    bm_names_all = bm_names["concept_name"].unique().to_list() + bm_concept_names
    
    biomarker_data = biomarker_df.filter(
        (pl.col(biomarker_column_names["biomarker_name_name"]).str.to_lowercase().is_in(bm_names_all)) &
        (pl.col("person_id").is_in(pat_ids_list))
    ).collect()
    
    print("BioMarker Data reading is completed", biomarker_data.shape)
    biomarker_data = biomarker_data.join(bm_names, left_on=biomarker_column_names["biomarker_name_name"], right_on = "concept_name", how="left")
        
    if "value_as_concept_id" in biomarker_df.columns:
        biomarker_data = biomarker_data.with_columns(
                    pl.col('value_as_concept_id').cast(pl.Int64)
        )
        biomarker_data = biomarker_data.join(concept_df.select(["concept_id", "concept_name"]),
                         left_on="value_as_concept_id",
                          right_on = "concept_id",
                          how="inner"
                     )
        biomarker_data = biomarker_data.rename({"concept_name_right": "value"})
    elif biomarker_column_names["value"] in biomarker_df.columns:
        biomarker_data = biomarker_data.rename({biomarker_column_names["value"]: "value"})

    biomarker_data = biomarker_data.rename({
                        biomarker_column_names["chai_patient_id"]: "person_id",
                        biomarker_column_names["report_date"]: "measurement_date",
                        biomarker_column_names["biomarker_name_name"]: "biomarker_name"
                    })
    
    return biomarker_data
        
def get_tumor_exam_results(patient_ids, parquet_path=s3_parquet_base_path,schema_type=None):
    """
    Method to fetch tumor grade exam results from parquet files for specific patient_ids
    
    ARGUMENTS:
    -----------
    patient_ids: List of Patient Ids to fetch the Tumor Exam results
    update_activity: bool: An option to save activity date from tumor_exam table. Optional to reduce multi-scan
    parquet_path: str: S3 Base path for loading the parquet files

    RETURNS:
    ----------
    tumor_exam_df: Polars.DataFrame: Dataframe with Tumor Grade Exam results
    """
    patient_col_names = tables.get_column_name("tumor_exam")
    tumor_exam_lazy_df = data_loader.get_table_df("tumor_exam", parquet_base_path=parquet_path, schema=schema_type)
    tumor_exam_df = tumor_exam_lazy_df.filter(
            (pl.col(patient_col_names["chai_patient_id"]).is_in(patient_ids)) &\
            (pl.col(patient_col_names["exam_date"]).dt.year() >= start_year)
        )\
        .collect()
    
    return tumor_exam_df


def get_patient_results(patient_ids, lab_test_codes=[], lab_test_names=[],parquet_path=s3_parquet_base_path, schema_type=None):
    """ 
    Method to fetch the Lab test results for specific patient_ids and only test results of test_codes and test_names
    
    ARGUMENTS:
    ----------
    patient_ids: List of Patient Ids to fetch the Lab test results
    lab_test_codes: List(str): Lab test codes to filter
    lab_test_names: List(Str): Lab test names to filter
    parquet_path: str: S3 Base path for loading the parquet files
    
    RETURNS:
    -----------
    patient_test_df: Polars.DataFrame: Dataframe with Lab test results and key
    """
    global concept_df
    patient_test_lazy = data_loader.get_table_df( "patient_test", join_concept=False, parquet_base_path=parquet_path, schema=schema_type)
    patient_test_column_names = tables.get_column_name("patient_test", schema=schema_type)
    print("Started Patient Test")
    
    pt_test_cols = [
            patient_test_column_names['chai_patient_id'], 
            patient_test_column_names['test_date'], 
            patient_test_column_names['test_value_numeric'],  
            patient_test_column_names['test_unit_source_name'],
            patient_test_column_names["test_value_name"],
            patient_test_column_names['source'],
            patient_test_column_names['measurement_doc_id']
    ]
    if 'measurement_concept_id' in patient_test_lazy.columns:
        pt_test_cols.append('measurement_concept_id')
    else:
        pt_test_cols.append(patient_test_column_names['test_name_standard_code'])
        pt_test_cols.append(patient_test_column_names['test_name_standard_name'])

    patient_test_df = patient_test_lazy.select(pt_test_cols)\
                           .filter(
                                        (pl.col(patient_test_column_names['chai_patient_id']).is_in(patient_ids)) & \
                                        (pl.col(patient_test_column_names['test_date']).dt.year() >= start_year)
                            )
    
    if 'measurement_concept_id' in patient_test_lazy.columns:
        lab_concept_df = concept_df.select("concept_id", "concept_code", "concept_name")\
                                    .filter(
                                        (pl.col("concept_code").is_in( lab_test_codes) )|
                                            (pl.col("concept_name").str.to_lowercase().is_in( lab_test_names ))
                                    )
    
        concept_id = lab_concept_df.select("concept_id").to_series().to_list() 
        print("Total Lab Tests to search:", len(concept_id))
        patient_test_df = patient_test_df.filter(
                            pl.col("measurement_concept_id").is_in(concept_id)
                        ).collect()
    else:
        patient_test_df = patient_test_df.filter(
                           ( pl.col(patient_test_column_names["test_name_standard_code"]).is_in(lab_test_codes)) |
                           ( pl.col(patient_test_column_names["test_name_standard_name"]).str.to_lowercase().is_in(lab_test_names))
                        ).collect()
    
    print("Patient tests:", patient_test_df.shape)
    patient_test_df = patient_test_df.rename({
                        patient_test_column_names['chai_patient_id']: 'person_id',
                        patient_test_column_names['test_date']: 'measurement_date',
                        patient_test_column_names['test_value_numeric']: 'value_as_number',
                        patient_test_column_names['test_unit_source_name']: 'measurement_unit_source_name',
                    })
    
    if 'measurement_concept_id' in patient_test_lazy.columns:  
        patient_test_df = patient_test_df\
                                    .join(lab_concept_df, left_on="measurement_concept_id", right_on="concept_id", how="inner")
    
    patient_test_df = patient_test_df.rename({
        patient_test_column_names["test_name_standard_code"]: "concept_code",
        patient_test_column_names["test_name_standard_name"]: "concept_name",
        patient_test_column_names["test_value_name"]: "test_value_name"
    })
    
    ### Tests that are Evidences
    evidence_test_codes = list()
    for pt_test in patient_test_as_evidence:
        evidence_test_codes.extend(patient_test_as_evidence[pt_test])
    
    ######
    #####################
    #Add key column for unit standarization 
    patient_test_df = patient_test_df.with_columns(
                               pl.when(pl.col('concept_code').is_in(lab_dict['m_protein_in_serum']))
                                    .then(pl.lit('m_protein_in_serum'))
                                    .when(pl.col('concept_code').is_in(lab_dict['m_protein_in_urine']))
                                    .then(pl.lit('m_protein_in_urine'))
                                    .when(pl.col('concept_code').is_in(lab_dict['ca']))
                                    .then(pl.lit('ca'))
                                    .when(pl.col('concept_code').is_in(lab_dict['serum_free_light']))
                                    .then(pl.lit('serum_free_light'))
                                    .when(pl.col('concept_code').is_in(lab_dict['hemoglobin_in_blood']))
                                    .then(pl.lit('hemoglobin_in_blood'))
                                    .when(pl.col('concept_code').is_in(lab_dict['neutrophils_count']))
                                    .then(pl.lit('neutrophils_count'))
                                    .when(pl.col('concept_code').is_in(lab_dict['lymphocytes_count']))
                                    .then(pl.lit('lymphocytes_count'))
                                    .when(pl.col('concept_code').is_in(lab_dict['platelets']))
                                    .then(pl.lit('platelets'))
                                    .when(pl.col('concept_code').is_in(lab_dict['na']))
                                    .then(pl.lit('na'))
                                    .when(pl.col('concept_code').is_in(lab_dict['mg']))
                                    .then(pl.lit('mg'))
                                    .when(pl.col('concept_code').is_in(lab_dict['cl']))
                                    .then(pl.lit('cl'))
                                    .when(pl.col('concept_code').is_in(lab_dict['phos']))
                                    .then(pl.lit('phos'))
                                    .when(pl.col('concept_code').is_in(lab_dict['k']))
                                    .then(pl.lit('k') )
                                    .alias('key'),
                            pl.col('measurement_unit_source_name').str.to_lowercase(),
                            #Preventing the test that are evidences and has Nulls as `value_as_number`
                            pl.when(pl.col('concept_code').is_in(evidence_test_codes) & pl.col('value_as_number').is_null())
                                .then(pl.lit(float('NaN')))
                                .otherwise(pl.col('value_as_number'))
                                .alias('value_as_number')
                    ).with_columns(
                        pl.when((pl.col('concept_code').is_in(lab_dict['hr']+lab_dict['dbp']+lab_dict['ecog'])) & (pl.col('measurement_unit_source_name').is_null()))
                        .then(pl.lit('valid'))
                        .otherwise(pl.col('measurement_unit_source_name'))
                        .alias('measurement_unit_source_name')
                    )
    
    patient_test_df = patient_test_df.with_columns(
                        pl.col('measurement_date').cast(pl.Date)
                ).filter(pl.col("measurement_date") < datetime.datetime.today())
   
    
    return patient_test_df


def get_test_results(patient_test_df, test_code, test_names = [], schema_type=None, default_key=None):
    """
    Method to filter the test results of test_code from the entire list and unit conversion and de-duplication
   
    Arguments:
    ---------
    patient_test_df: Polars.DataFrame: Lab Test Results
    test_code: List(str): List of test codes to fetch the details

    Returns:
    ---------
    test_results: Polars.DataFrame
    """
    test_results = patient_test_df.filter(pl.col("concept_code").is_in(test_code)|
                                            (pl.col("concept_name").str.to_lowercase().is_in( test_names )) )
    if default_key is not None:
        test_results = test_results.with_columns(
                pl.when(pl.col('key').is_null())
                    .then(pl.lit(default_key))
                    .otherwise(pl.col('key'))
                    .alias('key')
        )
    test_results = convert_unit(test_results)
    
    #Takes mean of same lab test with multiple results on a same day
    test_results = test_results.groupby(by=['person_id','measurement_date', "concept_code"])\
                            .agg(
                                    pl.col('value_as_number').mean(), 
                                    pl.col("original_value_as_number").mean(), 
                                    pl.col("original_unit_name"),
                                    pl.col("measurement_unit_source_name"),
                                    pl.col("source"),
                                    pl.col("measurement_doc_id")
                                    )
    
    test_results = test_results.with_columns(
                        pl.when(pl.col("original_unit_name").list.unique().list.lengths() == 1 )
                            .then(pl.col("original_unit_name").list.get(0))
                            .otherwise(pl.col("measurement_unit_source_name").list.get(0))
                            .alias("original_unit_name"),
                        pl.col("source").list.get(0),
                        pl.col("measurement_doc_id").list.get(0),
                        pl.when(pl.col("original_unit_name").list.unique().list.lengths() == 1 )
                            .then(pl.col("original_value_as_number"))
                            .otherwise(pl.col("value_as_number"))
                            .alias("original_value_as_number"),
                    ).drop("measurement_unit_source_name")
    
    test_results = test_results\
        .sort(['person_id','measurement_date','value_as_number'],descending=False)\
        .with_columns(
            pl.col('value_as_number').cast(pl.Float64)
        )
    
    return test_results

def calculate_nadir(lab_test_df):
    """
    Calcuates the Nadir value for a lab_test across the patient history till that date
    @Args: 
        lab_test_df: Polars.DataFrame:  ["person_id", "concept_code", "measurement_date", "value_as_number"] as columns
        
    @Returns:
        lab_test_df: Polars.DataFrame:  ["person_id", "concept_code", "measurement_date", "value_as_number", "nadir"]
    """
    def get_lowest_from_history(x): 
        x=list(x)
        total_elements = len(x)
        result = [ x[0] ]
        prev_min = x[0] 
        for i in range(1, total_elements,1):
            if x[i]<prev_min:
                prev_min = x[i]
            result.append(prev_min)
        return result
        
    if lab_test_df.shape[0]>0:
        
        lab_test_df = lab_test_df.sort(["person_id","concept_code", "measurement_date"])\
            .groupby(by=["person_id","concept_code"])\
            .agg(
                pl.col("measurement_date"), 
                pl.col("value_as_number"),
                pl.col("source"),
                pl.col("measurement_doc_id")            
            )\
            .with_columns(
                nadir = pl.col("value_as_number").apply(get_lowest_from_history)
            ).explode(["measurement_date", "value_as_number", "nadir", "source", "measurement_doc_id"])
    else:
        lab_test_df = lab_test_df.with_columns(
                    nadir = pl.col("value_as_number")
        )
    
    return lab_test_df

def get_imaging_data(pat_ids,  image_list, image_group_map, parquet_path=s3_parquet_base_path,schema_type=None):
    ds_column_names = tables.get_column_name("imaging", schema=schema_type)
    imaging_column_names = tables.get_column_name("imaging", schema = schema_type)
#    imaging_lazy_df = data_loader.get_table_df("imaging", parquet_base_path=parquet_path,schema=schema_type)
    imaging_df = data_loader.get_table_df("imaging", parquet_base_path=parquet_path, schema=schema_type).collect()

    if not imaging_df.is_empty():
        imaging_df = imaging_df.filter(
            pl.col(ds_column_names["chai_patient_id"]).is_in(pat_ids)
        )
    else:
        if isinstance(pat_ids[0], str):
            imaging_df = imaging_df.with_columns(
                pl.col((ds_column_names["chai_patient_id"])).cast(pl.Utf8)
            )
            
        imaging_df = imaging_df.with_columns(
            pl.col(ds_column_names["radiology_name_standard_name"]).cast(pl.Utf8),
            pl.col(ds_column_names["radiology_visitdate"]).cast(pl.Datetime(time_unit='ns'))
        )
    image_list = [i.lower() for i in image_list]
    if schema_type == "omop":
        imaging_df = imaging_df.filter(
            pl.col('src_type').is_in(['radiology', 'g2_pat_order'])
        ) 
    imaging_df = imaging_df.filter(
pl.col(imaging_column_names["radiology_name_standard_name"]).str.to_lowercase().is_in(image_list)
            )

    """
    schema = {'date': pl.Datetime(time_unit='ns', time_zone=None),
              'surgery_src_type': pl.Utf8,
              'procedure_concept_id': pl.Float64,
              'source': pl.Utf8,
              'person_id': pl.Utf8,
              'surgery_primary_concepts_name': pl.Utf8,
              'surgery_primary_concepts_code': pl.Utf8,
              'surgery_primary_concepts_vocab': pl.Utf8,
              'surgery_observation_id': pl.Utf8,
              'surgery_doc_id': pl.Utf8,
              'procedure_datetime': pl.Utf8,
              'procedure_standard_name': pl.Utf8,
              'curation_indicator': pl.Utf8}
    imaging_df = pl.DataFrame(schema)
    """
#    if 'src_type' in imaging_column_names:
#        imaging_lazy_df = imaging_lazy_df.filter(
#                            pl.col('src_type').is_in(['radiology', 'g2_pat_order'])
#                        )
#    imaging_df = imagining_df.filter(
#                    pl.col(imaging_column_names["chai_patient_id"]).is_in(pat_ids)
#                ).collect()
#    imaging_df = imaging_df.filter(
#                            pl.col(imaging_column_names["radiology_name_standard_name"]).is_in(image_list)
#                       )
    imaging_df = imaging_df.drop_nulls(subset = [
                    imaging_column_names["chai_patient_id"],
                    imaging_column_names["radiology_visitdate"],
                    imaging_column_names["radiology_name_standard_name"]
                ])
    
    if not imaging_df.is_empty():
        imaging_df = imaging_df.with_columns(
            pl.col(imaging_column_names["radiology_name_standard_name"]).str.to_lowercase().map_dict(
                {i.lower(): k for i, k in image_group_map.items()}, 
                default=pl.first()
            ).alias(imaging_column_names["radiology_name_standard_name"]),
                    
    )
         #pl.col(imaging_column_names["radiology_name_standard_name"]).apply(lambda x:image_group_map[x] if (x in image_group_map.keys()) else x).alias(imaging_column_names["radiology_name_standard_name"])
    
    imaging_df = imaging_df.rename({
                                imaging_column_names["chai_patient_id"]: 'person_id',
                                imaging_column_names["radiology_visitdate"]: 'procedure_date',
                                imaging_column_names["radiology_name_standard_name"]: 'concept_name'
                        })
    return imaging_df

def imaging_feature_extraction(imaging_data, criteria_df):
    imaging_column_names = tables.get_column_name("imaging")
    imaging_data = imaging_data.join(
                            criteria_df.select(['person_id','condition_start_date','random_point']),
                            left_on= imaging_column_names["chai_patient_id"], 
                            right_on='person_id',
                            how='inner')
    imaging_data = imaging_data.filter(
                        (pl.col( imaging_column_names['radiology_visitdate']) >= pl.col("condition_start_date")) & 
                        (
                            (
                                pl.col(imaging_column_names['radiology_visitdate']) - pl.col("condition_start_date")
                            ).dt.days() <= pl.col("random_point")
                        )
                    )                 

    imaging_pats = list(set(imaging_data['person_id'].unique().to_list()))                                
    pat_imaging_feat_df = pl.DataFrame(criteria_df['person_id'].unique())
    pat_imaging_feat_df = pat_imaging_feat_df.with_columns(
                                pl.col('person_id').is_in(imaging_pats).alias('imaging').cast(pl.Boolean)
                          )
    return pat_imaging_feat_df

def get_staging_data(pat_ids_list, parquet_path=s3_parquet_base_path,schema_type=None):
    """
    Method to fetch the stage details for the patients
    
    Arguments:
    -----------
    pat_ids_list: List: List of Patient Ids to track
    parquet_path: str: S3 Base path for loading the parquet files

    RETURNS:
    ----------
    medication_df: Polars.DataFrame:  Medication details
    
    """
    staging_column_names = tables.get_column_name("staging", schema=schema_type)
    staging_lazy_df = data_loader.get_table_df("staging", parquet_base_path=parquet_path, schema=schema_type)
    
    #Check DF is empty:
    if staging_lazy_df.limit(1).collect().is_empty():
        staging_df =  staging_lazy_df.collect()
        if isinstance(pat_ids_list[0], str):
            staging_df = staging_df.with_columns(
                pl.col(staging_column_names["chai_patient_id"]).cast(pl.Utf8),
            )
        possible_cols = [
                            "stage_group_concept_id", 
                            "tstage_concept_id", 
                            "mstage_concept_id", 
                            "nstage_concept_id",
                            staging_column_names["stage_group_standard_name"], 
                            staging_column_names["tstage_standard_name"], 
                            staging_column_names["mstage_standard_name"], 
                            staging_column_names["nstage_standard_name"]  
        ]
        for col in possible_cols:
            if col in staging_df.columns:
                staging_df = staging_df.with_columns(
                    pl.col(col).cast(pl.Utf8)
                )
            
        
    else:
        staging_df = staging_lazy_df.filter(
                        pl.col(staging_column_names["chai_patient_id"]).is_in(pat_ids_list)
                ).collect()
    global concept_df
    if "stage_group_concept_id" in staging_df.columns:
        staging_df = staging_df.join(
                        concept_df.select(["concept_id", "concept_name"]),
                        left_on = "stage_group_concept_id",
                        right_on = "concept_id",
                        how="left"
                    ).rename({"concept_name": "stage_group_standard_name"})
    elif(staging_column_names["stage_group_standard_name"] in staging_df):
        staging_df = staging_df.rename({staging_column_names["stage_group_standard_name"]:"stage_group_standard_name" })
        
    if "tstage_concept_id" in staging_df.columns:
        staging_df = staging_df.join(
                        concept_df.select(["concept_id", "concept_name"]),
                        left_on = "tstage_concept_id",
                        right_on = "concept_id",
                        how="left"
                    ).rename({"concept_name": "tstage_standard_name"})
    elif(staging_column_names["tstage_standard_name"] in staging_df):
        staging_df = staging_df.rename({staging_column_names["tstage_standard_name"]:"tstage_standard_name" })
        
    if "mstage_concept_id" in staging_df.columns:
        staging_df = staging_df.join(
                        concept_df.select(["concept_id", "concept_name"]),
                        left_on = "mstage_concept_id",
                        right_on = "concept_id",
                        how="left"
                    ).rename({"concept_name": "mstage_standard_name"})
    elif(staging_column_names["mstage_standard_name"] in staging_df):
        staging_df = staging_df.rename({staging_column_names["mstage_standard_name"]:"mstage_standard_name" })
        
    if "nstage_concept_id" in staging_df.columns:
        staging_df = staging_df.join(
                        concept_df.select(["concept_id", "concept_name"]),
                        left_on = "nstage_concept_id",
                        right_on = "concept_id",
                        how="left"
                    ).rename({"concept_name": "nstage_standard_name"})
    elif(staging_column_names["nstage_standard_name"] in staging_df):
        staging_df = staging_df.rename({staging_column_names["nstage_standard_name"]:"nstage_standard_name" })
    
    staging_df =staging_df.rename({
                    staging_column_names['chai_patient_id']: 'person_id',
                    staging_column_names['stage_date']: 'stage_date',
                }).with_columns(
                    pl.col('stage_date').cast(pl.Date)
                )
    
    return staging_df


def get_disease_status_data(pat_ids_list, parquet_path=s3_parquet_base_path,schema_type=None):
    ds_column_names = tables.get_column_name("disease_status", schema=schema_type)
    global concept_df
    ds_df = data_loader.get_table_df("disease_status", parquet_base_path=parquet_path, schema=schema_type).collect()
    if not ds_df.is_empty():
        ds_df = ds_df.filter(
                    pl.col(ds_column_names["chai_patient_id"]).is_in(pat_ids_list)
            )
    else:
        if len(ds_df.columns) == 0:
            return None # No Data
        
        if isinstance(pat_ids_list[0], str):
            ds_df = ds_df.with_columns(
                pl.col((ds_column_names["chai_patient_id"])).cast(pl.Utf8)
            )
        ds_df = ds_df.with_columns(
            pl.col(ds_column_names["assessment_value_standard_name"]).cast(pl.Utf8)
        )
    if "assessment_name_concept_id" in ds_df.columns:
        ds_df = ds_df.join(
                    concept_df.select(["concept_id", "concept_name"]),
                    left_on="assessment_name_concept_id",
                    right_on="concept_id",
                    how = "left"
                ).rename({"concept_name": "assessment_name_standard_name"})
    elif ds_column_names["assessment_name_standard_name"] in ds_df.columns:
        ds_df = ds_df.rename({ds_column_names["assessment_name_standard_name"]: "assessment_name_standard_name" })
        
    if "assessment_value_concept_id" in ds_df.columns:
        ds_df = ds_df.join(
                    concept_df.select(["concept_id", "concept_name"]),
                    left_on="assessment_value_concept_id",
                    right_on="concept_id",
                    how = "left"
                ).rename({"concept_name": "assessment_value_standard_name"})
    elif ds_column_names["assessment_value_standard_name"] in ds_df.columns:
        ds_df = ds_df.rename({ds_column_names["assessment_value_standard_name"]: "assessment_value_standard_name" })

    #Standardizing the `assessment_value_standard_name` to support NLP values
    ds_df = ds_df.with_columns(
        pl.when(pl.col("assessment_value_standard_name")=="complete_response")
            .then(pl.lit("Complete therapeutic response"))
            .when(pl.col("assessment_value_standard_name")=="partial_response")
            .then(pl.lit("Partial therapeutic response"))
            .when(pl.col("assessment_value_standard_name")=="stable_disease")
            .then(pl.lit("Stable"))
            .when(pl.col("assessment_value_standard_name").is_in(["progressive_disease", "progression"]))
            .then(pl.lit("Tumor progression"))
            .otherwise(pl.col("assessment_value_standard_name")).alias("assessment_value_standard_name")
    )
    ds_df = ds_df.rename({
                ds_column_names['chai_patient_id']: 'person_id',
                ds_column_names['assessment_date']: 'assessment_date'
            }).with_columns(
                pl.col("assessment_name_standard_name").cast(pl.Utf8),
                pl.col("assessment_value_standard_name").cast(pl.Utf8)
            )
    if (str(ds_df.schema["assessment_date"]) == "Utf8"):
        ds_df = ds_df.with_columns(
                pl.col("assessment_date").str.to_datetime("%Y-%m-%d")
        )
    if "source" not in ds_df.columns:
        ds_df = ds_df.with_columns(
                source = pl.lit(None).cast(pl.Utf8)
            )
    else:
        ds_df = ds_df.with_columns(
                pl.col("source").cast(pl.Utf8)
            )
    if "disease_status_doc_id" not in ds_df.columns:
        ds_df = ds_df.with_columns(
                disease_status_doc_id = pl.lit(None).cast(pl.Utf8)
            )
    else:
         ds_df = ds_df.with_columns(
                pl.col("disease_status_doc_id").cast(pl.Utf8)
            )
    if isinstance(pat_ids_list[0], int):
        ds_df = ds_df.with_columns(
                pl.col("person_id").cast(pl.Int64)
            )

    # Standardizing the `assessment_value_standard_name` to support NLP values
    ds_df = ds_df.with_columns(
        pl.when(pl.col("assessment_value_standard_name") == "complete_response")
        .then(pl.lit("Complete therapeutic response"))
        .when(pl.col("assessment_value_standard_name") == "partial_response")
        .then(pl.lit("Partial therapeutic response"))
        .when(pl.col("assessment_value_standard_name") == "stable_disease")
        .then(pl.lit("Stable"))
        .when(pl.col("assessment_value_standard_name") == "progressive_disease")
        .then(pl.lit("Tumor progression"))
        .otherwise(pl.col("assessment_value_standard_name")).alias("assessment_value_standard_name")
    )

    return ds_df

def get_surgery_details(pat_ids_list, surgery_list, parquet_path=s3_parquet_base_path,schema_type=None):
    """
    Method to fetch the surgeries undergone for the patients
    
    Arguments:
    ---------
    pat_ids_list: List: List of Patient Ids to track
    surgery_list: List: List of Surgeries undergone
    parquet_path: str: S3 Base path for loading the parquet files
    
    Returns:
    --------
    surgery_df:  Polars.DataFrame: Surgery Details
    """
    surgery_column_names = tables.get_column_name("surgery")
    surgery_lazy_df = data_loader.get_table_df("surgery", parquet_base_path=parquet_path)
    
    surgery_df = surgery_lazy_df.filter(
                    pl.col(surgery_column_names["chai_patient_id"]).is_in(pat_ids_list) &
                        pl.col("concept_name").is_in(surgery_list)
                ).collect()
    surgery_df = surgery_df.drop_nulls(subset=[surgery_column_names['surgery_date']])
    surgery_df = surgery_df.rename({
                    surgery_column_names["surgery_date"]: "procedure_date",
                    surgery_column_names["chai_patient_id"]: "person_id"
    })
    
    return surgery_df
    
    
    
def get_medication_details(pat_ids_list, medicines_list, parquet_path=s3_parquet_base_path,schema_type=None):
    """
    Method to fetch the medication prescription details for the patients and medicine details
    
    Arguments:
    -----------
    pat_ids_list: List: List of Patient Ids to track
    medicines_list: List : List of Medicines to filter
    update_activity: bool: An option to save activity date from tumor_exam table. Optional to reduce multi-scan
    parquet_path: str: S3 Base path for loading the parquet files

    RETURNS:
    ----------
    medication_df: Polars.DataFrame:  Medication details
    
    """
    def is_mm_med(med_value):
        """
        Method to update the medication value to one of the medicines_list
        
        Arguments:
        -----------
        med_value: Medication Value to check
        
        RETURNS:
        -----------
        med_value: Medicine Vlaue from the list if medicine value matches else the original name
        
        """
        for med in medicines_list:
            if med in med_value:
                return med
        return med_value
    
    
    medication_col_names = tables.get_column_name("medication", schema=schema_type)
    medication_lazy_df = data_loader.get_table_df("medication", parquet_base_path=parquet_path, schema=schema_type)
    
    #Check df is empty:
    if medication_lazy_df.limit(1).collect().is_empty():
        medication_df = medication_lazy_df.collect()
        if isinstance(pat_ids_list[0], str):
            medication_df = medication_df.with_columns(
                pl.col(medication_col_names["chai_patient_id"]).cast(pl.Utf8),
            )
        medication_df = medication_df.with_columns(
                pl.col(medication_col_names['med_start_date']).cast(pl.Date),
                pl.col(medication_col_names['med_end_date']).cast(pl.Date),
                pl.col(medication_col_names["med_generic_name_standard_name"]).cast(pl.Utf8),
                pl.col(medication_col_names["med_generic_name_standard_code"]).cast(pl.Utf8),
        )
    else:
        medication_df = medication_lazy_df.filter(
                        pl.col(medication_col_names["chai_patient_id"]).is_in(pat_ids_list)
                    ).drop_nulls(subset=[
                            medication_col_names['chai_patient_id'],
                            medication_col_names['med_start_date'],
                            medication_col_names['med_generic_name_standard_name']]
                    ).unique()\
                    .collect()
                    #.filter(pl.col(medication_col_names['med_start_date']) <= pl.col(medication_col_names['med_end_date']))\
    if not medication_df.is_empty():
        medication_df = medication_df.with_columns(
                pl.col(medication_col_names['med_generic_name_standard_name']).apply(is_mm_med)
            )\
            .filter(pl.col(medication_col_names['med_generic_name_standard_name']).is_in(medicines_list))

    medication_df = medication_df.rename({
                        medication_col_names["chai_patient_id"]: "person_id",
                        medication_col_names["med_start_date"]: "drug_exposure_start_date",
                        medication_col_names["med_end_date"]: "drug_exposure_end_date",
                        medication_col_names["med_generic_name_standard_name"]: "concept_name",
                        medication_col_names["med_generic_name_standard_code"]: "concept_code"
                    })
    return medication_df
    
def get_demographics_details(pat_ids_list, parquet_path=s3_parquet_base_path):
    """
    Method to fetch the Patient Demographics details
    
    ARGUMENTS:
    -----------
    pat_ids_list: List: List of Patients to fetch the demographics details
    update_activity: bool: An option to save activity date from tumor_exam table. Optional to reduce multi-scan
    parquet_path: str: S3 Base path for loading the parquet files
    
    RETURNS:
    ----------
    patient_df: Polars.DataFrame: DataFrame object with Patient Demographics
    
    """
    global last_activity_data
    patient_column_names = tables.get_column_name("patient")
    patient_lazy_df = data_loader.get_table_df("patient", parquet_base_path=parquet_path)
    patient_df = patient_lazy_df.filter(
                    pl.col(patient_column_names["chai_patient_id"]).is_in(pat_ids_list)
                ).collect()
    

    patient_df = patient_df.with_columns(
        pl.when(pl.col(patient_column_names['gender']) == 'MALE')
            .then(pl.lit(True))
            .when(pl.col(patient_column_names['gender']) == 'FEMALE')
            .then(pl.lit(False))
            .otherwise(pl.lit(None))
            .cast(pl.Boolean),
        #pl.col(patient_column_names['gender']).apply(lambda x:True if x=='MALE' else False),
        pl.col('source_age').cast(pl.Int64)
    ) 
    patient_df = patient_df.drop("gender_concept_id")
    #Rename columns
    
    return patient_df


def get_latest_activity_date(target_tables, pat_ids, parquet_path=s3_parquet_base_path,schema_type=None):
    """
    Method to get the last updated date from the target tables for th targeted Patients
    
    ARGUMENTS:
    ----------
    target_tables: Tables to target the fetch the activity details
    pat_ids: List of Patient Ids
    parquet_path: str: S3 Base path for loading the parquet files
    
    RETURNS:
    ---------
    last_medical_activity: Polars.DataFrame: Latest Medical Activity recorded from hte targeted tables
    
    """
    
    def get_dod_details(patients_ids, parquet_path):
        """
        Temporary Method to get the DoD details for the selected Patient_ids from Demographics
        """
        patient_column_names = tables.get_column_name("patient")
        patient_lazy_df = data_loader.get_table_df("patient", parquet_base_path=parquet_path)
        patient_df = patient_lazy_df.filter(
                        pl.col(patient_column_names["chai_patient_id"]).is_in(patients_ids)
                    ).collect()

        dod_df = patient_df.select(patient_column_names["chai_patient_id"], patient_column_names["date_of_death"])\
                        .drop_nulls(subset=[patient_column_names["date_of_death"]])\
                            .filter(pl.col(patient_column_names["date_of_death"]).dt.year() >= 1950)
        
        return dod_df

    
    last_activity_data = {} # Empty Initialization
    
    activity_df_list = list()
    for activity in target_tables:
        if activity in last_activity_data:
            print("Skipping Activity Details of table:", activity)
            activity_df_list.append(last_activity_data[activity])
        else:
            print("Loading activity details:", activity)
            try: 
                column_info = tables.get_column_name(activity)
            except KeyError:
                print("Table,{} is not registered".format(activity))
                continue
            data_lazy_df = data_loader.get_table_df(activity, parquet_base_path=parquet_path)
            last_activity_data[activity] = data_lazy_df.filter(pl.col(column_info['chai_patient_id']).is_in(pat_ids)).groupby(column_info['chai_patient_id']).agg(pl.col(column_info[target_tables[activity]]).max().alias("activity_date")).collect()
            activity_df_list.append(last_activity_data[activity])
    activity_df = pl.concat(activity_df_list)
        
    activity_df = activity_df.sort(['person_id', 'activity_date'], descending=True)\
                                    .unique(['person_id'], keep="first")
    

    if "date_of_death" in last_activity_data:
        patient_column_names = tables.get_column_name("patient")
        dod_df = last_activity_data["date_of_death"].drop_nulls(subset=[patient_column_names["date_of_death"]])\
                        .filter(pl.col(patient_column_names["date_of_death"]).dt.year() >= 1950)
    else:
        patient_column_names = tables.get_column_name("patient")
        dod_df = get_demographics_details(pat_ids, parquet_path=parquet_path)
        dod_df = dod_df.select(patient_column_names["chai_patient_id"], patient_column_names["date_of_death"])\
                        .drop_nulls(subset=[patient_column_names["date_of_death"]])\
                            .filter(pl.col(patient_column_names["date_of_death"]).dt.year() >= 1950)
        
    last_medical_activity = activity_df.join(dod_df, on="person_id", how='left')
    
    last_medical_activity = last_medical_activity\
            .with_columns(
                pl.min_horizontal('date_of_death','activity_date')\
                    .alias('last_activity_date').cast(pl.Date)
            ).drop(['date_of_death', 'activity_date'])

    
    return last_medical_activity

def get_criteria_date(pat_list, test_results, threshold, threshold_ratio=0.25):
    def rule_check(test_values,thresold, threshold_ratio=0.25):
        criteria_flag = [False]
        progression_flag = [False]
        min_val = test_values[0]
        nadir = [min_val]
        for i in range(1,len(test_values)):
            if min_val>test_values[i]:
                min_val=test_values[i]
            nadir.append(min_val)
            flag=False
            if (((test_values[i]-min_val)>=thresold) and (((test_values[i]-min_val)/(min_val+0.0001))>=threshold_ratio)):
                    flag =True
            criteria_flag.append(flag)
            progression_flag.append(flag and criteria_flag[-2])
        #return {'criteria_flag':progression_flag,'nadir':nadir}
        return progression_flag
    
    if test_results.is_empty():
        criteria_df = test_results.with_columns(
                        criteria_flag = pl.lit(None).cast(pl.Boolean)
                    )
    else:
        criteria_df = test_results.sort(['person_id','concept_code','measurement_date'])\
                                    .groupby(['person_id','concept_code'])\
                                    .agg(
                                        pl.col('value_as_number'),
                                        pl.col('measurement_date')
                                    ).filter(
                                        pl.col('value_as_number').list.lengths() > 1
                                    ).with_columns(
                                        criteria_flag = pl.col('value_as_number').apply(lambda x:rule_check(x,threshold), skip_nulls=False) 
                                                                            #TODO: threshold ratio not included
                                    ).explode(
                                        ['value_as_number','measurement_date','criteria_flag']
                                    ).filter(pl.col('criteria_flag')==True)\
                                    .sort(['person_id','measurement_date'])\
                                    .unique(subset='person_id',keep='first')
    result = pl.DataFrame({"person_id": pat_list}).join(criteria_df, on="person_id", how="left")
    #TODO: Rename measurement_date
    return result.select(["person_id", "measurement_date"]) 

def get_progression_list(pat_ids, serum_results, urine_results, flc_results, con_plasmacytoma_df ):
    total_pats = len(pat_ids)
    progression_result = pl.DataFrame(
                                        schema=[
                                                    ("person_id", serum_results.schema["person_id"]), 
                                                    ("progression_date", serum_results.schema["measurement_date"]) 
                                        ])
    
    
    while len(pat_ids) > 0:
        print("Total Patients to process", len(pat_ids))
        mprotein_serum_criteria = get_criteria_date(pat_ids, serum_results, 0.5)
        mprotein_serum_criteria = mprotein_serum_criteria.rename({"measurement_date": "criteria1_date"})

        mprotein_urine_criteria = get_criteria_date(pat_ids, urine_results, 200)
        mprotein_urine_criteria = mprotein_urine_criteria.rename({"measurement_date": "criteria2_date"})

        flc_criteria = get_criteria_date(pat_ids, flc_results, 10)
        flc_criteria = flc_criteria.rename({"measurement_date": "criteria3_date"})
        
        plasmacytoma_criteria = con_plasmacytoma_df.unique(subset=['person_id'],keep='first')\
                                    .rename({"condition_start_date": "criteria4_date"})
        
        progression_date = pl.DataFrame({"person_id": pat_ids})\
                            .join(mprotein_serum_criteria, on="person_id", how="left")\
                            .join(mprotein_urine_criteria, on="person_id", how="left")\
                            .join(flc_criteria, on="person_id",  how="left")\
                            .join(plasmacytoma_criteria, on="person_id", how="left")\
                            .with_columns(
                                progression_date = pl.min_horizontal("criteria1_date", "criteria2_date", "criteria3_date", "criteria4_date").cast(pl.Date)
                            )
        
        progression_date = progression_date.filter(pl.col('progression_date').is_not_null()).select(["person_id", "progression_date"])
        pat_ids = progression_date["person_id"].unique().to_list()
        
        #Update test_results
        serum_results = serum_results.join(progression_date, on="person_id")\
                        .filter(pl.col("measurement_date")>pl.col("progression_date"))\
                        .drop("progression_date")

        urine_results = urine_results.join(progression_date, on="person_id")\
                        .filter(pl.col("measurement_date")>pl.col("progression_date"))\
                        .drop("progression_date")
        
        flc_results = flc_results.join(progression_date, on="person_id")\
                        .filter(pl.col("measurement_date")>pl.col("progression_date"))\
                        .drop("progression_date")
        
        con_plasmacytoma_df = con_plasmacytoma_df.join(progression_date, on="person_id")\
                                .filter(pl.col("condition_start_date") > pl.col("progression_date"))\
                                .drop("progression_date")
        
        progression_result = pl.concat([progression_result, progression_date])
    
    return progression_result

def get_criteria_results_rb(pat_id_df, serum_results, urine_results, free_light_chain_results, con_plasmacytoma_df,  parquet_path=s3_parquet_base_path, progression_dates=None, schema_type=None):
    """
    Method to generate the criteria related data frame using m-protein in serum, urine tests, calcium, Free-Light Chain reaction tests
    
    ARGUMENTS:
    ----------
    pat_id_df: cohort group
    serum_results: m-protein in serum test results for MM Patients
    urine_results: m-protein in urine test results for MM Patients
    free_light_chain_results: Free Light chain Reaction Test results for MM Patients
    parquet_path: str: S3 Base path for loading the parquet files
    progression_dates: Patients with Progression dates met by criteria
    schema_type: srt: Schema type of the data
    
    cal_results: Calcium test results for MM Patients: REMOVED

    RETURNS:
    --------
    criteria_df: DataFrame Object()
    """
    def rule_check_new(test_values, nadir, threshold, threshold_ratio=0.25):
        test_points = len(test_values)
        criteria_flag = []
        progression_flag = []
        for i in range(test_points):
            flag = False
            test_value = test_values[i]
            nadir_value = nadir[i]
            if (((test_value-nadir_value)>=threshold) and (((test_value-nadir_value)/(nadir_value+0.0001))>=threshold_ratio)):
                flag =True
            criteria_flag.append(flag)
            progression_flag.append(flag and criteria_flag[-2])

        return progression_flag
    
    criteria_df = pl.DataFrame(pat_id_df['person_id'].unique())
    
    if progression_dates is not None:
        patients_list = progression_dates["person_id"].unique().to_list() #Patients list who met criteria atleast once
        
        #filter other patients' lab results
        serum_results = serum_results.filter(pl.col("person_id").is_in(patients_list))
        urine_results = urine_results.filter(pl.col("person_id").is_in(patients_list))
        free_light_chain_results = free_light_chain_results.filter(pl.col("person_id").is_in(patients_list))
        
        pat_single_progression_date = progression_dates.filter(pl.col("progression_date").list.lengths() == 1)
        pat_multiprogression_dates = progression_dates.filter(pl.col("progression_date").list.lengths() > 1).with_columns(pl.col("progression_date").list.get(-2))
        
        serum_results = pl.concat([
                            serum_results.filter(pl.col("person_id").is_in(pat_single_progression_date["person_id"].to_list())),
                            serum_results.join(pat_multiprogression_dates, on="person_id").filter(pl.col("measurement_date") > pl.col("progression_date")).drop("progression_date"),
        ])
        
        
        urine_results = pl.concat([
                            urine_results.filter(pl.col("person_id").is_in(pat_single_progression_date["person_id"].to_list())),
                            urine_results.join(pat_multiprogression_dates, on="person_id").filter(pl.col("measurement_date") > pl.col("progression_date")).drop("progression_date"),
        ])
        
        free_light_chain_results = pl.concat([
                            free_light_chain_results.filter(pl.col("person_id").is_in(pat_single_progression_date["person_id"].to_list())),
                            free_light_chain_results.join(pat_multiprogression_dates, on="person_id").filter(pl.col("measurement_date") > pl.col("progression_date")).drop("progression_date"),
        ])
        
        
    serum_results = calculate_nadir(serum_results)  
    urine_results = calculate_nadir(urine_results)
    free_light_chain_results = calculate_nadir(free_light_chain_results)
    
    if serum_results.is_empty():
        criteria_serum_results_status = serum_results.with_columns(
                                    criteria_flag = pl.lit(None).cast(pl.Boolean)
                                )
    else:
        criteria_serum_results_status = serum_results.sort(['person_id','measurement_date'],descending=False)\
                                    .groupby(['person_id','concept_code'])\
                                    .agg(
                                        pl.col('value_as_number'),
                                        pl.col('measurement_date'),
                                        pl.col('nadir'),
                                        pl.col('source'),
                                        pl.col('measurement_doc_id')
                                    ).filter(
                                        pl.col('value_as_number').list.lengths() > 1
                                    )
        criteria_serum_results_status = criteria_serum_results_status\
                                .with_columns(
                                    criteria_flag = pl.struct(['value_as_number','nadir']).apply(lambda x: rule_check_new(x["value_as_number"],x["nadir"],0.5), return_dtype=pl.List(pl.Boolean))
                                )\
                                .explode(['value_as_number','measurement_date','criteria_flag', 'nadir','source', 'measurement_doc_id'])

    
    criteria_serum_results = criteria_serum_results_status.join(pat_id_df,on='person_id',how='left')\
                                    .filter(
                                                (pl.col('criteria_flag')==True)
                                    )\
                                    .sort(['person_id','concept_code','measurement_date'], descending=True)\
                                    .with_columns(
                                          latest_pr_date = pl.col('measurement_date').max().over('person_id')
                                    )\
                                    .unique(subset='person_id',keep='first')
    serum_trend =  criteria_serum_results.sort(["person_id", "concept_code", "measurement_date"], descending=True)\
                    .unique(["person_id", "concept_code"], keep="first")\
                    .with_columns(trend = pl.col("value_as_number")-pl.col("nadir"),
                                 rate = (pl.col("value_as_number")-pl.col("nadir"))/pl.col("nadir")*100)
    evidence_serum_trend = serum_trend.select(["person_id", "measurement_date", "trend"])\
                                        .with_columns(
                                             doc_source = pl.lit("") ,
                                             description = pl.lit("Value Difference from Low Point to Latest Lab Test"), 
                                             measure = pl.lit("Criteria:mprotein_serum:Absolute Trend"),
                                             units = pl.lit("g/dl")
                                         ).rename({"measurement_date": "assessment_date", "trend": "value"})
    evidence_serum_pct = serum_trend.select(["person_id", "measurement_date", "rate"])\
                                        .with_columns(
                                             doc_source = pl.lit("") ,
                                             description = pl.lit("Percentage change from Low Point to Latest Lab test"),
                                             measure = pl.lit("Criteria:mprotein_serum:Increase Rate"),
                                             units = pl.lit("%")
                                         ).rename({"measurement_date": "assessment_date", "rate": "value"})
    
    evidence_criteria_serum = criteria_serum_results_status.join(
                                     criteria_serum_results, 
                                    on= ["person_id", "concept_code"]
                                ).filter(pl.col("measurement_date") <= pl.col("measurement_date_right")) #Filter Old Records
    if not evidence_criteria_serum.is_empty():
        evidence_criteria_serum = pl.concat(
                                pl.collect_all([ x.lazy().top_k(2, by='measurement_date') for x in evidence_criteria_serum.partition_by(['person_id', 'concept_code'])])
                            )
    evidence_criteria_serum = evidence_criteria_serum.select(["person_id", "measurement_date", "value_as_number","source", "measurement_doc_id"])\
                            .with_columns(
                                             #doc_source = pl.lit("Lab"),
                                             doc_source = pl.concat_str(
                                                [
                                                    pl.col("source") ,  
                                                    pl.when(pl.col('measurement_doc_id').is_null())
                                                        .then(pl.lit("NONE"))
                                                        .otherwise(pl.col('measurement_doc_id'))
                                                ], separator=":"),
                                             measure = pl.lit("Criteria:mprotein_serum"),
                                             units = pl.lit("g/dl")
                                         )\
                            .drop(["source", "measurement_doc_id"])\
                            .rename({"measurement_date": "assessment_date", "value_as_number": "value"})

    
    #evidence_criteria_serum = criteria_serum_results.filter((pl.col('criteria_flag')==True))\
    #    .select(["person_id", "measurement_date", "value_as_number"])\
    #    .with_columns(
    #                     doc_source = pl.lit("Lab"),
    #                     measure = pl.lit("Criteria:mprotein_serum"),
    #                     units = pl.lit("g/dl")
    #                 )\
    #    .rename({"measurement_date": "assessment_date", "value_as_number": "value"})
    
    serum_nadir = serum_results.join(criteria_serum_results, how="semi", on=["person_id", "concept_code"])\
                .filter(
                            (pl.col('nadir') == pl.col('value_as_number')) 
                )\
                .sort(["person_id", "concept_code", "measurement_date"], descending=True)\
                .unique(["person_id", "concept_code"], keep="first")\
                .select(["person_id", "measurement_date", "value_as_number", "source", "measurement_doc_id"])\
                .with_columns(
                         #doc_source = pl.lit("Lab"),
                        doc_source = pl.concat_str(
                                                [
                                                    pl.col("source") ,  
                                                    pl.when(pl.col('measurement_doc_id').is_null())
                                                        .then(pl.lit("NONE"))
                                                        .otherwise(pl.col('measurement_doc_id'))
                                                ], separator=":"),
                         measure = pl.lit("Criteria:mprotein_serum(Low Point*)"),
                         units = pl.lit("g/dl")
                )\
                .drop(["source", "measurement_doc_id"])\
                .rename({"measurement_date": "assessment_date", "value_as_number": "value"})
    
    
    if urine_results.is_empty():
        criteria_urine_results_status = urine_results.with_columns(pl.lit(None).cast(pl.Boolean).alias('criteria_flag'))
    else:
        criteria_urine_results_status = urine_results.sort(['person_id','measurement_date'],descending=False)\
                                        .groupby(['person_id','concept_code'])\
                                        .agg(
                                            pl.col('value_as_number'),
                                            pl.col('measurement_date'),
                                            pl.col('nadir'),
                                            pl.col('source'),
                                            pl.col('measurement_doc_id')
                                        )\
                                        .filter(pl.col('value_as_number').list.lengths() >1)   
        criteria_urine_results_status = criteria_urine_results_status\
                                    .with_columns(
                                        criteria_flag = pl.struct(['value_as_number','nadir']).apply(lambda x: rule_check_new(x["value_as_number"],x["nadir"],200), return_dtype=pl.List(pl.Boolean))
                                    )\
                                    .explode(['value_as_number','measurement_date','criteria_flag', 'nadir', 'source', 'measurement_doc_id'])

    criteria_urine_results = criteria_urine_results_status.join(pat_id_df,on='person_id',how='left')\
                                    .filter(
                                            (pl.col('criteria_flag')==True)
                                    )\
                                    .sort(['person_id','concept_code','measurement_date'], descending=True )\
                                    .with_columns(
                                          latest_pr_date = pl.col('measurement_date').max().over('person_id')
                                    )\
                                    .unique(subset='person_id',keep='first')
    urine_trend =  criteria_urine_results.sort(["person_id", "concept_code", "measurement_date"], descending=True)\
                    .unique(["person_id", "concept_code"], keep="first")\
                    .with_columns(trend = pl.col("value_as_number")-pl.col("nadir"),
                                 rate = (pl.col("value_as_number")-pl.col("nadir"))/pl.col("nadir")*100)
    evidence_urine_trend = urine_trend.select(["person_id", "measurement_date", "trend"])\
                                        .with_columns(
                                             doc_source = pl.lit("") ,
                                             description = pl.lit("Value Difference from Low Point to Latest Lab Test"),
                                             measure = pl.lit("Criteria:mprotein_urine:Absolute Trend"),
                                             units = pl.lit("g/dl")
                                         ).rename({"measurement_date": "assessment_date", "trend": "value"})
    evidence_urine_pct = urine_trend.select(["person_id", "measurement_date", "rate"])\
                                        .with_columns(
                                             doc_source = pl.lit("") ,
                                             description = pl.lit("Percentage change from Low Point to Latest Lab test"),
                                             measure = pl.lit("Criteria:mprotein_urine:Increase Rate"),
                                             units = pl.lit("%")
                                         ).rename({"measurement_date": "assessment_date", "rate": "value"})
    evidence_criteria_urine = criteria_urine_results_status.join(
                                     criteria_urine_results, 
                                    on= ["person_id", "concept_code"]
                                ).filter(pl.col("measurement_date") <= pl.col("measurement_date_right")) #Filter Old Records
    
    if not evidence_criteria_urine.is_empty():
        evidence_criteria_urine = pl.concat(
                                pl.collect_all([ x.lazy().top_k(2, by='measurement_date') for x in evidence_criteria_urine.partition_by(['person_id', 'concept_code'])])
                            )
    evidence_criteria_urine = evidence_criteria_urine.select(["person_id", "measurement_date", "value_as_number", "source", "measurement_doc_id"])\
                            .with_columns(
                                             #doc_source = pl.lit("Lab"),
                                             doc_source = pl.concat_str(
                                                [
                                                    pl.col("source") ,  
                                                    pl.when(pl.col('measurement_doc_id').is_null())
                                                        .then(pl.lit("NONE"))
                                                        .otherwise(pl.col('measurement_doc_id'))
                                                ], separator=":"),
                                             measure = pl.lit("Criteria:mprotein_urine"),
                                             units = pl.lit("mg/24hr")
                                         )\
                            .drop(["source", "measurement_doc_id"])\
                            .rename({"measurement_date": "assessment_date", "value_as_number": "value"})
    
    
    urine_nadir = urine_results.join(criteria_urine_results, how="semi", on=["person_id", "concept_code"])\
                .filter(
                            (pl.col('nadir') == pl.col('value_as_number')) 
                )\
                .sort(["person_id", "concept_code", "measurement_date"], descending=True)\
                .unique(["person_id", "concept_code"], keep="first")\
                .select(["person_id", "measurement_date", "value_as_number", "source", "measurement_doc_id"])\
                .with_columns(
                         #doc_source = pl.lit("Lab"),
                        doc_source = pl.concat_str(
                                        [
                                            pl.col("source") ,  
                                            pl.when(pl.col('measurement_doc_id').is_null())
                                                .then(pl.lit("NONE"))
                                                .otherwise(pl.col('measurement_doc_id'))
                                        ], separator=":"),
                         measure = pl.lit("Criteria:mprotein_urine(Low Point*)"),
                         units = pl.lit("mg/24hr")
                     )\
                .drop(["source", "measurement_doc_id"])\
            .rename({"measurement_date": "assessment_date", "value_as_number": "value"})
    
    rule1_df = criteria_serum_results.join(criteria_urine_results,on='person_id',how='outer',suffix='_urine')\
                            .with_columns(
                                min_date = pl.min_horizontal('measurement_date','measurement_date_urine'),
                                latest_pr_date = pl.max_horizontal('latest_pr_date', 'latest_pr_date_urine')
                            )
    
    criteria_df = criteria_df.join(rule1_df.select(['person_id','min_date', 'latest_pr_date']),on='person_id',how='left')\
                        .with_columns(
                            criteria_1 = pl.col('min_date').is_not_null()
                        )
    if free_light_chain_results.is_empty():
        criteria_flc_results_status = free_light_chain_results.with_columns(pl.lit(None).cast(pl.Boolean).alias('criteria_flag'))
    else:
        criteria_flc_results_status = free_light_chain_results.sort(['person_id','measurement_date'],descending=False)\
                                    .groupby(['person_id','concept_code'])\
                                    .agg(
                                        pl.col('value_as_number'),
                                        pl.col('measurement_date'),
                                        pl.col('nadir'),
                                        pl.col('source'),
                                        pl.col('measurement_doc_id')
                                    )\
                                    .filter(pl.col('value_as_number').list.lengths()>1)
    
        criteria_flc_results_status = criteria_flc_results_status\
                                .with_columns(
                                    criteria_flag = pl.struct(['value_as_number','nadir']).apply(lambda x: rule_check_new(x["value_as_number"],x["nadir"],10), return_dtype=pl.List(pl.Boolean))
            
                                )\
                                .explode(['value_as_number','measurement_date','criteria_flag', 'nadir', 'source', 'measurement_doc_id'])

    
    #evidence_criteria_flc = criteria_flc_results_status.filter(
    #                                    (pl.col('criteria_flag')==True)
    #                                )\
    #                                .sort(['person_id','measurement_date'], descending= True)\
    #                                .unique(subset='person_id',keep='first')\
    #                                .select(["person_id", "measurement_date", "value_as_number"])\
    #                                .with_columns(
    #                                                 doc_source = pl.lit("Lab"),
    #                                                 measure = pl.lit("Criteria:flc"),
    #                                                 units = pl.lit("mg/dl")
    #                                             )\
    #                                .rename({"measurement_date":"assessment_date", "value_as_number": "value"})
    
    
    flc_nadir = free_light_chain_results.join(criteria_flc_results_status.filter((pl.col('criteria_flag')==True)), how="semi", on=["person_id", "concept_code"])\
                .filter(
                            (pl.col('nadir') == pl.col('value_as_number')) 
                )\
                .sort(["person_id", "concept_code", "measurement_date"], descending=True)\
                .unique(["person_id", "concept_code"], keep="first")\
                .select(["person_id", "measurement_date", "value_as_number", "concept_code",'source', 'measurement_doc_id'])\
                .with_columns(
                         #doc_source = pl.lit("Lab"),
                         doc_source = pl.concat_str(
                                        [
                                            pl.col("source") ,  
                                            pl.when(pl.col('measurement_doc_id').is_null())
                                                .then(pl.lit("NONE"))
                                                .otherwise(pl.col('measurement_doc_id'))
                                        ], separator=":"),
                         #measure = pl.lit("Criteria:flc:nadir"),
                         measure = pl.when(pl.col('concept_code').is_in(lab_dict['serum_free_light_kappa']))
                                                .then(pl.lit('Criteria:Kappa Free Light Chain(Low Point*)'))
                                                .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_lambda']))
                                                .then(pl.lit('Criteria:Lambda Free Light Chain(Low Point*)')),
                         units = pl.lit("mg/dl")
                     )\
            .rename({"measurement_date": "assessment_date", "value_as_number": "value"})\
            .drop(["concept_code", 'source', 'measurement_doc_id'])
        
    
    criteria_flc_results = criteria_flc_results_status.join(pat_id_df,on='person_id',how='left')\
                                    .filter(
                                        (pl.col('criteria_flag')==True)
                                    )\
                                    .sort(['person_id','concept_code','measurement_date'], descending=True)\
                                    .with_columns(
                                          latest_pr_date = pl.col('measurement_date').max().over('person_id')
                                    )\
                                    .unique(subset='person_id',keep='first')\
                                    .select(['person_id','measurement_date', 'latest_pr_date', 'concept_code']) 
    
    flc_trend =  criteria_flc_results_status.filter((pl.col('criteria_flag')==True)).sort(["person_id", "concept_code", "measurement_date"], descending=True)\
                    .unique(["person_id", "concept_code"], keep="first")\
                    .with_columns(trend = pl.col("value_as_number")-pl.col("nadir"),
                                 rate = (pl.col("value_as_number")-pl.col("nadir"))/pl.col("nadir")*100)
    evidence_flc_trend = flc_trend.select(["person_id", "measurement_date", "trend", 'concept_code'])\
                                        .with_columns(
                                             doc_source = pl.lit("") ,
                                             description = pl.lit("Value Difference from Low Point to Latest Lab Test"),
                                             measure = pl.when(pl.col('concept_code').is_in(lab_dict['serum_free_light_kappa']))
                                                .then(pl.lit('Criteria:Kappa Free Light Chain:Absolute Trend'))
                                                .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_lambda']))
                                                .then(pl.lit('Criteria:Lambda Free Light Chain:Absolute Trend')),
                                             units = pl.lit("mg/dl")
                                         ).rename({"measurement_date": "assessment_date", "trend": "value"})\
                            .drop(["concept_code"])
    evidence_flc_pct = flc_trend.select(["person_id", "measurement_date", "rate", 'concept_code'])\
                                        .with_columns(
                                             doc_source = pl.lit("Percentage change from Low Point to Latest Lab test"),
                                              measure = pl.when(pl.col('concept_code').is_in(lab_dict['serum_free_light_kappa']))
                                                .then(pl.lit('Criteria:Kappa Free Light Chain:Increase Rate'))
                                                .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_lambda']))
                                                .then(pl.lit('Criteria:Lambda Free Light Chain:Increase Rate')),
                                             units = pl.lit("%")
                                         ).rename({"measurement_date": "assessment_date", "rate": "value"})\
                            .drop(["concept_code"])
    
    
    evidence_criteria_flc = criteria_flc_results_status.join(
                                     criteria_flc_results, 
                                    on= ["person_id", "concept_code"]
                                ).filter(pl.col("measurement_date") <= pl.col("measurement_date_right")) #Filter Old Records
    if not evidence_criteria_flc.is_empty():
        evidence_criteria_flc = pl.concat(
                                pl.collect_all([ x.lazy().top_k(2, by='measurement_date') for x in evidence_criteria_flc.partition_by(['person_id', 'concept_code'])])
                            )
    evidence_criteria_flc = evidence_criteria_flc.select(["person_id", "measurement_date", "value_as_number", "concept_code", "source", "measurement_doc_id"])\
                            .with_columns(
                                             #doc_source = pl.lit("Lab"),
                                             doc_source = pl.concat_str(
                                                    [
                                                        pl.col("source") ,  
                                                        pl.when(pl.col('measurement_doc_id').is_null())
                                                            .then(pl.lit("NONE"))
                                                            .otherwise(pl.col('measurement_doc_id'))
                                                    ], separator=":"),
                                             measure = pl.when(pl.col('concept_code').is_in(lab_dict['serum_free_light_kappa']))
                                                .then(pl.lit('Criteria:Kappa Free Light Chain'))
                                                .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_lambda']))
                                                .then(pl.lit('Criteria:Lambda Free Light Chain')),
                                             units = pl.lit("mg/dl")
                                         )\
                            .rename({"measurement_date": "assessment_date", "value_as_number": "value"})\
                            .drop(["concept_code", "source", "measurement_doc_id"])
    
    criteria_df = criteria_df.join(criteria_flc_results.drop(["concept_code"]),on='person_id',how='left', suffix="_criteria2")\
                        .with_columns(
                            min_date = pl.min_horizontal('min_date','measurement_date'),
                            criteria_2 = pl.col('measurement_date').is_not_null(),
                            latest_pr_date = pl.min_horizontal('latest_pr_date', 'latest_pr_date_criteria2')
                        ).drop(['measurement_date', 'latest_pr_date_criteria2'])
    
    evidence_criteria = pl.concat(
        [
            evidence_criteria_serum, 
            serum_nadir, 
            evidence_serum_trend,
            evidence_serum_pct,
             
            evidence_criteria_urine, 
            urine_nadir, 
            evidence_urine_trend, 
            evidence_urine_pct, 

            evidence_criteria_flc, 
            flc_nadir, 
            evidence_flc_trend, 
            evidence_flc_pct
        ], how="diagonal")
    
    
    rule4_df = con_plasmacytoma_df.with_columns(
                    min_date_4 = pl.col('condition_start_date')
                ).select(['person_id','min_date_4'])
    
    criteria_df = criteria_df.join(rule4_df,on='person_id',how='left')\
                        .with_columns(
                            min_date = pl.min_horizontal('min_date','min_date_4'),
                            criteria_4 = pl.col('min_date_4').is_not_null(),
                            latest_pr_date = pl.max_horizontal('latest_pr_date', 'min_date_4')
                        ).drop('min_date_4')
    
    criteria_df = criteria_df.with_columns(
                      final_selection =   (pl.col('criteria_1') | pl.col('criteria_2') | pl.col('criteria_4'))\
                    )
    
    return criteria_df, evidence_criteria


def set_index_point(criteria_df, diagnosis_df, n_days=0):
    """
    Method to Set the Index Point('random_point' used while training)

    ARGUMENTS:
    ------------
    criteria_df: Polars.DataFrame:
    diagnosis_df: Polars.DataFrame:
    n_days: Number of Days from Today for IndexPoint.
        If n_days = 0, Then index_point/random_point(from training) is set as Today

    RETURNS:
    -----------
    criteria_df: Polars.DataFrame: criteria_df with index point
    """
    criteria_df = criteria_df.join(diagnosis_df, on='person_id', how='inner')
    index_date = datetime.datetime.today() - timedelta(days=n_days)
    #criteria_df = criteria_df.with_columns(
    #                    diff = (pl.col('last_activity_date')-pl.col('condition_start_date')).dt.days()
    #            )
    criteria_df = criteria_df.with_columns(
                        random_point = (index_date - pl.col('condition_start_date')).dt.days()
                    )
    return criteria_df


def generate_random_point(criteria_df,diagnosis_df):
    """
    Method to Generate a random point for criteria_df, diagnosis_df
    
    ARGUMENTS:
    -----------
    criteria_df: Polars.DataFrame:
    diagnosis_df: Polars.DataFrame: 
    
    RETURNS:
    -----------
    criteria_df: Polars.DataFrame: criteria_df with random index point between condition_start_date and last activity date
    """
    criteria_df = criteria_df.join(diagnosis_df,left_on='person_id',right_on = "person_id", how='inner')
    criteria_df = criteria_df.with_columns(
                        diff = (pl.col('last_activity_date')-pl.col('condition_start_date')).dt.days()
                )
    criteria_df = criteria_df.filter(pl.col('diff')>=0)
    if criteria_df.is_empty():
        criteria_df = criteria_df.with_columns(pl.lit(None).cast(pl.Int64).alias('random_point'))
    else:
        criteria_df = criteria_df.with_columns(
                        random_point = pl.col('diff').apply(lambda x:random.randint(0,x))
                    )
    
    return criteria_df

def tumor_grade_feature_extraction(patient_ids, criteria_df, parquet_path=s3_parquet_base_path):
    """
    Method to fetch Tumor Grade as Feature for the selected patient_ids
    
    ARGUMENTS:
    -----------
    patient_ids: List : List of Patient Ids ofr tumor grade feature
    criteria_df: Polars.DataFrame: Criteria Data Frame to filter the Features which are older than the Index point
    parquet_path: str: S3 Base path for loading the parquet files
    RETURNS:
    ------------
    tumor_exam_feat_df: Polars.DataFrame: DataFrame with columns of Tumor grades
    
    """
    tumor_grade_exam_results = get_tumor_exam_results(patient_ids, parquet_path=parquet_path)
    #Join tumor_grade results and criteria dataframes

    tumor_grade_criteria = tumor_grade_exam_results.join(
                            criteria_df.select(['person_id','condition_start_date','random_point']), 
                            left_on = 'person_id',
                            right_on = 'person_id',
                            how = 'left'
                        ).drop_nulls()\
                        .with_columns(
                            exam_diff = (pl.col('exam_date') - pl.col('condition_start_date')).dt.days()
                        ).filter(
                            pl.col('exam_diff') < pl.col('random_point')
                        ).sort(by=['person_id','exam_date'],descending=True).unique(subset=['person_id'],keep='first')
    
    grades = ["Grade 1", "Grade 2", "Grade 3", "Grade X"]
    tumor_exam_feat_df = tumor_grade_criteria.select('person_id', 'concept_name')
    for grade in grades:
        tumor_exam_feat_df = tumor_exam_feat_df.with_columns(
            (pl.col('concept_name') == grade).alias(grade)
        )
    tumor_exam_feat_df = tumor_exam_feat_df.drop('concept_name')
    
    return tumor_exam_feat_df

def bone_marrow_feature_extraction(bone_marrow_df, surgery_df, criteria_df,schema_type=None):
    bone_marrow_df = bone_marrow_df.join(
                        criteria_df.select(['person_id','condition_start_date','random_point']), 
                                left_on = 'person_id',
                                right_on = 'person_id',
                                how = 'inner'
                    ).filter(
                        (pl.col("measurement_date") >= pl.col("condition_start_date")) & 
                        (
                            (pl.col("measurement_date") - pl.col("condition_start_date")).dt.days() <= pl.col("random_point")
                        )
                    )
    surgery_df = surgery_df.join( 
                                criteria_df.select(['person_id','condition_start_date','random_point']), 
                                left_on = 'person_id',
                                right_on = 'person_id',
                                how = 'inner'
                ).filter(
                        (pl.col("procedure_date") >= pl.col("condition_start_date")) & 
                        (
                            (pl.col("procedure_date") - pl.col("condition_start_date")).dt.days() <= pl.col("random_point")
                        )
                    )
    bone_marrow_patients = list(set(bone_marrow_df["person_id"].to_list()))
    surgery_patients = list(set(surgery_df["person_id"].to_list()))
    
    bone_marrow_feat_df = pl.DataFrame(criteria_df["person_id"].unique())
    bone_marrow_feat_df = bone_marrow_feat_df.with_columns(
                            pl.col("person_id").is_in(set(bone_marrow_patients + surgery_patients)).alias("bone_marrow").cast(pl.Boolean)
                        )
    
    return bone_marrow_feat_df

def lab_test_feature_extraction(patient_test_results, criteria_df,schema_type=None):
    """
    Method to extract Lab Test Results  as Feature for the selected patient_ids
    
    ARGUMENTS:
    -----------
    patient_test_results: Polars.DataFrame : Lab Test details
    criteria_df: Polars.DataFrame: Criteria Data Frame to filter the Features which are older than the Index point
    
    RETURNS:
    ------------
    lab_test_df_delta_feature: Polars.DataFrame: DataFrame with columns of Lab tests
    
    """
    lab_test_df = patient_test_results.join(
                    criteria_df.select(['person_id','condition_start_date','random_point']), 
                            left_on = 'person_id',
                            right_on = 'person_id',
                            how = 'inner'
                    )\
                .with_columns(
                  test_diff =  (pl.col('measurement_date') - pl.col('condition_start_date')).dt.days()
                ).sort(by=['person_id','concept_code','measurement_date'],descending=False)


    lab_test_df_delta_feature = lab_test_df.groupby(['person_id','concept_code'])\
                                    .agg(
                                            pl.col('value_as_number').diff().alias('delta_test_value_numeric'),
                                            pl.col('measurement_date').diff().dt.days().alias('delta_test_date'),
                                            pl.col('random_point'),
                                            pl.col('value_as_number'),
                                            pl.col('test_diff')            
                                    )\
                                    .explode(columns=['delta_test_date', 'delta_test_value_numeric','random_point','value_as_number','test_diff'])\
                                    .with_columns(
                                        (pl.col('delta_test_value_numeric')/(pl.col('delta_test_date')+0.001)).alias('delta')
                                    )
    
    lab_test_df_delta_feature = lab_test_df_delta_feature.with_columns(
                        pl.when(
                            (pl.col('test_diff') < pl.col('random_point')) & 
                                    (pl.col('test_diff') > pl.col('random_point')-INPUT_TIME_WINDOW_RANGE[0])
                        ).then(pl.lit(0))\
                        .when(
                            (pl.col('test_diff') < pl.col('random_point')) & 
                                    (pl.col('test_diff') > pl.col('random_point')-INPUT_TIME_WINDOW_RANGE[1])
                        ).then(pl.lit(1))\
                        .when(
                            (pl.col('test_diff') < pl.col('random_point')) & 
                                    (pl.col('test_diff') > pl.col('random_point')-INPUT_TIME_WINDOW_RANGE[2])
                        ).then(pl.lit(2))\
                        .otherwise(pl.lit(-1))
                        .alias('window_tag')   
                    ).filter(pl.col('window_tag')!=-1)
    
    #lab_test_df_delta_feature = lab_test_df_delta_feature.join(stat_df, on="person_id", how="inner")
    
    return lab_test_df_delta_feature
def get_latest_test_data(patient_test_results, criteria_df, feat_lab_list,schema_type=None):
    lab_test_df = patient_test_results.join(
                    criteria_df.select(['person_id','condition_start_date','random_point']), 
                            left_on = 'person_id',
                            right_on = 'person_id',
                            how = 'inner'
                    )\
                .with_columns(
                  test_diff =  (pl.col('measurement_date') - pl.col('condition_start_date')).dt.days()
                ).sort(by=['person_id','concept_code','measurement_date'],descending=False)
    
    stat_df = pl.DataFrame({'person_id': list(set(lab_test_df['person_id']))})
    stat_df = stat_df.with_columns(
                    latest_date = pl.lit(None).cast(pl.Date)
            )
    lab_test_df_latest = lab_test_df.filter(pl.col('test_diff')<=pl.col('random_point'))
    
    #Filter records in last 90 days(or within 90 days from refresh_date/random_point)
    lab_test_df_latest = lab_test_df_latest.filter(pl.col('test_diff')>=(pl.col('random_point')-90))
    
    lab_test_df_latest = lab_test_df_latest.sort(by=['person_id','concept_code','measurement_date'],descending=True)\
                             .unique(subset=['person_id','concept_code'],keep='first')\
                             .select(['person_id','concept_code','value_as_number', 'measurement_date', 'nadir'])
    
    for lab in feat_lab_list:
        #lab_name = test_code_map[lab]
        lab_name = lab_class_map[lab]  # Lab Test Groupping
        temp_df = lab_test_df_latest.filter(pl.col('concept_code')==lab)\
                            .with_columns(
                                (pl.col('value_as_number') - pl.col('nadir')).alias(f'abs_change_from_nadir_{lab_name}'),
                                ((pl.col('value_as_number') - pl.col('nadir'))/(pl.col('nadir')+0.0001)).alias(f'perc_change_from_nadir_{lab_name}'),
                            )\
                             .select(['person_id','value_as_number', 'measurement_date', f'abs_change_from_nadir_{lab_name}', f'perc_change_from_nadir_{lab_name}' ])\
                             .rename({'value_as_number':lab_name})
        if temp_df.shape[0]>0:
            stat_df = stat_df.join(temp_df,on='person_id',how='left')
            stat_df = stat_df.with_columns(
                latest_date = pl.max_horizontal('latest_date', 'measurement_date')
            ).drop('measurement_date')
        else:
            print(lab_name + " has no shape")
            stat_df = stat_df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias(lab_name),
                    pl.lit(None).cast(pl.Float64).alias(f'abs_change_from_nadir_{lab_name}'),
                    pl.lit(None).cast(pl.Float64).alias(f'perc_change_from_nadir_{lab_name}'),
            )
    return stat_df

def calculate_statistics(df, lab,schema_type=None, indices=[0,1,2]):
    """
    Generate the Lab statistics as Features in window phased manner along with the delta details
    
    ARGUMENTS:
    ----------
    df: Polars.DataFrame: DataFrame object to calculate the statistics
    lab: string: Lab Name
    
    RETURNS:
    --------
    final_df: Polars.DataFrame: Dataframe with the lab statistics
    
    """
    df = df.filter(pl.col("concept_code") == lab)
    final_df = pl.DataFrame({'person_id': list(set(df['person_id']))})
    
    # Group the data by patient ID and calculate the desired statistics
    for index in indices:
        df_window = df.filter(df['window_tag']==index)
        grouped_df =  df_window.groupby('person_id').agg(pl.col('value_as_number'))
        grouped_df_delta = df_window.groupby('person_id').agg(pl.col('delta'))
        
        df_window = df_window.with_columns(pl.col('test_diff')-10000)
        grouped_df_slope = df_window.sort(by=['test_diff'],descending=False).groupby('person_id').agg(pl.col('value_as_number'))
        
        if grouped_df_slope.shape[0]>0:
            grouped_df_slope = grouped_df_slope.filter(pl.col('value_as_number').list.lengths() > 1)
        if grouped_df_slope.shape[0]>0:
            grouped_df_slope = grouped_df_slope.join(
                                        df_window.sort(by=['test_diff'],descending=False)\
                                                .groupby('person_id').agg(pl.col('test_diff')),
                                        on='person_id',
                                        how='inner')
            
        lab_param = test_code_map[lab] + '_' + str(index)
        if grouped_df.shape[0]>0:
            # Calculate the statistics for each patient
            statistics_df = grouped_df.with_columns(
                    pl.col('value_as_number').list.mean().alias(f'Mean_{lab_param}'),
                    pl.col('value_as_number').list.eval(pl.element().median()).list.first().alias(f'Median_{lab_param}'),
                    pl.col('value_as_number').list.min().alias(f'Minimum_{lab_param}'),
                    pl.col('value_as_number').list.max().alias(f'Maximum_{lab_param}'),
                    pl.col("value_as_number").list.eval(pl.element().quantile(0.25, 'linear')).list.first().alias(f'25th_Percentile_{lab_param}'),
                    pl.col("value_as_number").list.eval(pl.element().quantile(0.75, 'linear')).list.first().alias(f'75th_Percentile_{lab_param}'),
                    (pl.col('value_as_number').list.min() - pl.col('value_as_number').list.max()).alias(f'Range_{lab_param}')                                                 
            )
            grouped_df_delta = grouped_df_delta.with_columns(
                    pl.col('delta').list.mean().alias(f'Mean_delta_{lab_param}'),
                    pl.col('delta').list.min().alias(f'Minimum_delta_{lab_param}'),
                    pl.col('delta').list.max().alias(f'Maximum_delta_{lab_param}')
            )
            
            statistics_df = statistics_df.join(grouped_df_delta,on='person_id',how='left')
            
            statistics_df = statistics_df.drop(['value_as_number','delta'])
        
           
            grouped_df = grouped_df.filter(pl.col('value_as_number').list.lengths() > 1)
            if grouped_df.shape[0]>0:
                stats_SD_df = grouped_df.with_columns(pl.col('value_as_number').apply(lambda x:x.std()).alias(f'SD_{lab_param}'))
                statistics_df = statistics_df.join(stats_SD_df.select(['person_id',f'SD_{lab_param}']),on='person_id',how='left')
              
                grouped_df = grouped_df.filter(pl.col('value_as_number').list.lengths()>2)
                if grouped_df.shape[0]>0:
                    stats_skew_df = grouped_df.with_columns(pl.col('value_as_number').apply(lambda x:x.skew()).alias(f'Skewness_{lab_param}'))              
                    statistics_df = statistics_df.join(stats_skew_df.select(['person_id',f'Skewness_{lab_param}']),on='person_id',how='left')

                    grouped_df = grouped_df.filter(pl.col('value_as_number').list.lengths()>3)
                    if grouped_df.shape[0]>0:
                        stats_kurtosis_df = grouped_df.with_columns(pl.col('value_as_number').apply(lambda x:x.kurtosis()).alias(f'Kurtosis_{lab_param}')) 
                        statistics_df = statistics_df.join(stats_kurtosis_df.select(['person_id',f'Kurtosis_{lab_param}']),on='person_id',how='left')
                    else:
                        statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'))
                else:
                    statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                                               pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'))
            else:
                statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                                           pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'),
                                                           pl.lit(None).cast(pl.Float64).alias(f'SD_{lab_param}'))
            print("Polyfitting Begins")
            print("Grouped DF Slope", grouped_df_slope)
            if grouped_df_slope.shape[0]>0:
                
                grouped_df_slope = grouped_df_slope.with_columns(pl.struct(['test_diff','value_as_number']).apply(lambda x:np.polyfit(x['test_diff'],x['value_as_number'],1)[0]).alias(f'slope_{lab_param}'))          
                statistics_df = statistics_df.join(grouped_df_slope.select(['person_id',f'slope_{lab_param}']),on = 'person_id',how='left')
            else:
                statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'slope_{lab_param}'))
                
            
            final_df = final_df.join(statistics_df, on="person_id", how="left")
        else:
            final_df = final_df.with_columns(
                                        pl.lit(None).cast(pl.Float64).alias(f'Mean_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Median_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Minimum_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Maximum_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'25th_Percentile_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'75th_Percentile_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Range_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Mean_delta_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Minimum_delta_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Maximum_delta_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'SD_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'slope_{lab_param}')
                                        )
    if len(indices) > 2:
        for stat in ["Mean_", "Minimum_", "Maximum_"]:
            f0 = f"{stat}{test_code_map[lab]}_{indices[0]}"
            f1 = f"{stat}{test_code_map[lab]}_{indices[1]}"
            f2 = f"{stat}{test_code_map[lab]}_{indices[2]}"


            if final_df.filter(pl.col(f0).is_not_null() & pl.col(f1).is_not_null()).shape[0]>0:
                try:
                    final_df = final_df.with_columns(
                    pl.when(pl.col(f0).is_not_null() & pl.col(f1).is_not_null())\
                    .then(
                        (pl.col(f0)- pl.col(f1))/(pl.col(f1)+0.001)
                    )\
                    .otherwise(None)
                    .alias(f"{stat}{test_code_map[lab]}_{indices[1]}_to_{indices[0]}_per_change"))
                except Exception as e:
                    traceback.print_exc()

            if f"{stat}{test_code_map[lab]}_{indices[1]}_to_{indices[0]}_per_change" not in list(final_df.columns):
                final_df = final_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{stat}{test_code_map[lab]}_{indices[1]}_to_{indices[0]}_per_change"))

            if final_df.filter(pl.col(f1).is_not_null() & pl.col(f2).is_not_null()).shape[0]>0:
                try:
                    final_df = final_df.with_columns(
                        pl.when(pl.col(f1).is_not_null() & pl.col(f2).is_not_null())\
                        .then(
                            (pl.col(f1)- pl.col(f2))/(pl.col(f2)+0.001)
                        )\
                        .otherwise(None)
                        .alias(f"{stat}{test_code_map[lab]}_{indices[2]}_to_{indices[1]}_per_change"))
                except Exception as e:
                    traceback.print_exc()

            if f"{stat}{test_code_map[lab]}_{indices[2]}_to_{indices[1]}_per_change" not in list(final_df.columns):
                final_df = final_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{stat}{test_code_map[lab]}_{indices[2]}_to_{indices[1]}_per_change"))
        
    final_df = final_df.with_columns(
                    pl.col('person_id').cast(pl.Utf8)
                )
    return final_df

def to_pandas(df):
    """
    Method to convert Polars DataFrame Object to Pandas Data Frame Object
    
    ARGUMENTS:
    ----------
    df: Polars.DataFrame: Polars Data Frame object to convert
    
    RETURNS:
    --------
    pandas_df: Pandas.DataFrame
    """
    
    pandas_df = df.to_pandas()
    
    #Polars boolean dtype is converted to object type through `to_pandas()`, recasting to boolean type
    bool_cols = df.select(pl.col(pl.Boolean)).columns
    for col in bool_cols:
        pandas_df[col] = pandas_df[col].astype('bool')
        
    float_cols = df.select(pl.col(pl.Float64)).columns
    for col in float_cols:
        pandas_df[col] = pandas_df[col].astype('float')
    
    return pandas_df


logger = utils.setup_logger("feature_generator")
global concept_df


def get_met_dates(cohort_type, pat_id_list, parquet_path,schema_type):
    """
    Method to fetch the earliest date of Metastatic cancer is identified using the met_rules specific to cancer_type
    @Args:
    cohort_type: str: Cancer indicator
    pat_id_list: List: List of Patients 
    
    @returns:
    met_df: Polars.DataFrame: DataFrame of Patient Id and correspoding Date identified
    evidence_df: Polars.DataFrame: DataFrame as Evidence for the identified date
    """
    
    met_df = pl.DataFrame({"person_id": pat_id_list})\
                    .with_columns(met_date = pl.lit(None).cast(pl.Date))\
                    .filter(pl.col('met_date').is_not_null())
    evidence_met = pl.DataFrame(
                                        {"person_id":[], "assessment_date": [], "value": [], "value_name": [], "doc_source": [], "measure": []},
                                        schema = {
                                                    "person_id": met_df.schema['person_id'],
                                                    "assessment_date": pl.Date,
                                                    "value": pl.Float64,
                                                    "value_name": pl.Utf8,
                                                    "doc_source": pl.Utf8,
                                                    "measure": pl.Utf8
                                                }
                                   )
    
    if cohort_type in met_rules:
        met_rule = met_rules[cohort_type]
    else:
        logger.info(f"No Metastatic Rulee defined for the cancer {cohort_type}. Skipping extracting the features")
        return met_df, evidence_met
    
    if met_rule["mdx"]:
        if cohort_type.lower() in ["nsclc", "sclc"]:
            diagnosis_codes =  get_diagnosis_codes("lung")
        elif cohort_type.lower() == "ovarian":
            diagnosis_codes =  get_diagnosis_codes("ofpc")
        else:
            diagnosis_codes = get_diagnosis_codes(cohort_type.lower())
        
        conition_test_lazy = data_loader.get_table_df("condition", parquet_base_path=parquet_path, schema=schema_type)
        condition_column_names = tables.get_column_name("condition", schema=schema_type)
        pat_id_df = conition_test_lazy.filter(
                             (pl.col(condition_column_names["diagnosis_code_standard_code"]).is_in(diagnosis_codes)) & \
                             (pl.col('person_id').is_in(pat_id_list))
                        ).collect()
        
        pat_id_df = pat_id_df.rename({
                    condition_column_names["chai_patient_id"]: "person_id",
                    condition_column_names["diagnosis_date"]: "condition_start_date",
                    condition_column_names["diagnosis_code_standard_code"]: "concept_code",
                    condition_column_names["diagnosis_code_standard_name"]: "concept_name",
                    condition_column_names["diagnosis_type_concept_id"]: "condition_type_concept_id"
                })
        
        pat_id_df = pat_id_df.with_columns(
                pl.col(condition_column_names["diagnosis_type_concept_id"]).cast(pl.Int64),
                pl.col('condition_start_date').cast(pl.Date)
            )
        
        condition_met_rule = pat_id_df.filter(pl.col("condition_type_concept_id") == 4032806)\
                .select(["person_id", "condition_start_date", "source", "condition_doc_id"])\
                .sort(by=["person_id", "condition_start_date"])\
                .unique(subset=['person_id'],keep='first')\
                .rename({"condition_start_date": "met_date"})
        met_df = pl.concat([met_df, condition_met_rule.select(["person_id", "met_date"]) ])
        
        evidence_met_dx = condition_met_rule.with_columns(
                            value = pl.lit(4032806).cast(pl.Float64),
                            measure = pl.lit("Metastatic:diagnosis_type"),
                            #doc_source = pl.lit("Diagnosis")
                            doc_source = pl.concat_str(
                                        [
                                            pl.col("source") ,  
                                            pl.when(pl.col('condition_doc_id').is_null())
                                                .then(pl.lit("NONE"))
                                                .otherwise(pl.col('condition_doc_id'))
                                        ], separator=":")
                        ).rename({"met_date": "assessment_date"}).drop(["source", "condition_doc_id"])
        evidence_met = pl.concat([evidence_met, evidence_met_dx ], how="diagonal")
    
    staging_df = get_staging_data(pat_id_list, parquet_path,schema_type=schema_type)
    #Fetch Met Rule on Stage Name
    for stage_name in met_rule["stage"]:
        met_stage = staging_df.select(["person_id", "stage_date", "stage_group_standard_name", "source", "staging_doc_id"])\
                    .filter(pl.col("stage_group_standard_name").str.contains("(?i)" +stage_name))\
                    .sort(by=["person_id", "stage_date"])\
                    .unique(subset=['person_id'],keep='first')\
                    .rename({"stage_date": "met_date"})
                    
        met_df = pl.concat([met_df, met_stage.select(["person_id", "met_date"]) ])
        evidence_stage =  met_stage.select(["person_id", "met_date", "stage_group_standard_name","source", "staging_doc_id"])\
                                .with_columns(
                                    measure = pl.lit("Metastatic:StageName"),
                                    #doc_source = pl.lit("Stage")
                                    doc_source = pl.concat_str(
                                        [
                                            pl.col("source") ,  
                                            pl.when(pl.col('staging_doc_id').is_null())
                                                .then(pl.lit("NONE"))
                                                .otherwise(pl.col('staging_doc_id'))
                                        ], separator=":")
                                ).rename({"met_date": "assessment_date", "stage_group_standard_name": "value_name"})
        evidence_met = pl.concat([evidence_met, evidence_stage ], how = "diagonal").drop(['source', "staging_doc_id"])
    #Fetch Met Rule on MStage Name
    for mstage in met_rule["mstage"]:
        met_stage = staging_df.select(["person_id", "stage_date", "mstage_standard_name", "source", "staging_doc_id"])\
                    .filter(pl.col("mstage_standard_name").str.contains("(?i)" +mstage))\
                    .sort(by=["person_id", "stage_date"])\
                    .unique(subset=['person_id'],keep='first')\
                    .rename({"stage_date": "met_date"})
                    
        met_df = pl.concat([met_df, met_stage.select(["person_id", "met_date"]) ])
        
        evidence_stage =  met_stage.select(["person_id", "met_date", "mstage_standard_name", "source", "staging_doc_id"])\
                                .with_columns(
                                    measure = pl.lit("Metastatic:MStageName"),
                                    #doc_source = pl.lit("Stage")
                                    doc_source = pl.concat_str(
                                        [
                                            pl.col("source") ,  
                                            pl.when(pl.col('staging_doc_id').is_null())
                                                .then(pl.lit("NONE"))
                                                .otherwise(pl.col('staging_doc_id'))
                                        ], separator=":")
                                ).rename({"met_date": "assessment_date", "mstage_standard_name": "value_name"})
        evidence_met = pl.concat([evidence_met, evidence_stage ], how="diagonal").drop(['source', "staging_doc_id"])
    
    #Filter for the earliest identified date
    met_df = met_df.sort(by=["person_id", "met_date"])\
                    .unique(subset=['person_id'],keep='first')
    
    return met_df, evidence_met


def generate_features(pat_ids_list=None, parquet_path=s3_parquet_base_path, is_training=False, data_traceability_wb=None, schema_type=None, configurations=MultipleMyelomaConfigurations()):
    """
    Feature Extraction for MM Specific cancer type patients
    """
    logger.info("Feature Extraction Begins")

    last_activity_data = dict()

    #Fetch Patient List of MM Cancer type(diagnosis_code)
    diagnosis_codes = get_diagnosis_codes("mm")
    if is_training:
        pat_id_df = get_patient_diagnosis(diagnosis_codes, parquet_path,schema_type=schema_type)
    else:
        pat_id_df = get_patient_diagnosis(diagnosis_codes, parquet_path, dx_year=None, schema_type=schema_type)

    if pat_ids_list is None:
        pat_ids_list = pat_id_df['person_id'].to_list()
        pat_ids_list = list(set(pat_ids_list))
    else:
        pat_id_df = pat_id_df.filter(pl.col('person_id').is_in(pat_ids_list))
          
    logger.info("Total Patients Targeted:" + str( len(pat_ids_list)) )
    
    pat_id_df = pat_id_df.select(['person_id','condition_start_date'])\
        .sort(by =['person_id','condition_start_date'],descending=False,nulls_last=True)\
        .unique(subset=['person_id'],keep='first')
    
    #Fetch the Patient Test Results
    serum_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["m_protein_serum"]
    urine_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["m_protein_urine"]
    free_light_chain_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["free_light_chain_reaction"]

    patient_test_codes = serum_test_code["concept_codes"] + urine_test_code["concept_codes"]  + free_light_chain_test_code["concept_codes"] #+ bone_marrow_test_code
    patient_test_names = serum_test_code["concept_names"] + urine_test_code["concept_names"]  + free_light_chain_test_code["concept_names"]
    
    logger.info("Fetching Targeted Patients Lab Tests")
    patient_test_df = get_patient_results(pat_ids_list, patient_test_codes, patient_test_names, parquet_path=parquet_path, schema_type=schema_type)\
                        .drop_nulls(subset='value_as_number')\
                        .drop("test_value_name")
    
    patient_test_df = patient_test_df.join(pat_id_df, on="person_id").filter(pl.col("measurement_date")>=pl.col("condition_start_date")).drop("condition_start_date")

    logger.debug("Processing Lab Tests")
    serum_results = get_test_results(patient_test_df, serum_test_code["concept_codes"],serum_test_code["concept_names"], schema_type=schema_type, default_key = 'm_protein_in_serum' )
    urine_results = get_test_results(patient_test_df, urine_test_code["concept_codes"],urine_test_code["concept_names"], schema_type=schema_type, default_key = 'm_protein_in_urine' )
    free_light_chain_results = get_test_results(patient_test_df, free_light_chain_test_code["concept_codes"],free_light_chain_test_code["concept_names"],schema_type=schema_type, default_key = 'serum_free_light')
    logger.debug("Lab Tests Extraction Completed")
    #print(serum_results, urine_results, free_light_chain_results )
    
    con_plasmacytoma_df = get_patient_diagnosis(['C90.2', 'C90.20', 'C90.21'], parquet_location=parquet_path, schema_type=schema_type)
    con_plasmacytoma_df = con_plasmacytoma_df.select(['person_id','condition_start_date'])\
                                            .sort(by =['person_id','condition_start_date'],descending=False, nulls_last=True)\
                                            .unique(subset=['person_id'],keep='first')\
                                            .drop_nulls()
    
    con_plasmacytoma_df = con_plasmacytoma_df.join(pat_id_df,on='person_id',how='left',suffix='_con')\
                                        .filter(pl.col('condition_start_date')>=pl.col('condition_start_date_con'))\
                                        .sort(by=['person_id','condition_start_date'],descending=False)\
                                        .unique(subset='person_id',keep='first')\
                                        .select(['person_id','condition_start_date'])
    
    
    progression_dates = get_progression_list(pat_ids_list, serum_results, urine_results, free_light_chain_results, con_plasmacytoma_df)
    progression_dates = progression_dates.sort(["person_id", "progression_date"]).groupby("person_id").agg(pl.col("progression_date"))
    
    
    logger.info("Generating Criteria Details")
    criteria_df, evidence_criteria = get_criteria_results_rb(pat_id_df, serum_results, urine_results, free_light_chain_results, con_plasmacytoma_df, parquet_path,schema_type=schema_type, progression_dates=progression_dates)
   
    serum_results = serum_results.join(progression_dates, on="person_id", how="left")\
                                .filter(
                                           (pl.col("progression_date").is_null()) | 
                                                   (pl.col("measurement_date") > pl.col("progression_date").list.get(-1))
                                ).drop("progression_date")
    urine_results = urine_results.join(progression_dates, on="person_id", how="left")\
                                .filter(
                                           (pl.col("progression_date").is_null()) | 
                                                   (pl.col("measurement_date") > pl.col("progression_date").list.get(-1))
                                ).drop("progression_date")
    
    free_light_chain_results = free_light_chain_results.join(progression_dates, on="person_id", how="left")\
                                .filter(
                                           (pl.col("progression_date").is_null()) | 
                                                   (pl.col("measurement_date") > pl.col("progression_date").list.get(-1))
                                ).drop("progression_date")
    
    progression_dates = progression_dates.with_columns(
                            prg_cnt_2_years = pl.col("progression_date").list.lengths()
                        )
    
    #Calculate Nadir for the lab tests
    logger.info("Calculating the Nadir for serum:")
    serum_results = calculate_nadir(serum_results)
    logger.info("Calculating the Nadir for Urine:")
    urine_results = calculate_nadir(urine_results)
    logger.info("Calculating the Nadir for FLC:")
    free_light_chain_results = calculate_nadir(free_light_chain_results)
    logger.info("Calculating the Nadir Completed")
    
    pat_all_tests = pl.concat([
        urine_results,
        serum_results,
        free_light_chain_results  
    ], how="diagonal" )
    
    #print("Evidence Criteria\n", evidence_criteria)
    
    if is_training or (app_environment == 'C3'):
        logger.debug("Fetching Last Activity Recorded")
        activity_df = get_latest_activity_date(activity_date_fields,pat_ids_list, parquet_path=parquet_path,schema_type=schema_type)
 
        logger.info(f"CriteriaDF with Final selection - {criteria_df.filter(pl.col('final_selection') == 1).shape}")

        criteria_df = criteria_df.join( activity_df, 
                    on = "person_id",
                    how = 'left')

        criteria_df = criteria_df.with_columns(
            pl.when(pl.col("final_selection") == 1)\
                .then(pl.col("min_date"))\
                .otherwise(pl.col("last_activity_date"))
                .alias('last_activity_date')
        )
        today = datetime.date.today()

        criteria_df = criteria_df.with_columns(
                pl.when(pl.col('last_activity_date') > today)
                .then(today)
                .otherwise(pl.col('last_activity_date'))
                .alias('last_activity_date')
            )

        logger.info("Generating Random Point")
        random.seed(123)
        
        criteria_df = generate_random_point(criteria_df, pat_id_df )
        criteria_df = criteria_df.with_columns(
            label = (pl.col('diff')-pl.col('random_point'))
        )
    else:
        criteria_df = set_index_point(criteria_df, pat_id_df)

    logger.info("Lab Tests Feature Extraction")
    
    ##### Lab Test Groupping ####
    #lab_dict.update({'serum_free_light_kappa':['36916-5','11050-2'],'serum_free_light_lambda':['33944-0','11051-0']})
    #TODO: Remove lab_dict dependencies
    pat_all_tests = pat_all_tests.with_columns(
                                    pl.when(pl.col('concept_code').is_in(serum_test_code["concept_codes"]))
                                    .then(pl.lit('0'))
                                    .when(pl.col('concept_code').is_in(urine_test_code["concept_codes"]))
                                    .then(pl.lit('1'))
                                    .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_kappa']))
                                    .then(pl.lit('3'))
                                    .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_lambda']))
                                    .then(pl.lit('4'))
                                    .alias('concept_code')
    )
    ##### Lab Test Groupping ####
    
    lab_test_df_delta_feature = lab_test_feature_extraction(pat_all_tests, criteria_df,schema_type=schema_type)
    

    run_labs = lab_test_df_delta_feature['concept_code'].unique()
    final_stat_df = pl.DataFrame({'person_id': list(set(pat_ids_list))})

    #feat_lab_list = set(lab_dict['m_protein_in_serum'] + lab_dict['m_protein_in_urine'] + lab_dict['ca'] + lab_dict['serum_free_light'])           
    feat_lab_list = list(set(lab_class_map.keys()))  #Lab Test Groupping
        
    latest_test_result = get_latest_test_data(pat_all_tests, criteria_df, feat_lab_list,schema_type=schema_type)
    latest_data = utils.get_latest_data(pat_all_tests, ["person_id", "concept_code"], "measurement_date", descending=True)
    evidence_latest_data = latest_data.select(["person_id", "measurement_date", "concept_code", "value_as_number"])\
                                        .with_columns(doc_source=pl.lit("Lab"))\
                                        .with_columns(
                                            pl.when(pl.col("concept_code") == "0")
                                            .then(pl.lit("m-protein_serum"))
                                            .when(pl.col("concept_code") == "1")
                                            .then(pl.lit("m-protein_urine"))
                                            .when(pl.col("concept_code") == "3")
                                            .then(pl.lit("K-FLC"))
                                            .when(pl.col("concept_code") == "4")
                                            .then(pl.lit("L-FLC"))
                                            .otherwise(pl.lit("other"))
                                            .alias("measure")
                                        )\
                                        .rename({
                                            "measurement_date": "assessment_date",
                                            "value_as_number": "value"
                                        }).drop("concept_code")

    final_stat_df = final_stat_df.join(latest_test_result, on="person_id", how='outer')\
                                .join(
                                        progression_dates.select(["person_id", "prg_cnt_2_years"]),
                                          on="person_id", how='outer')\
                                .with_columns(
                                    pl.col("prg_cnt_2_years").fill_null(0)
                                )


    print("Final Stat Columns", final_stat_df.columns)
    logger.info(f"final_stat_df - {final_stat_df.shape} - {len(list(set(final_stat_df['person_id'].to_list())))}")

    logger.info("Generating Lab Statistics features")
    stats = [] #test_code_map.keys()
    for lab in list(stats):
        logger.debug("Processing Lab Feature:" + test_code_map[lab])
        if lab in run_labs:
            statistics_df = calculate_statistics(lab_test_df_delta_feature, lab,schema_type=schema_type)
            final_stat_df = final_stat_df.join(statistics_df, on="person_id", how="left")
        else:
            for index in [0,1,2]:
                lab_param = test_code_map[lab] + '_' + str(index)
                final_stat_df = final_stat_df.with_columns(
                                        pl.lit(None).cast(pl.Float64).alias(f'Mean_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Median_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Minimum_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Maximum_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'25th_Percentile_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'75th_Percentile_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Range_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Mean_delta_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Minimum_delta_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Maximum_delta_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'SD_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'slope_{lab_param}')
                                   )
            
            for stat in ["Mean_", "Minimum_", "Maximum_"]:
                final_stat_df = final_stat_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{stat}{test_code_map[lab]}_1_to_0_per_change"),
                                                pl.lit(None).cast(pl.Float64).alias(f"{stat}{test_code_map[lab]}_2_to_1_per_change"))
        
    logger.info("Lab Statistics FE Completed")
    
    
    logger.info("Metastatic Date Extraction Strated")
    met_df, evidence_met = get_met_dates("mm", pat_ids_list, parquet_path=parquet_path,schema_type=schema_type)
    
    
    logger.info("Disease Status Data Extaction Started")
    disease_status_df = get_disease_status_data(pat_ids_list, parquet_path,schema_type=schema_type)
    if disease_status_df is None:
        disease_status_df = pl.DataFrame(schema=[
                                                    ('person_id', pl.Utf8 if isinstance(pat_ids_list[0], str) else pl.Int64), 
                                                    ('assessment_date', pl.Date),
                                                    ('assessment_value_standard_name', pl.Utf8), 
                                                    ('assessment_name_standard_name', pl.Utf8),
                                                    ('source', pl.Utf8),
                                                    ('disease_status_doc_id', pl.Utf8)
                                                ])
    disease_status_df = disease_status_df.select(['person_id', 'assessment_date', 
                                'assessment_value_standard_name', 'assessment_name_standard_name', 'source', 'disease_status_doc_id'])
   
    disease_status_df = disease_status_df.filter(pl.col('assessment_date').is_not_null())\
                            .unique()
    
    tp_nlp_data = disease_status_df.filter(pl.col('assessment_value_standard_name') == "Tumor progression")\
                                        .sort(["person_id", 'assessment_date'], descending=True)\
                                        .unique(subset=["person_id"], keep = "first")\
                                        .select(["person_id", "assessment_date", "source", "disease_status_doc_id"])\
                                        .with_columns(
                                            pl.col("assessment_date").cast(pl.Date)
                                        )
    evidence_tp_nlp = tp_nlp_data.with_columns(
                            value_name = pl.lit("Tumor progression"),
                            doc_source = pl.col("source") + pl.lit(':') + pl.col('disease_status_doc_id').cast(pl.Utf8),
                            measure = pl.lit("Tumor Response Status")
                        ).drop(["source", "disease_status_doc_id"])
    
    evidence = pl.concat([evidence_latest_data, evidence_tp_nlp, evidence_criteria, evidence_met], how="diagonal")
    
    logger.info(f"final_stat_df - {final_stat_df.shape} - {len(list(set(final_stat_df['person_id'].to_list())))}")
    logger.info(f"criteria_df - {criteria_df.shape} - {len(list(set(criteria_df['person_id'].to_list())))}")
    if not is_training: 
        final_stat_df = final_stat_df.join(
                                criteria_df.select('person_id','latest_pr_date', 'min_date'),
                                on='person_id',
                                how='outer')\
                        .join(tp_nlp_data.select(["person_id", "assessment_date"]), on='person_id', how="left")\
                        .drop(final_stat_df.select(pl.col(pl.List)).columns)
    else:
        final_stat_df = final_stat_df.join(criteria_df.select('person_id','label','final_selection', 'latest_pr_date', 'min_date'),on='person_id',how='outer')\
                        .join(tp_nlp_data.select(["person_id", "assessment_date"]), on='person_id', how="left")\
                        .with_columns(pl.col("final_selection").cast(pl.Boolean))\
                        .drop(final_stat_df.select(pl.col(pl.List)).columns)
    

    #Add Metstat Details
    final_stat_df = final_stat_df.join(met_df, on ="person_id", how="left")
    final_stat_df = final_stat_df.fill_nan(None)

    return final_stat_df, evidence
    
    
def generate_features_sc_v2(cohort, pat_ids_list=None, parquet_path=s3_parquet_base_path, schema_type=None, configurations=PanSolidsConfigurations()):
    
    #print("Validating", configurations.stage_group_mapper)
    refresh_date = datetime.datetime.today()    #Optional: This can be parameterized
    logger.info("Parquet Path: "+parquet_path)
    print("Cohort Type", cohort)
    if (app_environment == 'C3'):
        refresh_date = datetime.date(2020, 6, 1)
    
    if cohort.lower() in ["nsclc", "sclc"]:
        diagnosis_codes =  get_diagnosis_codes("lung")
    elif cohort.lower() == "ovarian":
        diagnosis_codes =  get_diagnosis_codes("ofpc")
    else:
        diagnosis_codes = get_diagnosis_codes(cohort.lower())
    
    logger.info("Fetching Patient Details")
    pat_id_df = get_patient_diagnosis(diagnosis_codes, parquet_path, dx_year=None, schema_type=schema_type)
    if pat_ids_list is None:
        pat_ids_list = pat_id_df['person_id'].to_list()
        pat_ids_list = list(set(pat_ids_list))
    else:
        pat_id_df = pat_id_df.filter(pl.col('person_id').is_in(pat_ids_list))
    
    pat_id_df = pat_id_df.select(['person_id','condition_start_date'])\
                        .sort(by =['person_id','condition_start_date'],descending=False, nulls_last=True)\
                        .unique(subset=['person_id'],keep='first')
    
    print(pat_id_df.head)

    pat_id_df = pl.concat([
                            pat_id_df, 
                            pl.DataFrame({"person_id": pat_ids_list}).with_columns(condition_start_date = pl.lit("2000-1-1").str.strptime(pl.Date, "%Y-%m-%d").cast(pl.Date)) 
                        ], how="diagonal").sort(by =['person_id','condition_start_date'],descending=True, nulls_last=True)\
                        .unique(subset=['person_id'],keep='first')
    #print(pat_id_df)
    
    logger.info(f"Total Patients Eligbile {pat_id_df.shape}")
    pat_id_df = pat_id_df.with_columns(pl.lit(refresh_date).alias('random_point').cast(pl.Date))
    #pat_id_df (X,3) -> ['person_id','condition_start_date','random_point']
    
    logger.info("Fetching Medication Details")
    #Medication Feature Extraction:
    med_list = set(configurations.get_cancer_specific_variables()["medications"])
    medication_df = get_medication_details(pat_ids_list, med_list, parquet_path=parquet_path,schema_type=schema_type)
    
    medication_df = medication_df.with_columns(
                        pl.col('drug_exposure_end_date').cast(pl.Date)
                    )
    medication_df = medication_df.join(
                                        pat_id_df, 
                                        on="person_id", 
                                        how="inner"
                    )
    medication_df = medication_df.filter(
                           (((pl.col('drug_exposure_start_date') - pl.col('random_point')).dt.days()) >= -90)
                    )
    
    evidence_medicaiton = medication_df.select(["person_id", "drug_exposure_start_date", "concept_name"])\
                                        .rename({"drug_exposure_start_date": "assessment_date", "concept_name": "measure"})\
                                        .with_columns(
                                            value = pl.lit(None).cast(pl.Float64),
                                            doc_source = pl.lit("Medication"),
                                            assessment_date = pl.col("assessment_date").cast(pl.Date)
                                        )
    
    #print("Evidence Medication:", evidence_medicaiton)
    
    #Initialize Medication Feature
    med_feature = pl.DataFrame({"person_id": pat_ids_list})
    med_feature = med_feature.with_columns([
                    pl.lit(0).cast(pl.Int32).alias(m) for m in med_list
                ])
    
    med_feature = med_feature.join(
                        medication_df.groupby('person_id').agg(pl.col('concept_name')), 
                        on="person_id", 
                        how="left"
                )
    
    for m in med_list:
        med_feature = med_feature.with_columns(
            pl.col('concept_name').list.contains(m).alias(m).cast(pl.Int32)
        )
    med_feature = med_feature.drop("concept_name")
    logger.info("Medication Feature Shape:"+ str(med_feature.shape))
    #med_feature(X, med_list.length+1) -> ['person_id', med_list[0], med_list[1],...]
    
    
        
    logger.info("Fetching Staging Details")
    #Staging Feature Extraction
    staging_df = get_staging_data(pat_ids_list, parquet_path,schema_type=schema_type)
    staging_df = staging_df.join(pat_id_df, on="person_id", how="inner")\
                        .filter(
                            (pl.col('stage_date').is_not_null())
                        ).unique()
        
    staging_df = staging_df.with_columns(
            pl.col('stage_group_standard_name').str.to_lowercase().map_dict(
                {i.lower(): k for i, k in configurations.stage_group_mapper.items()}).cast(pl.Int64).alias('stage'),
            tstage = pl.lit(None).cast(pl.Int64),
            mstage = pl.lit(None).cast(pl.Int64),
            nstage = pl.lit(None).cast(pl.Int64)
        )
    print("Unsupported Stage Names:", staging_df.select(["stage_group_standard_name", "stage"]).unique().filter(pl.col("stage").is_nan())["stage_group_standard_name"].to_list())
    for s in stage_mapper:
        staging_df = staging_df.with_columns(
                        tstage = pl.when(pl.col("tstage_standard_name").str.contains(s))
                            .then(pl.lit(stage_mapper[s]))
                            .otherwise(pl.col('tstage')),
                        mstage = pl.when(pl.col("mstage_standard_name").str.contains(s))
                            .then(pl.lit(stage_mapper[s]))
                            .otherwise(pl.col('mstage')),
                        nstage = pl.when(pl.col("nstage_standard_name").str.contains(s))
                            .then(pl.lit(stage_mapper[s]))
                            .otherwise(pl.col('nstage'))
                    )
        
    evidence_stage_name = staging_df.select(["person_id", "stage_date", "stage"])\
                                .sort(by=["person_id", "stage", "stage_date"], nulls_last=True)\
                                .unique("person_id", keep='first')\
                                .rename({"stage_date": "assessment_date", "stage": "value"})\
                                .with_columns(
                                            measure = pl.lit("stage"),
                                            doc_source = pl.lit("Lab"),
                                            value = pl.col("value").cast(pl.Float64),
                                            assessment_date = pl.col("assessment_date").cast(pl.Date)
                                )
    evidence_tstage = staging_df.select(["person_id", "stage_date", "tstage"])\
                                .sort(by=["person_id", "tstage", "stage_date"], nulls_last=True)\
                                .unique("person_id", keep='first')\
                                .rename({"stage_date": "assessment_date", "tstage": "value"})\
                                .with_columns(
                                            measure = pl.lit("tstage"),
                                            doc_source = pl.lit("Lab"),
                                            value = pl.col("value").cast(pl.Float64),
                                            assessment_date = pl.col("assessment_date").cast(pl.Date)
                                )
    evidence_mstage = staging_df.select(["person_id", "stage_date", "mstage"])\
                                .sort(by=["person_id", "mstage", "stage_date"], nulls_last=True)\
                                .unique("person_id", keep='first')\
                                .rename({"stage_date": "assessment_date", "mstage": "value"})\
                                .with_columns(
                                            measure = pl.lit("mstage"),
                                            doc_source = pl.lit("Lab"),
                                            value = pl.col("value").cast(pl.Float64),
                                            assessment_date = pl.col("assessment_date").cast(pl.Date)
                                )
    evidence_nstage = staging_df.select(["person_id", "stage_date", "nstage"])\
                                .sort(by=["person_id", "nstage", "stage_date"], nulls_last=True)\
                                .unique("person_id", keep='first')\
                                .rename({"stage_date": "assessment_date", "nstage": "value"})\
                                .with_columns(
                                            measure = pl.lit("nstage"),
                                            doc_source = pl.lit("Lab"),
                                            value = pl.col("value").cast(pl.Float64),
                                            assessment_date = pl.col("assessment_date").cast(pl.Date)
                                )
    evidence_stage = pl.concat([evidence_stage_name, evidence_tstage, evidence_mstage, evidence_nstage ])
    
    stage_feature = staging_df.select(['person_id','stage_date', 'stage', 'tstage', 'mstage', 'nstage'])
    stage_feature = stage_feature.with_columns(
                        pl.col('stage').fill_null(pl.col('stage').max().over('person_id')),
                        pl.col('tstage').fill_null(pl.col('tstage').max().over('person_id')),
                        pl.col('mstage').fill_null(pl.col('mstage').max().over('person_id')),
                        pl.col('nstage').fill_null(pl.col('nstage').max().over('person_id')),
                    )\
                    .with_columns(
                          pl.col('stage').fill_nan(pl.col('stage').max().over('person_id')),
                          pl.col('tstage').fill_nan(pl.col('tstage').max().over('person_id')),
                          pl.col('mstage').fill_nan(pl.col('mstage').max().over('person_id')),
                          pl.col('nstage').fill_nan(pl.col('nstage').max().over('person_id'))
                    ).sort(["person_id", "stage_date"]).unique(subset=['person_id'], keep='last') 
    #stage_feature(X, 6) -> ['person_id','stage_date', 'stage', 'tstage', 'mstage', 'nstage']
    logger.info("Staging Feature Shape:" + str( stage_feature.shape))
        
    #TODO: Remove after Testing
    if (app_environment == 'C3'):
        logger.info("Generating Dummy values for Disease Status")
        ds_feature = pl.DataFrame({"person_id": pat_ids_list})
        for wind in ['0_45', '45_180']:
            for status in ['Complete therapeutic response', 'Partial therapeutic response', 'Stable', 'Tumor progression']:
                ds_feature = ds_feature.with_columns(
                                pl.lit(0).alias(status+"_"+wind)
                            )
    else:
        
        logger.info("Fetching Disease Status Details")
        #Disease Status Feature Extraction
        disease_status_df = get_disease_status_data(pat_ids_list, parquet_path,schema_type=schema_type)
        if disease_status_df is None:
            disease_status_df = pl.DataFrame(schema=[
                                                    ('person_id', pl.Utf8 if isinstance(pat_ids_list[0], str) else pl.Int64), 
                                                    ('assessment_date', pl.Date),
                                                    ('assessment_value_standard_name', pl.Utf8), 
                                                    ('assessment_name_standard_name', pl.Utf8),
                                                    ('source', pl.Utf8),
                                                    ('disease_status_doc_id', pl.Utf8)
                                                ])
        disease_status_df = disease_status_df.select(['person_id', 'assessment_date', 
                                'assessment_value_standard_name', 'assessment_name_standard_name', 'source', 'disease_status_doc_id'])
        #disease_status_df = disease_status_df.filter(pl.col("assessment_name_standard_name") == 'Finding related to therapeutic response')
        
        disease_status_df = disease_status_df.filter(pl.col('assessment_date').is_not_null())\
                            .unique()
        tp_nlp_data = disease_status_df.filter(pl.col('assessment_value_standard_name') == "Tumor progression")\
                                        .sort(["person_id", 'assessment_date'], descending=True)\
                                        .unique(subset=["person_id"], keep = "first")\
                                        .select(["person_id", "assessment_date", "source", "disease_status_doc_id"])\
                                        .with_columns(
                                            pl.col("assessment_date").cast(pl.Date)
                                        )
        evidence_tp_nlp = tp_nlp_data.with_columns(
                            value_name = pl.lit("Tumor progression"),
                            doc_source = pl.col("source") + pl.lit(':') + pl.col('disease_status_doc_id').cast(pl.Utf8),
                            measure = pl.lit("Tumor Response Status")
                        ).drop(["source", "disease_status_doc_id"]) 
        
        
        
        disease_status_df = disease_status_df.join(pat_id_df, on = ['person_id'], how="inner")\
                            .with_columns(
                                diff = (pl.col('random_point') - pl.col('assessment_date')).dt.days()
                            )    
        disease_status_df = disease_status_df.with_columns(
                                pl.when(pl.col('diff')<= 45)
                                .then(pl.lit('0_45'))
                                .when(pl.col('diff')<= 180)
                                .then(pl.lit('45_180'))
                                .otherwise(pl.lit('na'))
                                .alias('random_window')
                            )
        
        evidence_ds = disease_status_df.filter(pl.col("random_window").is_in(["0_45", "45_180"]))\
                                        .select(["person_id", "assessment_date", "assessment_value_standard_name", "source", "disease_status_doc_id"])\
                                        .rename({"assessment_value_standard_name": "value_name"})\
                                        .with_columns(
                                            measure = pl.lit("Disease Status"),
                                            doc_source = pl.col("source") + pl.lit(':') + pl.col('disease_status_doc_id').cast(pl.Utf8),
                                            assessment_date = pl.col("assessment_date").cast(pl.Date),
                                            value = pl.lit(None).cast(pl.Float64)
                                        ).drop(["source", "disease_status_doc_id"])
        
        ds_feature = pl.DataFrame(disease_status_df['person_id'].unique())
        for wind in ['0_45', '45_180']:

            df_temp = disease_status_df.filter(pl.col('random_window') == wind)
            df_temp = df_temp.groupby(['person_id','assessment_value_standard_name'])\
                            .agg(pl.col('assessment_date').count())
            df_temp = df_temp.pivot( index = ['person_id'],
                              columns = 'assessment_value_standard_name',
                              values = ['assessment_date'])
            for status in ['Complete therapeutic response', 'Partial therapeutic response', 'Stable', 'Tumor progression']:
                if  status not in  df_temp.columns:
                    df_temp = df_temp.with_columns(
                                pl.lit(0).alias(status)
                            )
            df_temp = df_temp.rename({
                                        'Complete therapeutic response': 'Complete therapeutic response_wind_' + wind,
                                        'Partial therapeutic response': 'Partial therapeutic response_wind_' + wind,
                                        'Stable': 'Stable_wind_' + wind,
                                        'Tumor progression': 'Tumor progression_wind_' + wind
                          })
            df_temp = df_temp.fill_null(0)
            ds_feature = ds_feature.join(df_temp, on = ['person_id'], how = 'left').fill_nan(0)
            
    #ds_feature(X,assessment_value_list*2+1)
    logger.info("Disease Status Feature Shape:"+str(ds_feature.shape))
    
    #Biomarker Feature
    bm_list = configurations.get_cancer_specific_variables()["biomarkers"]
     
    
    biomarker_feature = pl.DataFrame({"person_id": pat_ids_list})
    evidence_biomarkers = None
    bm_codes_all = list()
    bm_names_all = list()
    for bm in bm_list:
        bm_codes_all = bm_codes_all + [i["concept_code"] for i in configurations.biomarkers["concept_codes"][bm]]
        bm_names_all = bm_names_all + configurations.biomarkers["synonyms"][bm]
    biomarker_data_all_bms = get_biomarker_data(pat_ids_list, bm_codes_all, bm_names_all, parquet_path, schema_type=schema_type)
    print(biomarker_data_all_bms)
    #print(biomarker_data_all_bms)
    for bm in bm_list:
        biomarker_label = 'biomarker_'+bm
        biomarker_data =  biomarker_data_all_bms.filter(
            pl.col("concept_code").is_in(
                [ i["concept_code"] for i in configurations.biomarkers["concept_codes"][bm] ]) |
            pl.col("biomarker_name").is_in(configurations.biomarkers["synonyms"][bm])
        )
        
        biomarker_data = biomarker_data.with_columns(
                            pl.when(pl.col("value").str.to_lowercase().is_in(configurations.bm_positive_value))\
                                    .then(pl.lit(1))\
                                    .when(pl.col("value").str.to_lowercase().is_in(configurations.bm_negative_value))\
                                    .then(pl.lit(-1))\
                                    .when(pl.col("value").str.to_lowercase().is_in(configurations.bm_unknown_value))\
                                    .then(pl.lit(0))\
                                    .otherwise(pl.lit(-99))
                                    .alias(biomarker_label)
                        )

        biomarker_data = biomarker_data.sort(by=["person_id", "value", "measurement_date"], descending=True)\
                    .unique(subset=["person_id"], keep="first")
        evidence_bm = biomarker_data.select(["person_id", "measurement_date", "value", biomarker_label])\
                                    .with_columns(
                                            measure = pl.lit(biomarker_label),
                                            doc_source = pl.lit("Lab"),
                                            measurement_date = pl.col("measurement_date").cast(pl.Date),
                                        )\
                                    .rename({"measurement_date": "assessment_date", "value": "value_name" })\
                                    .with_columns(
                                        value = pl.col(biomarker_label).cast(pl.Float64)
                                    ).drop(biomarker_label)
        biomarker_feature = biomarker_feature.join(
                                    biomarker_data.select(['person_id', biomarker_label]),
                                    on="person_id",
                                    how="left"
                            )
        evidence_biomarkers = pl.concat([evidence_biomarkers, evidence_bm]) if evidence_biomarkers is not None else evidence_bm
        #print(evidence_biomarkers)
    
    
    logger.info("Lab Test Feature Extraction Begins")
    ecog_code = configurations.get_cancer_specific_variables()["lab_test_codes"]['ecog']["concept_codes"]
    ecog_names = configurations.get_cancer_specific_variables()["lab_test_codes"]['ecog']["concept_names"]
    pat_result = get_patient_results(pat_ids_list, lab_test_codes=ecog_code, lab_test_names=ecog_names,parquet_path=parquet_path,schema_type=schema_type)\
                        .drop_nulls(subset='test_value_name')\
                        .with_columns(
                            value_as_number = pl.col('test_value_name').map_dict(configurations.ecog_grade_mapper)
                        ).drop_nulls(subset="value_as_number").drop("test_value_name")\
                        .unique(["person_id", "measurement_date"])
    
    
    
    pat_result = pat_result.join(pat_id_df, on="person_id", how="left")
    pat_result = pat_result.filter(
                                (pl.col('measurement_date') <= pl.col('random_point'))
                ).with_columns(
                        test_diff = (pl.col('measurement_date')-pl.col('condition_start_date')).dt.days(),
                        window_tag = pl.when((pl.col('random_point') - pl.col('measurement_date')).dt.days() <= 45)
                            .then(pl.lit("0_45"))
                            .when((pl.col('random_point') - pl.col('measurement_date')).dt.days() <= 180)
                            .then(pl.lit("45_180"))
                            .otherwise(pl.lit("others"))
                )
    pat_result = pat_result.filter(pl.col("window_tag") != "others").sort(["person_id", "measurement_date"])
    
    evidence_ecog = pat_result.select(['person_id', "measurement_date", "value_as_number"])\
                                .rename({"measurement_date": "assessment_date", "value_as_number": "value"})\
                                .with_columns(
                                    units = pl.lit(None), 
                                    measure = pl.lit("ECOG"), 
                                    doc_source = pl.lit('Lab'),
                                    assessment_date = pl.col("assessment_date").cast(pl.Date),
                                    value = pl.col("value").cast(pl.Float64)
                                )\
    
    
    lab_test_df_delta_feature = pat_result.groupby(['person_id']).agg(
                                        pl.col('value_as_number').diff().alias('delta_test_value_numeric'),
                                        pl.col('measurement_date').diff().dt.days().alias('delta_test_date'),
                                        pl.col('random_point'),
                                        pl.col('window_tag'),
                                        pl.col('value_as_number'),
                                        pl.col('test_diff')
                                ).explode(columns=['delta_test_value_numeric', 
                                                   'delta_test_date',
                                                   'random_point',
                                                   'window_tag',
                                                   'value_as_number','test_diff']
                                ).with_columns(
                                        delta = (pl.col('delta_test_value_numeric')/(pl.col('delta_test_date')+0.001))
                                )
    
    #AS ECOG is the only lab test, No for-loop inplace and groupped all the tests as `ecog`
    lab_test_df_delta_feature = lab_test_df_delta_feature.with_columns(concept_code = pl.lit("ecog"))
    lab_feature = calculate_statistics(lab_test_df_delta_feature, "ecog", indices=["0_45", "45_180"])
    
    
    logger.info("Fetching Imaging Details")
    #Imaging Feature Extraction
    #image_list = set(configurations.get_cancer_specific_variables()["imaging"])
    image_list = set(configurations.image_group_map.keys())
    
    imaging_df = get_imaging_data(pat_ids_list, image_list, configurations.image_group_map, parquet_path=parquet_path,schema_type=schema_type)
    imaging_df = imaging_df.join(pat_id_df, on="person_id", how="inner")\
                        .filter(
                            (pl.col('procedure_date') >= pl.col('condition_start_date')) &
                            (pl.col('procedure_date') <= pl.col('random_point'))
                        )
    #Initialize Imaging Feature
    imaging_feature = pl.DataFrame({"person_id": pat_ids_list})
    imaging_feature = imaging_feature.with_columns([
                        pl.lit(False).cast(pl.Boolean).alias(i) for i in set(configurations.image_group_map.values())
                    ])
    evidence_imaging = imaging_df.select(["person_id", "procedure_date", "concept_name"])\
                                    .with_columns(
                                        value = pl.lit(None).cast(pl.Float64),
                                        doc_source = pl.lit('Lab'),
                                        procedure_date = pl.col("procedure_date").cast(pl.Date)
                                    )\
                                    .rename({"procedure_date": "assessment_date",  "concept_name":"measure"})
    
    imaging_feature = imaging_feature.join(
                            imaging_df.groupby('person_id').agg(pl.col('concept_name')), 
                            on="person_id", 
                            how="left"
                        )
    for i in set(image_group_map.values()):
        imaging_feature = imaging_feature.with_columns(
                                pl.col('concept_name').list.contains(i).alias(i).cast(pl.Boolean)
                            )
    imaging_feature = imaging_feature.drop("concept_name")
    logger.info("Imaging Feature Shape:"+ str(imaging_feature.shape))
    #imaging_feature(X, image_categories+1) -> ['person_id', "CT", "PET", "MRI", "Radioisotope"]
    
    
    logger.info("Metastatic Date Extraction Strated")
    met_df, evidence_met = get_met_dates(cohort, pat_ids_list, parquet_path=parquet_path,schema_type=schema_type)
    
    evidence_cols = ["person_id", "assessment_date", "value", "measure", "doc_source"]
    
    
    evidence = pl.concat([
                            evidence_medicaiton.select(evidence_cols), 
                            evidence_stage.select(evidence_cols), 
                            evidence_ds.select(evidence_cols), 
                            evidence_ecog.select(evidence_cols), 
                            evidence_imaging.select(evidence_cols),
                            evidence_met,
                            evidence_biomarkers,
                            evidence_tp_nlp
                ], how="diagonal")
    latest_data = evidence.select(["person_id", "assessment_date"]).sort(["person_id", "assessment_date"]).unique("person_id", keep="first").rename({"assessment_date": "latest_date"})
    final_feature_df = pl.DataFrame({"person_id": pat_ids_list}).join(latest_data, on="person_id", how="left")
    final_feature_df = final_feature_df.with_columns(
                            cohort = pl.lit(cohort_map[cohort.lower()])
                        )
    final_feature_df = final_feature_df.join(med_feature, on="person_id", how="left")\
                                .join(imaging_feature, on="person_id", how="left")\
                                .join(stage_feature, on="person_id", how="left")\
                                .join(ds_feature, on="person_id", how="left")\
                                .join(lab_feature,on="person_id", how="left")\
                                .join(biomarker_feature,on="person_id", how="left")\
                                .join(met_df, on="person_id", how="left")\
                                .join(tp_nlp_data.select(["person_id", "assessment_date"]), on="person_id", how="left")\
                                
    
    final_feature_df = final_feature_df.rename({
        'Mean_delta_ecog_0_45' : 'Delta_mean_ecog_0_45',
        'Minimum_delta_ecog_0_45': 'Delta_min_ecog_0_45', 
        'Maximum_delta_ecog_0_45': 'Delta_max_ecog_0_45',  
        'Mean_delta_ecog_45_180': 'Delta_mean_ecog_45_180', 
        'Minimum_delta_ecog_45_180': 'Delta_min_ecog_45_180', 
        'Maximum_delta_ecog_45_180': 'Delta_max_ecog_45_180', 
    })
    print(final_feature_df.columns)
    return final_feature_df, evidence



def generate_features_mds(pat_ids_list=None, parquet_path=s3_parquet_base_path, is_training=False, data_traceability_wb=None, schema_type=None, configurations=MDSConfigurations()):
    """
    Feature Extraction for MDS(Myelodysplastic Syndromes) Specific cancer type patients
    """

    refresh_date = datetime.datetime.today() 

    logger.info("Feature Extraction for MDS begins")
    #Fetch Patient List of MM Cancer type(diagnosis_code)
    diagnosis_codes = get_diagnosis_codes("mds")
    #Temporary:
    #diagnosis_codes = ['C50.029', 'C50.819', 'C18.4', 'C50.312', 'C44.92', 'C82.89', 'C85.17', 'C50.419', 'C85.10', 'C85.88', 'C85.19', 'C50.929', 'C83.35', 'C34.12', 'C82.98', 'C50.311', 'C83.34', 'C85.23', 'C50.011', 'C18.9', 'C83.08', 'C43.72', 'C83.10', 'C34.10', 'C84.42', 'C85.91', 'C50.229', 'C50.112', 'C90.00', 'C82.08', 'C83.01', 'C85.15', 'C50.411', 'C44.622', 'C82.11', 'C50.912', 'C85.89', 'C34.01', 'C34.32', 'C85.99', 'C50.022', 'C16.8', 'C85.80', 'C84.91', 'C18.1', 'C18.8', 'C85.83', 'C82.93', 'C85.97', 'C83.50', 'C43.62', 'C50.211', 'C43.4', 'C82.04', 'C43.9', 'C82.09', 'C83.38', 'C83.51', 'C83.03', 'C34.81', 'C83.31', 'C64.2', 'C34.80', 'C44.729', 'C83.15', 'C44.229', 'C50.412', 'C83.18', 'C82.10', 'C83.09', 'C84.A9', 'C67.4', 'C44.629', 'C64.9', 'C85.90', 'C83.97', 'C82.99', 'C43.30', 'C34.11', 'C15.8', 'C20', 'C67.0', 'C82.80', 'C85.93', 'C83.37', 'C44.329', 'C82.19', 'C83.07', 'C83.00', 'C18.7', 'C15.5', 'C15.9', 'C22.8', 'C83.12', 'C25.9', 'C83.33', 'C50.511', 'C43.61', 'C34.00', 'C85.81', 'C90.01', 'C82.94', 'C67.9', 'D07.5', 'C18.5', 'C16.9', 'C82.00', 'C50.811', 'C25.4', 'C84.41', 'C67.2', 'C50.019', 'C43.39', 'C64.1', 'C83.87', 'C84.48', 'C44.722', 'C44.222', 'C85.82', 'C82.18', 'C16.6', 'C82.92', 'C82.40', 'C67.5', 'C82.49', 'C50.111', 'C83.30', 'C61', 'C67.1', 'C83.80', 'C16.2', 'C83.11', 'C50.119', 'C67.8', 'C50.812', 'C50.319', 'C44.42', 'C43.60', 'C34.90', 'C44.320', 'C34.91', 'C44.221', 'C50.911', 'C34.31', 'C83.89', 'C50.512', 'C18.3', 'C67', 'C85.13', 'C50.212', 'C34.2', 'C82.03', 'C34.92', 'C82.90', 'C16.3', 'C82.91', 'C43.59', 'C83.13', 'C82.01', 'C83.39', 'C84.A0', 'C34.02', 'C85.85', 'C15.3', 'C18.6', 'C83.70', 'C50.919', 'C82.88']
    if is_training:
        pat_id_df = get_patient_diagnosis(diagnosis_codes, parquet_path,schema_type=schema_type)
    else:
        pat_id_df = get_patient_diagnosis(diagnosis_codes, parquet_path, dx_year=None, schema_type=schema_type)

    if pat_ids_list is None:
        pat_ids_list = pat_id_df['person_id'].to_list()
        pat_ids_list = list(set(pat_ids_list))
    else:
        pat_id_df = pat_id_df.filter(pl.col('person_id').is_in(pat_ids_list))
          
    logger.info("Total Patients Targeted:" + str( len(pat_ids_list)))
    pat_id_df = pl.concat([
                            pat_id_df, 
                            pl.DataFrame({"person_id": pat_ids_list})\
                                .with_columns(
                                        condition_start_date = pl.lit("2000-1-1").str.strptime(pl.Date, "%Y-%m-%d")
                                ) 
                        ], how="diagonal").sort(by =['person_id','condition_start_date'],descending=True, nulls_last=True)\
                        .unique(subset=['person_id'],keep='first')
    
    pat_id_df = pat_id_df.select(['person_id','condition_start_date'])\
        .sort(by =['person_id','condition_start_date'],descending=False,nulls_last=True)\
        .unique(subset=['person_id'],keep='first')
    
    pat_id_df = pat_id_df.with_columns(pl.lit(refresh_date).alias('random_point').cast(pl.Date))
    
    #Fetch the Patient Test Results
    wbc_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["wbc"]
    hemoglobin_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["hemoglobin"]
    neutrophils_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["neutrophils"]
    platelets_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["platelets"]
    ferritin_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["ferritin"]
    blasts_count_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["blasts_count"]
    blasts_percent_test_code = configurations.get_cancer_specific_variables()["lab_test_codes"]["blasts_percent"]

    patient_test_codes = wbc_test_code["concept_codes"] \
                            + hemoglobin_test_code["concept_codes"] \
                            + neutrophils_test_code["concept_codes"] \
                            + platelets_test_code["concept_codes"] \
                            + ferritin_test_code["concept_codes"] \
                            + blasts_count_test_code["concept_codes"] \
                            + blasts_percent_test_code["concept_codes"]
    patient_test_names = wbc_test_code["concept_names"] \
                            + hemoglobin_test_code["concept_names"] \
                            + neutrophils_test_code["concept_names"] \
                            + platelets_test_code["concept_names"] \
                            + ferritin_test_code["concept_names"] \
                            + blasts_count_test_code["concept_names"] \
                            + blasts_percent_test_code["concept_names"]
    
    logger.info("Fetching Targeted Patients Lab Tests")
    #TODO: Update get_patient_results wrt lab_dict
    patient_test_df = get_patient_results(pat_ids_list, patient_test_codes, patient_test_names, parquet_path=parquet_path, schema_type=schema_type)\
                        .drop_nulls(subset='value_as_number')
    
    patient_test_df = patient_test_df.join(pat_id_df, on="person_id")\
                        .filter(
                            pl.col("measurement_date")>=pl.col("condition_start_date")
                        )\
                        .drop("condition_start_date")\
                        .filter((refresh_date - pl.col('measurement_date')).dt.days() <= 180)

    logger.debug("Processing Lab Tests")
    wbc_results = get_test_results(
                        patient_test_df, 
                        wbc_test_code["concept_codes"],
                        wbc_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'wbc' 
                )
    hemoglobin_results = get_test_results(
                        patient_test_df, 
                        hemoglobin_test_code["concept_codes"],
                        hemoglobin_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'hemoglobin_in_blood' 
                )
    neutrophils_results = get_test_results(
                        patient_test_df, 
                        neutrophils_test_code["concept_codes"],
                        neutrophils_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'neutrophils_count' 
                )
    platelets_results = get_test_results(
                        patient_test_df, 
                        platelets_test_code["concept_codes"],
                        platelets_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'platelets' 
                )
    ferritin_results = get_test_results(
                        patient_test_df, 
                        ferritin_test_code["concept_codes"],
                        ferritin_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'ferritin' 
                )
    blasts_count_results = get_test_results(
                        patient_test_df, 
                        blasts_count_test_code["concept_codes"],
                        blasts_count_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'blasts_count' 
                )
    blasts_percent_results = get_test_results(
                        patient_test_df, 
                        blasts_percent_test_code["concept_codes"],
                        blasts_percent_test_code["concept_names"], 
                        schema_type=schema_type, 
                        default_key = 'blasts_percent' 
                )

    logger.debug("Lab Tests Extraction Completed")

    all_tests = {
        "wbc": wbc_results,
        "hemoglobin": hemoglobin_results,
        "neutrophils": neutrophils_results,
        "platelets": platelets_results,
        "blastscount": blasts_count_results,
        "blastspercent": blasts_percent_results,
        "ferritin": ferritin_results
    }
    
    final_stat_df = pl.DataFrame({'person_id': list(set(pat_ids_list))}).with_columns(pl.col("person_id").cast(pl.Utf8))
    for lab_test_code, lab_test in all_tests.items():
            print("Processing ", lab_test_code)
            window_tag = utils.window_tagger(lab_test, "measurement_date", [45,180], refresh_date, filter_outside_range=True)\
                                .with_columns(
                                                pl.when(pl.col("window_tag") == 0)\
                                                    .then(pl.lit("0_45"))\
                                                    .when(pl.col("window_tag") == 1)\
                                                    .then(pl.lit("45_180"))\
                                                    .otherwise(pl.lit("180+"))\
                                                    .alias("window_tag")
                
                                    )

            lab_test_df_delta_feature = window_tag\
                                .join(pat_id_df, on="person_id", how="left")\
                                .with_columns(
                                    test_diff = (pl.col('measurement_date')-pl.col('condition_start_date')).dt.days(),
                                )\
                                .groupby(['person_id', "concept_code"]).agg(
                                            pl.col('value_as_number').diff().alias('delta_test_value_numeric'),
                                            pl.col('measurement_date').diff().dt.days().alias('delta_test_date'),
                                            pl.col('random_point'),
                                            pl.col('window_tag'),
                                            pl.col('value_as_number'),
                                            pl.col('test_diff')
                                    ).explode(columns=['delta_test_value_numeric', 
                                                    'delta_test_date',
                                                    'random_point',
                                                    'window_tag',
                                                    'value_as_number','test_diff']
                                    ).with_columns(
                                            delta = (pl.col('delta_test_value_numeric')/(pl.col('delta_test_date')+0.001))
                                    )    
            lab_codes = lab_test_df_delta_feature["concept_code"].unique().to_list()
            group_features = None
            for code in lab_codes:
                test_code_map[code] = f"{lab_test_code}_{code}"
                lab_feature = calculate_statistics(lab_test_df_delta_feature, code, indices=["0_45", "45_180"])\
                                .select( pl.col("person_id"),
                                pl.selectors.starts_with("Mean_"), 
                                pl.selectors.starts_with("Minimum_"),
                                pl.selectors.starts_with("Maximum_"),
                                pl.selectors.starts_with("Skewness_"),
                                pl.selectors.starts_with("Kurtosis_"),
                                pl.selectors.starts_with("slope_"),
                            )
                group_features = group_features.join(lab_feature, on="person_id", how="outer") if group_features is not None else lab_feature
        
            if (group_features is None )or( group_features.is_empty()):
                print("DF not available", group_features, )
                group_features = pl.DataFrame({"person_id": pat_ids_list})
                df_schema = []
                for wi in ["0_45", "45_180"]:
                    df_schema += [(f"Mean_{lab_test_code}_{wi}", pl.Float32),
                                (f"Minimum_{lab_test_code}_{wi}", pl.Float32),
                                (f"Maximum_{lab_test_code}_{wi}", pl.Float32),
                                (f"Mean_delta_{lab_test_code}_{wi}", pl.Float32),
                                (f"Skewness_{lab_test_code}_{wi}", pl.Float32),
                                (f"Kurtosis_{lab_test_code}_{wi}", pl.Float32),
                                (f"slope_{lab_test_code}_{wi}", pl.Float32)
                                ]
                group_features = group_features.join(
                    pl.DataFrame(schema=[('person_id', pl.Utf8)]+df_schema),
                    how = "outer",
                    on = "person_id"
                )
            else:   
                for wi in ["0_45", "45_180"]:
                    #print(group_features.columns, wi)
                    group_features = group_features.with_columns(
                        pl.concat_list(
                                            [f"Mean_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"Mean_{lab_test_code}_{wi}"),
                        pl.concat_list(
                                            [f"Minimum_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"Minimum_{lab_test_code}_{wi}"),
                        pl.concat_list(
                                            [f"Maximum_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"Maximum_{lab_test_code}_{wi}"),
                        pl.concat_list(
                                            [f"Mean_delta_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"Mean_delta_{lab_test_code}_{wi}"),
                        pl.concat_list(
                                            [f"Skewness_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"Skewness_{lab_test_code}_{wi}"),
                        pl.concat_list(
                                            [f"Kurtosis_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"Kurtosis_{lab_test_code}_{wi}"),
                        pl.concat_list(
                                            [f"slope_{lab_test_code}_{lc}_{wi}" for lc in lab_codes]
                                    ).apply(lambda x: [i for i in x if i]).list.mean().alias(f"slope_{lab_test_code}_{wi}"),
                    )
            final_stat_df = final_stat_df.join(group_features, on="person_id", how="left")
    
    # Support of Disease Status: Duplication from the other feature generation models with the evidences
    disease_status_df = get_disease_status_data(pat_ids_list, parquet_path,schema_type=schema_type)
    if disease_status_df is None:
        disease_status_df = pl.DataFrame(schema=[
                                                    ('person_id', pl.Utf8 if isinstance(pat_ids_list[0], str) else pl.Int64), 
                                                    ('assessment_date', pl.Date),
                                                    ('assessment_value_standard_name', pl.Utf8), 
                                                    ('assessment_name_standard_name', pl.Utf8),
                                                    ('source', pl.Utf8),
                                                    ('disease_status_doc_id', pl.Utf8)
                                                ])
    disease_status_df = disease_status_df.select(['person_id', 'assessment_date', 
                                'assessment_value_standard_name', 'assessment_name_standard_name', 'source', 'disease_status_doc_id'])
   
    disease_status_df = disease_status_df.filter(pl.col('assessment_date').is_not_null())\
                            .unique()
    
    tp_nlp_data = disease_status_df.filter(pl.col('assessment_value_standard_name') == "Tumor progression")\
                                        .sort(["person_id", 'assessment_date'], descending=True)\
                                        .unique(subset=["person_id"], keep = "first")\
                                        .select(["person_id", "assessment_date", "source", "disease_status_doc_id"])\
                                        .with_columns(
                                            pl.col("assessment_date").cast(pl.Date)
                                        )
    evidence_tp_nlp = tp_nlp_data.with_columns(
                            value_name = pl.lit("Tumor progression"),
                            doc_source = pl.col("source") + pl.lit(':') + pl.col('disease_status_doc_id').cast(pl.Utf8),
                            measure = pl.lit("Tumor Response Status")
                        ).drop(["source", "disease_status_doc_id"])
    
    evidence = evidence_tp_nlp

    final_stat_df = final_stat_df.join(tp_nlp_data.select(["person_id", "assessment_date"]), on="person_id", how="left")\
                    .fill_nan(None)

    return final_stat_df, evidence
