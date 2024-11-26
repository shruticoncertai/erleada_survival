import datetime
import json
import logging
import numpy as np
import polars as pl
import random
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
from utils import data_loader
from utils import table_details as tables
from utils import utils
from unit_converter.convert import *
from PA_GE_test.rule_execution import rules_check
from datetime import timedelta

from patient_availability_model import *


logger = utils.setup_logger("feature_generator")

def generate_features(pat_ids_list=None, parquet_path=s3_parquet_base_path, is_training=False, data_traceability_wb=None, schema_type=None):
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
        #.drop_nulls()
    
    ######## RULE CHECK #######
    if data_traceability_wb:
        rules_check(
                pat_id_df.rename(
                        {
                            'person_id': 'chai_patient_id', 
                            'condition_start_date': 'diagnosis_date'
                        }
                ), 
                'condition', 
                os.environ.get("SCHEMA_NAME", "omop"),
                data_traceability_wb )

    ########  ########  #######

    #Fetch the Patient Test Results
    serum_test_code = cancer_specific_constants["multiple_myeloma"]["lab_test_codes"]["m_protein_serum"]
    urine_test_code = cancer_specific_constants["multiple_myeloma"]["lab_test_codes"]["m_protein_urine"]
    calcium_test_code = cancer_specific_constants["multiple_myeloma"]["lab_test_codes"]["calcium_test"]
    free_light_chain_test_code = cancer_specific_constants["multiple_myeloma"]["lab_test_codes"]["free_light_chain_reaction"]
    bone_marrow_test_code = cancer_specific_constants["multiple_myeloma"]["lab_test_codes"]["bone_marrow_test"]
    other_lab_test_names = cancer_specific_constants["multiple_myeloma"]["lab_test_names"] #TODO: Use test codes than the names for commonality
    patient_test_codes = serum_test_code + urine_test_code + calcium_test_code + free_light_chain_test_code + bone_marrow_test_code
    
    logger.info("Fetching Targeted Patients Lab Tests")
    patient_test_df = get_patient_results(pat_ids_list, patient_test_codes, other_lab_test_names, parquet_path=parquet_path, schema_type=schema_type)

    ########## RULE CHECK ##########
    if data_traceability_wb:
        rules_check(
            patient_test_df.rename({
                    'concept_code': 'test_name_standard_code', 
                    'measurement_date': 'test_date',
                    'value_as_number': 'test_value_numeric',
                    'measurement_unit_source_name': 'test_unit_source_name',
                    'person_id': 'chai_patient_id'}),
            'patient_test_raw',
            os.environ.get("SCHEMA_NAME", "omop"),
            data_traceability_wb)

    ####### **************** #######

    logger.debug("Processing Lab Tests")
    serum_results = get_test_results(patient_test_df, serum_test_code,schema_type=schema_type)
    urine_results = get_test_results(patient_test_df, urine_test_code,schema_type=schema_type)
    calcium_results = get_test_results(patient_test_df, calcium_test_code,schema_type=schema_type)
    free_light_chain_results = get_test_results(patient_test_df, free_light_chain_test_code,schema_type=schema_type)
    other_lab_tests = get_lab_test_results(patient_test_df, other_lab_test_names,schema_type=schema_type)
    logger.debug("Lab Tests Extraction Completed")

    pat_all_tests = pl.concat([
        urine_results,
        serum_results,
        free_light_chain_results,
        calcium_results,
        other_lab_tests   
    ])
    
    bone_marrow_results = patient_test_df.filter(
                                pl.col("concept_code").is_in(bone_marrow_test_code)
                        )

    ########### RULE CHECK ###############
    
    if data_traceability_wb:
        lab_test_df = convert_unit(patient_test_df,schema_type=schema_type)
        lab_test_df = lab_test_df.sort(['person_id','measurement_date','value_as_number'],descending=False)\
            .unique(subset=['person_id','measurement_date','concept_code'],keep='first')

        rules_check(
            lab_test_df.rename({
                    'concept_code': 'test_name_standard_code', 
                    'measurement_date': 'test_date',
                    'value_as_number': 'test_value_numeric',
                    'measurement_unit_source_name': 'test_unit_source_name',
                    'person_id': 'chai_patient_id'}),
            'patient_test_std',          
            os.environ.get("SCHEMA_NAME", "omop"),
            data_traceability_wb
        )

    ######### ****************** #########
                                                            #Skipped as feature is no-longer-in-use
    #logger.debug("Fetching Demographics")
    #demographics = get_demographics_details(pat_ids_list,parquet_path=parquet_path)
    #demographics = demographics.drop("date_of_death")

    ########### RULE CHECK ###############
    #if data_traceability_wb:
    #    patient_column_names = tables.get_column_name("patient")
    #    patient_lazy_df = data_loader.get_table_df("patient", parquet_base_path=parquet_path,schema=schema_type )
    #    patient_df = patient_lazy_df.filter(
    #                    pl.col(patient_column_names["chai_patient_id"]).is_in(pat_ids_list)
    #                ).collect()
    #    patient_df = patient_df.rename({'concept_name': 'gender'})
    #    patient_df = patient_df.with_columns( pl.col('source_age').cast(int))

    #    rules_check(
    #        patient_df,
    #        'patient',
    #        os.environ.get("SCHEMA_NAME", "omop"),
    #        data_traceability_wb
    #    )

    ######### ****************** #########  
    

    logger.info("Generating Criteria Details")
    criteria_df = get_criteria_results_rb(pat_id_df, serum_results, urine_results, free_light_chain_results, calcium_results, parquet_path,schema_type=schema_type)

    if is_training or (app_environment == 'C3'):
        logger.debug("Fetching Last Activity Recorded")
        activity_df = get_latest_activity_date(activity_date_fields,pat_ids_list, parquet_path=parquet_path,schema_type=schema_type)
 
        #print("CriteriaDF with Final selection", criteria_df.filter(pl.col("final_selection") == 1).shape)
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

    
    
    
    ###########################################
             #Skipped as feature is no-longer-in-use
    #logger.info("Tumor Grade Feature Extraction")
    #tumor_exam_feat_df = tumor_grade_feature_extraction(pat_ids_list, criteria_df, parquet_path=parquet_path)
    
    #################################################
    
    #logger.info("Fetching Surgery Data")
    #surgery_list = cancer_specific_constants["multiple_myeloma"]["surgeries"]
    #surgery_df= get_surgery_details(pat_ids_list, surgery_list, parquet_path=parquet_path,schema_type=schema_type)
    
    #Bone Marrow Feature Extraction
    #logger.info("Bone Marrow Feature Extraction")
    #bone_marrow_feature = bone_marrow_feature_extraction(bone_marrow_results, surgery_df, criteria_df, schema_type=schema_type)
    #print("Bone Marrow Tests:", bone_marrow_results.shape)
    #print("Surgery Data:", surgery_df.shape )
    #print("Bone Marrow Feature Results:", bone_marrow_feature.shape )
    #print("Bone Marrow Feature Stats:", bone_marrow_feature.describe())
    
    #logger.info("Fetching Imaging Data")
    #imaging_list = cancer_specific_constants["multiple_myeloma"]["imaging"]
    #imaging_df = get_imaging_data(pat_ids_list, imaging_list, parquet_path=parquet_path)
    
    #logger.info("Imaging Feature Extraction")
    #imaging_feature = imaging_feature_extraction(imaging_df, criteria_df)
    #print("Imaging Feature Results:", imaging_feature.shape)
    #print("Imaging Feature Stats:", imaging_feature.describe())
    

    ######### RULE CHECK ###########
    #if data_traceability_wb:
    #    pat_tumor_exam_df = get_tumor_exam_results(pat_ids_list, parquet_path=parquet_path,schema_type=schema_type)
    #    pat_tumor_exam_df = pat_tumor_exam_df.filter(pl.struct(['person_id','exam_date','concept_name']).is_not_null())
    #    pat_tumor_exam_df_rc = pat_tumor_exam_df.rename({'person_id': 'chai_patient_id', 'concept_name': 'tumor_grade_standard_name'})
    #    rules_check(
    #        pat_tumor_exam_df_rc,
    #        'tumor_exam',
    #        os.environ.get("SCHEMA_NAME", "omop"),
    #        data_traceability_wb
    #    )

    ###### **************** ########

            #Skipped as feature is no-longer-in-use
    #logger.info("Medication Grade Feature Extraction")
    #medication_feat_df   = medication_feature_extraction(pat_ids_list, criteria_df, parquet_path=parquet_path)

    ######### RULE CHECK ###########
    #if data_traceability_wb:
    #    mm_med_list = set(cancer_specific_constants["multiple_myeloma"]["medications"])
    #    medication_df = get_medication_details(pat_ids_list, mm_med_list, parquet_path=parquet_path,schema_type=schema_type)
    #    
    #    col_names = tables.column_names["omop"]["medication"]
    #    rename_cols = { col_names[e]: e for e in col_names }

    #    pat_medication_df_rc = medication_df.rename(rename_cols)

    #    rules_check(
    #        pat_medication_df_rc,
    #        'medication',
    #        os.environ.get("SCHEMA_NAME", "omop"),
    #        data_traceability_wb
    #    )

    ###### **************** ########

    logger.info("Lab Tests Feature Extraction")
    
    ##### Lab Test Groupping ####
    lab_dict.update({'serum_free_light_kappa':['36916-5','11050-2'],'serum_free_light_lambda':['33944-0','11051-0']})
    
    pat_all_tests = pat_all_tests.with_columns(
                                    pl.when(pl.col('concept_code').is_in(lab_dict['m_protein_in_serum']))
                                    .then(pl.lit('0'))
                                    .when(pl.col('concept_code').is_in(lab_dict['m_protein_in_urine']))
                                    .then(pl.lit('1'))
                                    .when(pl.col('concept_code').is_in(lab_dict['ca']))
                                    .then(pl.lit('2'))
                                    .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_kappa']))
                                    .then(pl.lit('3'))
                                    .when(pl.col('concept_code').is_in(lab_dict['serum_free_light_lambda']))
                                    .then(pl.lit('4'))
                                    .alias('concept_code')
    )
    
    ##### Lab Test Groupping ####
    
    lab_test_df_delta_feature = lab_test_feature_extraction(pat_all_tests, criteria_df,schema_type=schema_type)
    

    
    
    #logger.info("Bone Marrow Feature Extraction Completed")

    run_labs = lab_test_df_delta_feature['concept_code'].unique()
    final_stat_df = pl.DataFrame({'person_id': list(set(pat_ids_list))})

    #feat_lab_list = set(lab_dict['m_protein_in_serum'] + lab_dict['m_protein_in_urine'] + lab_dict['ca'] + lab_dict['serum_free_light'])           
    feat_lab_list = list(set(lab_class_map.keys()))  #Lab Test Groupping
        
        
    latest_test_result = get_latest_test_data(pat_all_tests, criteria_df, feat_lab_list,schema_type=schema_type)

    final_stat_df = final_stat_df.join(latest_test_result, on="person_id", how='outer')

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
    logger.info(f"final_stat_df - {final_stat_df.shape} - {len(list(set(final_stat_df['person_id'].to_list())))}")
    #logger.info(f"medication_feat_df - {medication_feat_df.shape} - {len(list(set(medication_feat_df['person_id'].to_list())))}")
    #logger.info(f"tumor_exam_feat_df - {tumor_exam_feat_df.shape} - {len(list(set(tumor_exam_feat_df['person_id'].to_list())))}")
    #logger.info(f"demographics - {demographics.shape} - {len(list(set(demographics['person_id'].to_list())))}")
    logger.info(f"criteria_df - {criteria_df.shape} - {len(list(set(criteria_df['person_id'].to_list())))}")

    final_stat_df = final_stat_df.join(criteria_df.select('person_id','label','final_selection', 'latest_pr_date', 'min_date'),on='person_id',how='outer')\
                        .with_columns(pl.col("final_selection").cast(pl.Boolean))\
                        .drop(final_stat_df.select(pl.col(pl.List)).columns)
#.join(tumor_exam_feat_df,left_on=['person_id'], right_on = ["person_id"], how='outer')\
#                       .join(medication_feat_df,left_on=['person_id'], right_on = ["person_id"],how='outer')\
#                       .join(demographics,left_on=['person_id'], right_on = ["person_id"],how='outer')\
    
    final_stat_df = final_stat_df.fill_nan(None)

    return final_stat_df

def generate_features_sc(cohort, pat_ids_list=None, parquet_path=s3_parquet_base_path, schema_type=None):
    
    
    refresh_date = datetime.datetime.today()    #Optional: This can be parameterized
    if (app_environment == 'C3'):
        refresh_date = datetime.date(2020, 6, 1)
    
    if cohort.lower() in ["nsclc", "sclc"]:
        diagnosis_codes =  get_diagnosis_codes("lung")
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
    
    pat_id_df = pat_id_df.with_columns(pl.lit(refresh_date).alias('random_point').cast(pl.Date))
    #pat_id_df (X,3) -> ['person_id','condition_start_date','random_point']
    
    logger.info("Fetching Medication Details")
    #Medication Feature Extraction:
    med_list = set(cancer_specific_constants["pan_solids"]["medications"])
    medication_df = get_medication_details(pat_ids_list, med_list, parquet_path=parquet_path,schema_type=schema_type)
    medication_df = medication_df.join(
                                        pat_id_df, 
                                        on="person_id", 
                                        how="inner"
                    ).filter(
                        (pl.col('drug_exposure_start_date') >= pl.col('condition_start_date')) &
                           (((pl.col('drug_exposure_end_date') - pl.col('random_point')).dt.days()) >= -90)
                    )
    
    #Initialize Medication Feature
    med_feature = pl.DataFrame({"person_id": pat_ids_list})
    med_feature = med_feature.with_columns([
                    pl.lit(False).cast(pl.Boolean).alias(m) for m in med_list
                ])
    
    med_feature = med_feature.join(
                        medication_df.groupby('person_id').agg(pl.col('concept_name')), 
                        on="person_id", 
                        how="left"
                )
    for m in med_list:
        med_feature = med_feature.with_columns(
            pl.col('concept_name').list.contains(m).alias(m).cast(pl.Boolean)
        )
    med_feature = med_feature.drop("concept_name")
    logger.info("Medication Feature Shape:"+ str(med_feature.shape))
    #med_feature(X, med_list.length+1) -> ['person_id', med_list[0], med_list[1],...]
    
    logger.info("Fetching Imaging Details")
    #Imaging Feature Extraction
    image_list = set(cancer_specific_constants["pan_solids"]["imaging"])
    image_group_map = {
                        'Computed tomography': "CT",
                        'Positron emission tomography with computed tomography': "PET",
                        'Positron emission tomography': "PET",
                        'MRI with contrast': "MRI",
                        'MRI without contrast': "MRI",
                        'Magnetic resonance imaging': "MRI",
                        'Radioisotope scan of bone': "Radioisotope"
                    }
    imaging_df = get_imaging_data(pat_ids_list, image_list, image_group_map, parquet_path=parquet_path,schema_type=schema_type)
    imaging_df = imaging_df.join(pat_id_df, on="person_id", how="inner")\
                        .filter(
                            
                            (pl.col('procedure_date') <= pl.col('random_point'))
                        )
    #Initialize Imaging Feature
    imaging_feature = pl.DataFrame({"person_id": pat_ids_list})
    imaging_feature = imaging_feature.with_columns([
                        pl.lit(False).cast(pl.Boolean).alias(i) for i in set(image_group_map.values())
                    ])
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
        
    logger.info("Fetching Staging Details")
    #Staging Feature Extraction
    staging_df = get_staging_data(pat_ids_list, parquet_path,schema_type=schema_type)
    staging_df = staging_df.join(pat_id_df, on="person_id", how="inner")\
                        .filter(
                            (pl.col('stage_date').is_not_null()) &
                            ((pl.col('random_point') - pl.col('stage_date')).dt.days() <=90)
                        ).unique()
        
    staging_df = staging_df.with_columns(
            pl.col('stage_group_standard_name').map_dict(stage_group_mapper, default=float("NaN"), return_dtype=pl.Int64).alias('stage'),
            tstage = pl.lit(None).cast(pl.Int64),
            mstage = pl.lit(None).cast(pl.Int64),
            nstage = pl.lit(None).cast(pl.Int64)
        )
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
        disease_status_df = disease_status_df.select(['person_id', 'assessment_date', 
                                'assessment_value_standard_name', 'assessment_name_standard_name'])
        #disease_status_df = disease_status_df.filter(pl.col("assessment_name_standard_name") == 'Finding related to therapeutic response')
        disease_status_df.filter(pl.col('assessment_date').is_not_null())
        disease_status_df = disease_status_df.unique()
        disease_status_df = disease_status_df.join(pat_id_df, on = ['person_id'], how="inner")\
                            .with_columns(
                                                diff = (pl.col('random_point') - pl.col('assessment_date')).dt.days()
                                        )    
        disease_status_df = disease_status_df.with_columns(
                                pl.when(pl.col('diff')<= 45)
                                .then(pl.lit('0_45'))
                                .when(pl.col('diff')<= 180)
                                .then(pl.lit('45_180'))
                                .otherwise('na')
                                .alias('random_window')
                            )
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
    tp53_codes = cancer_specific_constants["pan_solids"]["biomarker_codes"]
    biomarker_data = get_biomarker_data(pat_ids_list, tp53_codes, parquet_path, "omop")
    biomarker_data = biomarker_data.with_columns(
                        tp53_mutation = pl.when(
                                    pl.col("value").is_in(tp_53_positive)
                                ).then(pl.lit(1))\
                                .when(pl.col("value").is_in(tp_53_negative))\
                                .then(pl.lit(-1))\
                                .otherwise(pl.lit(0))     
                    )

    biomarker_data = biomarker_data.sort(by=["person_id", "measurement_date"], descending=True)\
                .unique(subset=["person_id"], keep="first")
    biomarker_feature = biomarker_data.select(['person_id', 'tp53_mutation'])
    
    #TODO: Use get_biomarker_data and generate features(Above commented)
    biomarker_feature = pl.DataFrame({'person_id': pat_ids_list})
    biomarker_feature = biomarker_feature.with_columns(
                            tp53_mutation = pl.lit(None).cast(pl.Int32)
                        )
    
    #Lab Test Feature
    logger.info("Lab Test Feature Extraction Begins")
    ecog_code = cancer_specific_constants["pan_solids"]["lab_test_codes"]['ecog']
    pat_result = get_patient_results(pat_ids_list, lab_test_codes=ecog_code, lab_test_names=[],parquet_path=parquet_path,schema_type=schema_type)
    pat_result = pat_result.join(pat_id_df, on="person_id", how="left")
    pat_result = pat_result.filter(
                                (pl.col('measurement_date') <= pl.col('random_point'))
                ).with_columns(
                        test_diff = (pl.col('measurement_date')-pl.col('condition_start_date')).dt.days(),
                        window_tag = pl.when((pl.col('random_point') - pl.col('measurement_date')).dt.days() <= 45)
                            .then(pl.lit(0))
                            .when((pl.col('random_point') - pl.col('measurement_date')).dt.days() <= 180)
                            .then(pl.lit(1))
                            .otherwise(pl.lit(-1))
                )
    pat_result = pat_result.filter(pl.col("window_tag")>-1)
    
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
    lab_feature = calculate_statistics(lab_test_df_delta_feature, "ecog")
    
    
    final_feature_df = pl.DataFrame({"person_id": pat_ids_list})
    final_feature_df = final_feature_df.with_columns(
                            cohort = pl.lit(cohort_map[cohort.lower()])
                        )
    final_feature_df = final_feature_df.join(med_feature, on="person_id", how="left")\
                                .join(imaging_feature, on="person_id", how="left")\
                                .join(stage_feature, on="person_id", how="left")\
                                .join(ds_feature, on="person_id", how="left")\
                                .join(lab_feature,on="person_id", how="left")\
                                .join(biomarker_feature,on="person_id", how="left")
    
    final_feature_df = final_feature_df.rename({
        'Mean_delta_ecog_0' : 'Delta_mean_89247-1_0',
        'Minimum_delta_ecog_0': 'Delta_min_89247-1_0', 
        'Maximum_delta_ecog_0': 'Delta_max_89247-1_0', 
        'Mean_ecog_0':'Mean_89247-1_0',
        'Median_ecog_0': 'Median_89247-1_0', 
        'SD_ecog_0':'SD_89247-1_0', 
        'Minimum_ecog_0': 'Minimum_89247-1_0',
        'Maximum_ecog_0': 'Maximum_89247-1_0', 
        'Skewness_ecog_0': 'Skewness_89247-1_0', 
        'Kurtosis_ecog_0': 'Kurtosis_89247-1_0', 
        '25th_Percentile_ecog_0': '25th_Percentile_89247-1_0',
        '75th_Percentile_ecog_0': '75th_Percentile_89247-1_0', 
        'Range_ecog_0': 'Range_89247-1_0', 
        'slope_ecog_0': 'slope_89247-1_0', 
        'Mean_delta_ecog_1': 'Delta_mean_89247-1_1', 
        'Minimum_delta_ecog_1': 'Delta_min_89247-1_1', 
        'Maximum_delta_ecog_1': 'Delta_max_89247-1_1', 
        'Mean_ecog_1': 'Mean_89247-1_1', 
        'Median_ecog_1': 'Median_89247-1_1', 
        'SD_ecog_1': 'SD_89247-1_1', 
        'Minimum_ecog_1': 'Minimum_89247-1_1', 
        'Maximum_ecog_1': 'Maximum_89247-1_1', 
        'Skewness_ecog_1': 'Skewness_89247-1_1', 
        'Kurtosis_ecog_1': 'Kurtosis_89247-1_1', 
        '25th_Percentile_ecog_1': '25th_Percentile_89247-1_1', 
        '75th_Percentile_ecog_1': '75th_Percentile_89247-1_1', 
        'Range_ecog_1': 'Range_89247-1_1', 
        'slope_ecog_1': 'slope_89247-1_1'
    })
    
    return final_feature_df
    
