import os

S3_PARQUET_BASE_LOCATION = "c0-dts-stag-rwe-spark/qcca_cdm_master_omop_latest_stage"
S3_PARQUET_CONCEPT_LOCATION = "c0-dts-stag-rwe-spark/omop_vocabulary/concept.parquet"
#ENVIRONMENT SSPECIFIC Variables

models_base_path = os.environ.get("MODEL_BASEPATH", "/app/models/")
predict_results_basepath = os.environ.get("RESULTS_BASEPATH", "/app/predict_results")

s3_prefix = os.environ.get("S3_PREFIX", 'availability')
start_year = int(os.environ.get("DATA_START_YEAR", 2020))
s3_schema_bucket = os.environ.get('S3_SCHEMA_BUCKET', "dev-eureka-rwe-spark")

#s3_schema_bucket = os.environ.get('S3_SCHEMA_BUCKET', "c0-dts-stag-rwe-spark")
s3_parquet_base_path = os.environ.get("S3_PARQUET_BASE_PATH", S3_PARQUET_BASE_LOCATION)
s3_parquet_concept_path = os.environ.get("S3_PARQUET_CONCEPT_PATH", S3_PARQUET_CONCEPT_LOCATION)
app_environment = os.environ.get("APP_ENVIRONMENT", "C3")

DEFAULT_MM_MODEL_NAME = os.environ.get("DEFAULT_MM_MODEL_NAME", "mm_model_nadirrs_v3")
DEFAULT_MDS_MODEL_NAME = os.environ.get("DEFAULT_MDS_MODEL_NAME", "mds_05_31")
DEFAULT_SC_MODEL_NAME = os.environ.get("DEFAULT_SC_MODEL_NAME", "pan_solids_v2")

MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', "https://mlflow-c3dev.concertohealth.ai/")

MAX_PROGRESSION_DAYS_RB = 60
MAX_MM_CRITERIA_DAYS_RB = 60  #For Rule based criteria max daysas per RWE-10227
MAX_MET_DAYS_RB = 60

IS_CLQ = os.environ.get('IS_CLQ', True)

SYNONYM_URL = os.environ.get('SYNONYM_URL', "https://dts-dev-be.concertohealth.ai/concepts/synonyms")
#C3 Synonym URL: https://c3dev5be.concertohealth.ai/concepts/synonyms

#CANCER SPECIFIC Variables
cancer_specific_constants = {
    "pan_solids": {
        "lab_test_codes": {
            "ecog": {
                        "concept_codes": ['89262-0', '89247-1', '42800-3', '273546003', '89245-5', '89243-0'],
                        "concept_names": []
            }

        },
        "medications" : [
                            'carboplatin','pembrolizumab','pemetrexed','paclitaxel',
                            'durvalumab','osimertinib','nivolumab', 'docetaxel',
                            'ipilimumab','gemcitabine','ramucirumab','etoposide',
                            'atezolizumab','bevacizumab','cisplatin','sotorasib',
                            'brigatinib','erlotinib','mobocertinib','dacomitinib',
                            'alectinib','necitumumab','lorlatinib','amivantamab', 
                            'afatinib','gefitinib','ceritinib','adagrasib',
                            'crizotinib'
                        ],
        
        "biomarkers": ['TP53', 'PTEN', 'PD-L1', 'TMB', 'KRAS', 'BRAF', 'EGFR', 'KIT', 'PIK3CA', 'STK11', 'TERT', 'CDH1', 'ALK', 'ROS1', 'MET', 'RET', 'ERBB2', 'NTRK1', 'NTRK2', 'NTRK3', 'FGFR3', 'FGFR4'],
        "imaging": [
            'Positron emission tomography with computed tomography',
            'Positron emission tomography',
            'Computed tomography',
            'MRI with contrast',
            'Magnetic resonance imaging',
            'MRI without contrast',
            'Radioisotope scan of bone'
        ],
        "charts": [
                        {
                            "chart_name": "medication",
                            "chart_type": "stacked_bar",
                            "feature_set": [
                                "carboplatin","pembrolizumab","pemetrexed","paclitaxel",
                                "durvalumab","osimertinib","nivolumab", "docetaxel",
                                "ipilimumab","gemcitabine","ramucirumab","etoposide",
                                "atezolizumab","bevacizumab","cisplatin","sotorasib",
                                "brigatinib","erlotinib","mobocertinib","dacomitinib",
                                "alectinib","necitumumab","lorlatinib","amivantamab", 
                                "afatinib","gefitinib","ceritinib","adagrasib",
                                "crizotinib"
                            ]
                        },
                        {
                            "chart_name": "biomarker",
                            "chart_type": "stacked_bar",
                            "feature_set": [
                                "biomarker_KRAS", "biomarker_TP53", "biomarker_TMB", 
                                "biomarker_PD-L1", "biomarker_EGFR", "biomarker_BRAF", 
                                "biomarker_PTEN", "biomarker_TERT", "biomarker_KIT", 
                                "biomarker_PIK3CA", "biomarker_CDH1", "biomarker_STK11", 
                                "biomarker_ALK",
                                #biomarker_ROS, biomarker_MET, biomarker_RET, biomarker_ERBB2,
                            ]
                        },
                        {
                            "chart_name": "stage",
                            "chart_type": "stacked_bar",
                            "feature_set": [ "stage", "tstage", "mstage", "nstage"],
                        },
                        {
                            "chart_name": "imaging",
                            "chart_type": "stacked_bar",
                            "feature_set": ["CT", "PET", "MRI", "Radioisotope"],
                        }
                    ]
    },
    "mds": {
        "lab_test_codes": {
            "wbc": {
                "concept_codes": ['6690-2', '26464-8', '12227-5'],
                "concept_names": []
            },
            "hemoglobin": {
                "concept_codes": ['718-7'],
                "concept_names": []
            },
            "neutrophils": {
                "concept_codes": ['26499-4', '751-8', '753-4', '768-2', '30451-9' ],
                "concept_names": []
            },
            "platelets":  {
                "concept_codes": ['777-3', '26515-7'],
                "concept_names": []
            },
            "ferritin": {
                "concept_codes": ['24373-3', '2276-4'],
                "concept_names": []
            },
            "blasts_count": {
                "concept_codes": ['708-8', '30376-8'],
                "concept_names": []
            },
            "blasts_percent": {
                "concept_codes": ['709-6', '26446-5'],
                "concept_names": []
            }
        }
    },
    "multiple_myeloma": {
        "lab_test_codes": {
            "m_protein_serum": {
                "concept_codes": ['33358-3','51435-6','35559-4','94400-9','50796-2'],
                "concept_names": []
            },
            "m_protein_urine": {"concept_codes": ['42482-0','35560-2'] , "concept_names": [] } ,
            "free_light_chain_reaction": {"concept_codes": ['36916-5','33944-0'] , "concept_names": [] } ,
            #"bone_marrow_test": ["40688-4", '33721-2', 'LP212327-3', '11118-7', '11150-0']
        },
        #TODO: Convert the names to codes and add to the above section
        "lab_test_names": ["heart rate","diastolic blood pressure","hemoglobin [mass/volume] in blood",
                            "platelets [#/volume] in blood","ecog performance status [interpretation]",
                            "Hemoglobin [Mass/volume] in Blood by calculation",
                            "Hemoglobin [Mass/volume] in Arterial blood",
                            "Hemoglobin [Mass/volume] in Blood --pre therapeutic phlebotomy",
                            "Neutrophils [#/volume] in Blood by Automated count",
                            "Neutrophils [#/volume] in Blood",
                            "Segmented neutrophils [#/volume] in Blood by Manual count",
                            "Segmented neutrophils [#/volume] in Blood",
                            "Neutrophils [#/volume] in Blood by Manual count",
                            "Lymphocytes [#/volume] in Blood",
                            "Lymphocytes [#/volume] in Blood by Manual count",
                            "Lymphocytes [#/volume] in Blood by Automated count",
                            "Platelets [#/volume] in Blood by Automated count",
                            "Platelets panel - Blood by Automated count",
                            "Platelets [#/volume] in Blood by Estimate",
                            "Platelets [#/volume] in Blood by Manual count]",
                            "chloride [moles/volume] in serum or plasma"
                        ],
        
    }

}

activity_date_fields = {
                        'medication': 'med_start_date', 
                        'patient_test': 'test_date', 
                        'tumor_exam': 'exam_date',
                        'care_goal': 'treatment_plan_start_date', 
                        'surgery': 'surgery_date', 
                        'radiation': 'rad_start_date',
                        'condition': 'diagnosis_date',
#                        'adverse_event': 'adverse_event_date',
                        'encounter': 'encounter_date',
#                        'disease_status': 'assessment_date',
                        'staging': 'stage_date'
        }

lab_class_map =  {
                        '0':'m_protein_in_serum',
                        '1':'m_protein_in_urine',
                        '3':'serum_free_light_kappa',
                        '4':'serum_free_light_lambda'
}



image_group_map = {
                        'Computed tomography': "CT",
                        'Positron emission tomography with computed tomography': "PET",
                        'Positron emission tomography': "PET",
                        'MRI with contrast': "MRI",
                        'MRI without contrast': "MRI",
                        'Magnetic resonance imaging': "MRI",
                        'Radioisotope scan of bone': "Radioisotope"
                    }
test_code_map = {
                    '33358-3' : 'Protein.monoclonal in Serum or Plasma by Electrophoresis',
                    '51435-6' : 'Protein.monoclonal band 1 in Serum or Plasma by Electrophoresis',
                    '35559-4' : 'Protein.monoclonal band 2 in Serum or Plasma by Electrophoresis',
                    '94400-9' : 'Protein.monoclonal in Serum or Plasma',
                    '33647-9' : 'protein.monoclonal/protein.total in serum or plasma by electrophoresis',
                    '50796-2' : 'Protein.monoclonal band 3 in Serum or Plasma by Electrophoresis',
                    '33647-9' : 'Protein.monoclonal/Protein.total in Serum or Plasma by Electrophoresis',
                    '56766-9' : 'protein.monoclonal band 1/protein.total in serum or plasma by electrophoresis',
                    '44932-2' : 'Protein.monoclonal band 2/Protein.total in Serum or Plasma by Electrophoresis',
                    '50792-1' : 'Protein.monoclonal band 3/Protein.total in Serum or Plasma by Electrophoresis',
                    '42482-0' : 'Protein.monoclonal in 24 hour Urine by Electrophoresis',
                    '40661-1' : 'Protein.monoclonal in Urine by Electrophoresis',
                    '35560-2' : 'Protein.monoclonal in Urine',
                    '36916-5' : 'Kappa light chains.free in Serum',
                    '33944-0' : 'lambda light chains.free in serum or plasma',
                    '33944-0' : 'Lambda light chains.free in Serum or Plasma',
                    '11051-0' : 'Lambda light chains in Serum or Plasma',
                    '11050-2' : 'Kappa light chains in Serum or Plasma',
                    '17861-6' : 'Calcium in Serum or Plasma',
                    '49765-1' : 'Calcium in Blood',
                    '8867-4'  :'heart rate',
                    '8462-4'  :'diastolic blood pressure',
                    '718-7'   :'hemoglobin in blood',
                    '26515-7' :'platelets in blood',
                    '89262-0' :'ecog performance status',
                    '20509-6' : 'Hemoglobin in Blood by calculation',
                    '30313-1' : 'Hemoglobin in Arterial blood',
                    '48725-6' : 'Hemoglobin in Blood --pre therapeutic phlebotomy',
                    '751-8' : 'Neutrophils in Blood by Automated count',
                    '26499-4' :'Neutrophils in Blood', 
                    '768-2' : 'Segmented neutrophils in Blood by Manual count',
                    '30451-9' : 'Segmented neutrophils in Blood',
                    '753-4' : 'Neutrophils in Blood by Manual count',
                    '26474-7' : 'Lymphocytes in Blood',
                    '732-8' : 'Lymphocytes in Blood by Manual count',
                    '731-0' : 'Lymphocytes in Blood by Automated count',
                    '777-3' : 'Platelets in Blood by Automated count',
                    '53800-9' : 'Platelets panel - Blood by Automated count',
                    '49497-1' : 'Platelets in Blood by Estimate',
                    '778-1' : 'Platelets in Blood by Manual count',
                    '2955-3' : 'sodium in urine',
                    '2951-2' : 'sodium in serum or plasma',
                    '2823-3' : 'potassium in serum or plasma',
                    '2828-2' : 'potassium in urine',
                    '2075-0' : 'chloride in serum or plasma',
                    '21377-7': 'magnesium in blood',
                    '19123-9': 'magnesium in serum or plasma',
                    '2777-1' : 'phosphate in serum or plasma',
                    'ecog': 'ecog'    #TODO: Adjust Apropriately
                }

lab_dict = {'m_protein_in_serum':['33358-3','51435-6','35559-4','94400-9','33647-9','50796-2','33647-9','56766-9','44932-2','50792-1'],
            'm_protein_in_urine':['42482-0','40661-1','35560-2'],
            'ca':['17861-6','49765-1'],
            'serum_free_light':['36916-5','33944-0','11051-0','11050-2'],
            'serum_free_light_kappa':['36916-5','11050-2'],
            'serum_free_light_lambda':['33944-0','11051-0'],
            'hemoglobin_in_blood':['718-7','20509-6','30313-1','48725-6','30350-3','30351-1','93846-4'],
            'neutrophils_count':['751-8','26499-4','768-2','30451-9','753-4'],
            'lymphocytes_count':['26474-7','732-8','731-0'],
            'platelets':['777-3','26515-7','53800-9','49497-1','778-1', '26516-5','74464-9','13056-7'],
            'na':['2951-2','2955-3'],
            'mg':['21377-7','19123-9'],
            'cl':['2075-0'],
            'phos' : ['2777-1'],
            'hr' : ['8867-4'],
            'dbp' : ['8462-4'],
            'ecog' : ['89262-0'],
            'k' : ['2823-3','2828-2'],
            'wbc':['6690-2','26464-8','49498-9','33256-9','12227-5','804-5'],
            'ferritin':['24373-3','2276-4','489004'],
            'blasts_percent':['709-6','26446-5'],
            'blasts_count':['708-8','30376-8']
           }

patient_test_as_evidence = {
    'bone_marrow_biopsy': ["40688-4", '33721-2', 'LP212327-3', '11118-7', '11150-0']
}

#Cohort Feature Mapper used for Solid Cancers. 
#NOTE: The values are to be aligned with the mapper used to train the model, `DEFAULT_SC_MODEL_NAME`
cohort_map = {
                'lung':0,
                'pancreas':1,
                'melanoma':2,
                'colorectal':3,
                'prostate':4,
                'bladder':5,
                'breast':6,
                'gastricesophagus':7,
                'hcc':8,
                'renal':9,
                'ovarian':10
        }


#Metstatic Date Extraction Rules specific to cohort type 
#Structure: 
#{
#    <cohort/cancer_indicator> : 
#                "stage": [<stage names>]
#                "mstage": [<mstage names>]
#                "mdx": True/False
#                    }
met_rules= {
    'mm': {
        #Confirmation on rule is pending, Being a liquid cancer type. Values are temporary and for testing
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'lung':{
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'breast':{
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'hcc':{
        "stage": ["stage 4b", "stage ivb","stage 4c", "stage ivc"],
        "mstage": ["m1"],
        "mdx": True
    },
    'prostate': {
        "stage": ["stage 4b", "stage ivb","stage 4c", "stage ivc"],
        "mstage": ["m1"],
        "mdx": True
    },
    'renal':{
        "stage": [],
        "mstage": ["m1"],
        "mdx": True
    },
    'pancreas':{
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'melanoma':{
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'bladder':{
        "stage": ["stage 4b", "stage ivb","stage 4c", "stage ivc"],
        "mstage": ["m1"],
        "mdx": True
    },
    'colorectal':{
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'gastricesophagus':{
        "stage": ["stage 4b", "stage ivb","stage 4c", "stage ivc"],
        "mstage": ["m1"],
        "mdx": True
    },
    'sclc':{
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    },
    'ovarian': {
        "stage": ["stage 4", "stage iv"],
        "mstage": ["m1"],
        "mdx": True
    }
}
