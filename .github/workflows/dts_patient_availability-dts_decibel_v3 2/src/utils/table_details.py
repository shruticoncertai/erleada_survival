table_names = {
    "omop": {
#        "adverse_event": "adverse_event",
        "care_goal": "caregoal",
        "condition": "condition_occurrence",
        "disease_status": "disease_status",
        "encounter": "encounter",
        "medication": "drug_exposure",
        "patient": "person",
        "patient_test": "measurement",
        "radiation": "radiation",
        "staging": "staging",
        "surgery": "procedure_occurrence",
        "tumor_exam": "tumor_exam",
        "cot": "cot",
        "lot": "lot",
        "concept": "concept",
        "concept_relationship": "concept_relationship",
        "concept_synonym": "concept_synonym",
        "concept_ancestor": "concept_ancestor",
        "imaging": "procedure_occurrence",
        "biomarker": "measurement"
    },
    "federated_omop": {
        "condition": "condition",
        "medication": "drug_exposure",
        "patient": "demographics",
        "patient_test": "measurement",
        "radiation": "radiation",
        "staging": "staging",
        "surgery": "surgery",
        "tumor_exam": "tumor_exam",
        "cot": "cot",
        "lot": "lot",
        "concept": "concept",
        "concept_relationship": "concept_relationship",
        "concept_synonym": "concept_synonym",
        "concept_ancestor": "concept_ancestor",
        "disease_status": "disease_status",
        "biomarker": "measurement",
        "imaging": "surgery"
    },
    "cdm": {
        "adverse_event": "adverse_event",
        "care_goal": "care_goal",
        "condition": "condition",
        "condition_mo": "condition_mo",
        "disease_status": "disease_status",
        "encounter": "encounter",
        "medication": "medication",
        "patient": "patient",
        "patient_test": "patient_test",
        "patient_test_mo": "patient_test_mo",
        "radiation": "radiation",
        "staging": "staging",
        "surgery": "surgery",
        "tumor_exam": "tumor_exam",
        "cot": "lot_cot",
        "lot": "lot_cot",
        "concept": "concept",
        "concept_relationship": "concept_relationship",
        "concept_synonym": "concept_synonym",
        "concept_ancestor": "concept_ancestor",
    }
}

column_names = {
    "omop":{
        "condition":{
            "chai_patient_id": "person_id",
            "diagnosis_date": "condition_start_date",
            "concept_id":  "condition_concept_id",
            "diagnosis_code_standard_code": "concept_code",
            "diagnosis_code_standard_name": "concept_name",
            "diagnosis_type_concept_id": "condition_type_concept_id"
        },
        "patient_test": {
            "chai_patient_id": "person_id",
            "test_date" : "measurement_date",
            "test_name_standard_code": "concept_code",
            "test_name_standard_name" : "concept_name",
            "test_value_numeric" : "value_as_number",
            "test_unit_source_name" : "measurement_unit_source_name",
            "concept_id": "measurement_concept_id",
            "unit_concept_id" : "unit_concept_id"
        },
        "medication":{
            "chai_patient_id": "person_id",
            "med_start_date":  "drug_exposure_start_date",
            "med_end_date": "drug_exposure_end_date",
            "concept_id": "drug_concept_id",
            "med_generic_name_standard_name": "concept_name",
            "med_generic_name_standard_code": "concept_code"
        },
        "tumor_exam": {
            "chai_patient_id": "person_id",
            "exam_date": "exam_date",
            "tumor_grade_standard_name": "concept_name",
            "concept_id": "tumor_grade_concept_id"
        },     
        "patient": {
            "concept_id":  "gender_concept_id",
            "gender": "concept_name",
            "chai_patient_id": "person_id",
            "source_age": "source_age",
            "date_of_death": "date_of_death"
        },
        "surgery":{
            "surgery_date": 'procedure_date',
            "chai_patient_id": "person_id",
            "concept_id": "procedure_concept_id",
            "procedure_name": "concept_name"
        },
        "imaging": {
            "chai_patient_id": "person_id",
            "radiology_visitdate": "procedure_date",
            "radiology_name_standard_name": "concept_name",
            "concept_id": "procedure_concept_id",
            "curation_indicator": "curation_indicator",
            "src_type": "src_type"
        },
        "biomarker": {
            "chai_patient_id": "person_id",
            "concept_id": "measurement_concept_id",
            "biomarker_name_name": "concept_name" ,
            "value_as_concept_id": "value_as_concept_id",
            "report_date": "measurement_date",

        },
        "radiation": {
            "chai_patient_id": "person_id",
            "rad_start_date": "rad_start_date"
        },
        "adverse_event": {
            "chai_patient_id": "person_id",
            "adverse_event_date": "adverse_event_date"
        },
        "encounter": {
            "chai_patient_id": "person_id",
            "encounter_date": "encounter_date"
        },
        "disease_status":{
            'chai_patient_id': 'person_id',
            'assessment_date': 'assessment_date',
            'assessment_name_concept_id': 'assessment_name_concept_id',
            'assessment_value_concept_id': 'assessment_value_concept_id',
            'source': 'source',
            'disease_status_doc_id': 'disease_status_doc_id'
        }, 
        "staging": {
                'chai_patient_id': 'person_id',
                'stage_date': 'stage_date',
                'curation_indicator': 'curation_indicator',
                'stage_group_concept_id': 'stage_group_concept_id',
                'tstage_concept_id': 'tstage_concept_id',
                'nstage_concept_id': 'nstage_concept_id',
                'mstage_concept_id': 'mstage_concept_id'
        }
    },
    "federated_omop": {
        "adverse_event": {
                    "patient_id": "person_id",
                    "adverse_event_date": "adverse_event_datetime",
                    "adverse_event_code_code": "adverse_event_concept_id",
                    "adverse_event_code_name": "adverse_event_code_name",
                    "curation_indicator": "curation_indicator",
        },
        "condition": {
            "chai_patient_id": "person_id",
            "diagnosis_date": "date",
            "diagnosis_code_standard_code": "condition_primary_concepts_code",
            "diagnosis_code_standard_name": "condition_primary_concepts_name",
            "diagnosis_type_concept_id": "condition_type_concept_id",
            #"diagnosis_code_standard_vocabulary": "condition_primary_concepts_vocab",
            # diagnosis_type needed for metastasis sites feature -- Currently not available
            #"diagnosis_type_standard_code": "condition_type_concept_id",
            #"diagnosis_type_standard_name": "condition_type_standard_name",
            #"diagnosis_type_standard_vocabulary": "condition_type_standard_vocabulary",
            #"curation_indicator": "curation_indicator",
            "source": "source",
            "condition_doc_id": "condition_doc_id",
        },
        "encounter": {
            "patient_id": "person_id",
            "encounter_date": "encounter_date",
        },
        "medication": {
            "chai_patient_id": "person_id",
            "med_start_date": "date",
            "med_end_date": "drug_exposure_end_date",
            "med_generic_name_standard_code": "drug_exposure_primary_concepts_code",
            "med_generic_name_standard_name": "drug_exposure_primary_concepts_name",
            "curation_indicator": "curation_indicator",
        },
        "patient": {
            "patient_id": "person_id",
            "gender": "gender_concept_id",
            "race": "race_concept_id",
            "ethnicity": "ethnicity_concept_id",
            "date_of_birth": "date",
        },
        "biomarker": {
            "chai_patient_id": "person_id",
            "biomarker_name_name": "measurement_primary_concepts_name" ,
            "value": "measurement_value_name",
            "report_date": "date",

        },
        "patient_test": {
            "chai_patient_id": "person_id",
            "test_date": "date",
            "test_name_standard_code": "measurement_primary_concepts_code",
            "test_name_standard_name": "measurement_primary_concepts_name",
            "test_value_numeric": "measurement_value_number",
            "test_unit_source_name": "measurement_unit_name",
            "curation_indicator": "curation_indicator",
            "test_value_name": "measurement_value_name",
            "source": "source",
            "measurement_doc_id": "measurement_doc_id"
            #"test_value_source_name": "measurement_value_name",
            #"test_value_standard_code": "measurement_value_code",
            #"test_value_standard_name": "measurement_value_name",
            #"test_value_standard_vocabulary": "measurement_value_vocab", # noqa
            #"test_unit_standard_code": "measurement_unit_code",
        },
        "disease_status":{
            'chai_patient_id': 'person_id',
            'assessment_date': 'date',
            'assessment_name_standard_name': 'disease_status_primary_concepts_name',
            'assessment_value_standard_name': 'disease_status_assessment_value_name',
            'source': 'source',
            'disease_status_doc_id': 'disease_status_doc_id'
        },
        "radiation": {
            "chai_patient_id": "person_id",
            "rad_start_date": "rad_start_datetime",
            "rad_end_date": "rad_end_datetime",
            "site_source_code": "site_source_concept_id",
            "site_source_name": "site_source_name",
            "site_standard_code": "site_concept_id",
            "site_standard_name": "site_standard_name",
            "curation_indicator": "curation_indicator",
        },
        "staging": {
            "chai_patient_id": "person_id",
            "stage_date": "date",
            "stage_group_standard_code": "staging_primary_concepts_code",
            "stage_group_standard_name": "staging_primary_concepts_name",
            "tstage_standard_code": "staging_tstage_code",
            "tstage_standard_name": "staging_tstage_name",
            "nstage_standard_code": "staging_nstage_code",
            "nstage_standard_name": "staging_nstage_name",
            "mstage_standard_code": "staging_mstage_code",
            "mstage_standard_name": "staging_mstage_name",
            "curation_indicator": "curation_indicator",
            "source": "source",
            "staging_doc_id": "staging_doc_id",
        },
        "imaging": {
            "chai_patient_id": "person_id",
            "radiology_visitdate": "date",
            "radiology_name_standard_name": "surgery_primary_concepts_name",
            "curation_indicator": "curation_indicator",
        },
        "surgery": {
            "chai_patient_id": "person_id",
            "surgery_date": "procedure_datetime",
            "surgery_standard_code": "surgery_primary_concepts_code",
            "surgery_standard_name": "surgery_primary_concepts_name",
            "curation_indicator": "curation_indicator",
        },
        "tumor_exam": {
            "patient_id": "person_id",
            "exam_date": "date",
            "tumor_histology_standard_code": "tumor_exam_primary_concepts_code",
            "tumor_histology_standard_name": "tumor_exam_primary_concepts_name",
            "curation_indicator": "curation_indicator",
        },
    }
}

def get_column_name(standard_table_name: str, column_name: str=None, schema: str="omop") -> str:
    """
    Give schema, standard_table_name and column_name to get the column_name of the schema
    
    Parameters
    ----------
    standard_table_name
    column_name
    schema

    Returns
    -------
    str/dict
    """
    if column_name is None:
        return column_names[schema][standard_table_name]
    
    return column_names[schema][standard_table_name][column_name]

def get_table(schema: str, standard_table_name: str) -> str:
    """
    Give schema and standard table name, returns schema table name

    Parameters
    ----------
    schema
    standard_table_name

    Returns
    -------
    str
    """

    return table_names[schema][standard_table_name]

def get_schema_type(schema) -> str:
    if "fed_omop" in schema:
        #logger.info('Infered short schema type is federated_omop')
        return "federated_omop"
    elif "omop" in schema:
        #logger.info('Infered short schema type is omop')
        return "omop"
    elif "cdm" in schema:
        #logger.info('Infered short schema type is cdm')
        return "cdm"
    else:
        #logger.info('Infered short schema type is pt360/ rwd360')
        return "pt360"
