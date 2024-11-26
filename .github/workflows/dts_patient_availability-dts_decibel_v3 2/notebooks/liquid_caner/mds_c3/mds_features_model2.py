import json
import pandas as pd
import random
import copy
import numpy as np
import pickle
import datetime
from mds_labs_model2 import run_labs,lab_name_dict, calculate_statistics, convert_unit, calculate_statistics_pl
from data_transfer_utility.application import DataTransferUtility


REDSHIFT_CREDENTIALS = json.load(open('../secrets/dtu_redshift_credentials.json', 'r'))
dtu = DataTransferUtility(df_mode='pandas', **REDSHIFT_CREDENTIALS)

nlp_tables = {
    'Bladder': {'schema': 'c3_nlp_data_lake_bladder_202306'},
    'Prostate': {'schema': 'c3_nlp_data_lake_prostate_202306'},
    'Melanoma': {'schema': 'c3_nlp_data_lake_melanoma_202306'},
    'Breast': {'schema': 'c3_nlp_data_lake_breast_202306'},
    'Lung': {'schema': 'c3_nlp_data_lake_lung_202306',#unreconciled
            'condition':'model_factory_dev.lung_202306_condition_reconciled_delivery',
            'patient_test': 'model_factory_dev.lung_202306_patient_test_reconciled_delivery',
            'radiation':'model_factory_dev.lung_202306_radiation_reconciled_delivery',
            'staging': 'model_factory_dev.lung_202306_staging_reconciled_delivery',
            'surgery': 'model_factory_dev.lung_202306_surgery_reconciled_delivery',
            'tumor_exam': 'model_factory_dev.lung_202306_tumor_exam_reconciled_delivery',
            }
}

biomarker_list = ['U2AF1', 'ASXL1', 'GATA2', 'CEBPA', 'IDH1', 'IDH2', 'TET2', 'SF3B1', 'TP53', 'RAD21', 'SRSF2', 'RUNX1']

#['KRAS','TP53','EGFR','ALK','BRAF','TMB','ROS1','MET','ERBB2','RET','PD-L1','PTEN','TERT','KIT','PIK3CA','CDH1','STK11','NTRK1','NTRK2','NTRK3','FGFR3','FGFR4']

#MDS biomarker : 

def get_condition_table(schema, cohort):
    # lab_code_df = dtu.redshift_to_df('''SELECT code,type FROM model_factory_dev.cancer_dx_list_icd10_unified_v6_6''')
    lab_code_df = dtu.redshift_to_df('''SELECT code,type FROM model_factory_dev.cancer_dx_list_icd10_unified_v8_0''')
    if cohort=='Lung':       
        lab_code_list = list(lab_code_df[lab_code_df['type']==str.lower(cohort)]['code'].unique())
        lab_code_list += ['254637007','254632001']
    elif cohort=='RCC':       
        lab_code_list = list(lab_code_df[lab_code_df['type']==str.lower(cohort)]['code'].unique())
        lab_code_list += ['41607009']
    elif cohort=='Gastricesophagus':       
        lab_code_list = list(lab_code_df[lab_code_df['type']==str.lower('gastric')]['code'].unique())
        lab_code_list += list(lab_code_df[lab_code_df['type']==str.lower('esophagus')]['code'].unique())
    elif cohort=='Ovarian':       
        lab_code_list = list(lab_code_df[lab_code_df['type']==str.lower('ofpc')]['code'].unique())
    else:
        lab_code_list = list(lab_code_df[lab_code_df['type']==str.lower(cohort)]['code'].unique())
    print(lab_code_list)
        
    code_list = ','.join(["'"+str(x)+"'" for x in lab_code_list])
    cond_table = dtu.redshift_to_df(''' SELECT chai_patient_id, curation_indicator, diagnosis_date, diagnosis_code_standard_code, diagnosis_code_standard_name FROM {schema}.condition where (lower(diagnosis_code_standard_code) in ({code_list}))'''.format(schema = schema, code_list = code_list))
    
    # print(cond_table['curation_indicator'].value_counts())
    cond_table = cond_table[cond_table['curation_indicator']==1]
    cond_table = cond_table.dropna(subset=['diagnosis_date'])
    cond_table = cond_table.sort_values(by=['diagnosis_date'],ascending=True)
    cond_table = cond_table.drop_duplicates(subset=['chai_patient_id'])
    cond_table = cond_table[cond_table['diagnosis_date'].dt.year>2014]
    return cond_table

def get_index_table(index_date_table, cond_table):
    patient_ids = list(cond_table['chai_patient_id'].unique())
    pat_str = "','".join(patient_ids)
    index_date_df = dtu.redshift_to_df(''' SELECT * FROM {index_date_table} where chai_patient_id in ('{pat_str}')'''.format(index_date_table = index_date_table,pat_str=pat_str))
    return index_date_df

def get_index_table_wo_cond(index_date_table):
    index_date_df = dtu.redshift_to_df(''' SELECT * FROM {index_date_table}'''.format(index_date_table = index_date_table))
    return index_date_df

def get_mds_initial_dx_table(initial_dx_table):
    lab_code_df = dtu.redshift_to_df('''SELECT code,type FROM model_factory_dev.cancer_dx_list_icd10_unified_v8_0''')
    lab_code_list = list(lab_code_df[lab_code_df['type']==str.lower('mds')]['code'].unique())
    code_list = ','.join(["'"+str(x)+"'" for x in lab_code_list])
    initial_dx_df = dtu.redshift_to_df(''' SELECT chai_patient_id, curation_indicator, diagnosis_date, diagnosis_code_standard_code, diagnosis_code_standard_name FROM {initial_dx_table} where (lower(diagnosis_code_standard_code) in ({code_list}))'''.format(initial_dx_table = initial_dx_table,code_list = code_list))
    return initial_dx_df

def get_lot_table(lot_table_name, cond_table):
    patient_ids = list(cond_table['chai_patient_id'].unique())
    pat_str = "','".join(patient_ids)
    lot_table = dtu.redshift_to_df(''' SELECT chai_patient_id,line_start,line_end,line_number,regimen_start, regimen_end FROM {lot_table} where chai_patient_id in ('{pat_str}')'''.format(lot_table = lot_table_name,pat_str=pat_str))

    # lot_table = lot_table.merge(cond_table[['chai_patient_id','diagnosis_date']],on='chai_patient_id',how='inner')
#     lot_table = lot_table[lot_table['line_end']>='2020-01-01']
    # lot_table = lot_table[lot_table['line_number'].isin(line_numbers)]
    # lot_table = lot_table[lot_table.apply(lambda x:x['line_start']>=x['diagnosis_date'],axis=1)]
#     lot_table = lot_table[['chai_patient_id','line_start','line_end', 'line_number', 'med_name_standard_name', 'index_date']]
    lot_table= lot_table.drop_duplicates()
    return lot_table

def get_medication_lot(schema,lot_table):
#     med_list = list(lot_table.med_name_standard_name.unique())
    med_list = list(set(['carboplatin','pembrolizumab','pemetrexed','paclitaxel','durvalumab','osimertinib','nivolumab', 'docetaxel', 'ipilimumab','gemcitabine','ramucirumab','etoposide','atezolizumab','bevacizumab','cisplatin','sotorasib','erlotinib','afatinib','gefitinib'
,'dacomitinib','amivantamab','mobocertinib','necitumumab','adagrasib','crizotinib','ceritinib','alectinib','brigatinib','lorlatinib']))
    
    med_df = pd.DataFrame(columns=['chai_patient_id', 'line_start'] + med_list)
    med_df[['chai_patient_id', 'line_start']] = lot_table[['chai_patient_id', 'line_start']].drop_duplicates()
    for col in med_list:
        med_df[col] = False
    for i,j in med_df.iterrows():
        pat_df = lot_table[(lot_table['chai_patient_id']==j['chai_patient_id'])&(lot_table['line_start']==j['line_start'])]
        med_list_pat = list(pat_df.med_name_standard_name.unique())
        med_list_pat = [x for x in med_list_pat if x in med_list]
        med_df.loc[(med_df['chai_patient_id']==j['chai_patient_id'])&(med_df['line_start']==j['line_start']),med_list_pat]=True
    return med_df

def get_medication(schema, cond_table):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
    med_list = list(set(['carboplatin','pembrolizumab','pemetrexed','paclitaxel','durvalumab','osimertinib','nivolumab', 'docetaxel', 'ipilimumab','gemcitabine','ramucirumab','etoposide','atezolizumab','bevacizumab','cisplatin','sotorasib','erlotinib','afatinib','gefitinib'
,'dacomitinib','amivantamab','mobocertinib','necitumumab','adagrasib','crizotinib','ceritinib','alectinib','brigatinib','lorlatinib']))
    patient_ids = list(set(cond_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    med = dtu.redshift_to_df("""
        select chai_patient_id,curation_indicator,med_start_date,med_generic_name_standard_name from {schema}.medication
        where chai_patient_id IN ('{pat_str}')
    """.format(schema=schema,pat_str=pat_str))
    
    # print(f"medication curation value count : {med['curation_indicator'].value_counts()}")
    # # med = med[med['curation_indicator']==0]
    # med = med[~((med['curation_indicator']==1) & (med['chai_patient_id'].isin(non_curation_pat_list)))] 
    
    med = med.merge(cond_table[['chai_patient_id','random_date']],on='chai_patient_id',how='inner')
    
    med = med[med.apply(lambda x:(0<=(x['random_date']-x['med_start_date']).days<90),axis=1)]
    # med = med[med.apply(lambda x:(0<=(x['random_date']-x['med_start_date']).days<180),axis=1)]
    med_subset = med[['chai_patient_id','med_generic_name_standard_name']].drop_duplicates()
    
    med_subset = med_subset[med_subset['med_generic_name_standard_name'].isin(med_list)]
    # conmed_list = list(conmed_subset.med_generic_name_standard_name.unique())
    med_df = pd.DataFrame(columns=['chai_patient_id'] + med_list)
    med_df[['chai_patient_id']] = cond_table[['chai_patient_id']]
    for i,j in med_df.iterrows():
        pat_df = med_subset[(med_subset['chai_patient_id']==j['chai_patient_id'])]
        med_list_pat = list(pat_df.med_generic_name_standard_name.unique())
        med_df.loc[(med_df['chai_patient_id']==j['chai_patient_id']),med_list_pat]=1
    med_df = med_df.drop_duplicates()
    med_df = med_df.fillna(0)
    return med_df

def dod_subtraction(schema, lot_table):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    patient = dtu.redshift_to_df(
        '''SELECT * FROM {schema}.patient WHERE chai_patient_id IN ('{pat_str}')'''.format(schema=schema,
                                                                                           pat_str=pat_str))
    patient = patient[~pd.isna(patient['date_of_death'])][['chai_patient_id', 'date_of_death']]
    lot_table = lot_table.merge(patient, on='chai_patient_id', how='left')
    lot_table = lot_table[~lot_table.apply(lambda x: x['line_end'] == x['date_of_death'], axis=1)]
    lot_table = lot_table.drop('date_of_death', axis=1)
    return lot_table


def get_conmed(schema, cond_table):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
    conmed_list = pd.read_csv("../conmed_mapping_file.csv")
    conmed_list.child_code = conmed_list.child_code.astype(str)
    conmed_list.parent_code = conmed_list.parent_code.astype(str)
    parent_code = list(conmed_list.parent_code.unique())
    child_code = list(conmed_list.child_code.unique())
    patient_ids = list(set(cond_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    conmed = dtu.redshift_to_df("""
        select chai_patient_id,curation_indicator,med_start_date,med_generic_name_standard_name from {schema}.medication
        where chai_patient_id IN ('{pat_str}')
        AND (med_generic_name_standard_code IN ('{parent_codes}') OR
        med_brand_name_standard_code IN ('{child_codes}'))

    """.format(schema=schema,
               pat_str=pat_str,
              parent_codes="','".join(parent_code),
              child_codes="','".join(child_code)))
    
    # print(f"conmed curation value count : {conmed['curation_indicator'].value_counts()}")
    # # conmed = conmed[conmed['curation_indicator']==0]
    # conmed = conmed[~((conmed['curation_indicator']==1) & (conmed['chai_patient_id'].isin(non_curation_pat_list)))] 

#     conmed_value_df = pd.DataFrame(conmed.med_generic_name_standard_code.value_counts())
#     conmed_value_df = conmed_value_df.reset_index().rename(columns={'index':'code',
#                                                                     'med_generic_name_standard_code':'count'})
#     conmed_value_df = conmed_value_df.sort_values('count',ascending=False)[:20]
#     med_code_list = conmed_value_df.code.unique()
    conmed = conmed.merge(cond_table[['chai_patient_id','random_date']],on='chai_patient_id',how='inner')
#     conmed['diff'] = (conmed['med_start_date']-conmed['line_start']).dt.days
#     conmed_subset = conmed[(conmed.med_generic_name_standard_code.isin(med_code_list)) &
#           (conmed['diff']>-60) & (conmed['diff']<30)].drop_duplicates()
#     conmed = conmed[conmed.apply(lambda x:x['line_end']>=x['med_start_date'],axis=1)]
#     conmed = conmed[conmed.apply(lambda x:x['med_start_date']>=x['line_start'],axis=1)]
#     conmed_subset = conmed
    
    conmed = conmed[conmed.apply(lambda x:(0<=(x['random_date']-x['med_start_date']).days<90),axis=1)]
    # conmed = conmed[conmed.apply(lambda x:(0<=(x['random_date']-x['med_start_date']).days<180),axis=1)]
    conmed_subset = conmed[['chai_patient_id',
                                                     'med_generic_name_standard_name']].drop_duplicates()
    
    conmed_subset = conmed_subset[conmed_subset['med_generic_name_standard_name'].isin(['dexamethasone'])]
    conmed_list = list(conmed_subset.med_generic_name_standard_name.unique())
    conmed_df = pd.DataFrame(columns=['chai_patient_id'] + conmed_list)
    conmed_df[['chai_patient_id']] = cond_table[['chai_patient_id']]
    for i,j in conmed_df.iterrows():
        pat_df = conmed_subset[(conmed_subset['chai_patient_id']==j['chai_patient_id'])]
        med_list_pat = list(pat_df.med_generic_name_standard_name.unique())
        conmed_df.loc[(conmed_df['chai_patient_id']==j['chai_patient_id']),med_list_pat]=1
    conmed_df = conmed_df.drop_duplicates()
    conmed_df = conmed_df.fillna(0)
    return conmed_df

def get_comorbidity(schema, lot_table):
    icd = ['G30.9', 'M35.9', 'Z86.73', 'J44.9', 'K76.9',
           'I50.9', 'M35.9', 'E14.0', 'G81.9', 'C95.9', 'C96.9',
           'C79.9', 'I25.2', 'Z87.448', 'Z86.7', 'Z87.11']
    icd_dx = list(map(lambda x: x.lower(), icd))
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    comorbidities = dtu.redshift_to_df("""
    Select * from {schema}.condition
    where chai_patient_id IN ('{pat_str}')
    and (diagnosis_code_standard_code ILIKE 'B20%' OR diagnosis_type_standard_code ILIKE 'B20%'
    OR lower(diagnosis_code_standard_code) IN ('{code_str}') OR lower(diagnosis_type_standard_code) IN ('{code_str}'))
    """.format(schema=schema,
               pat_str=pat_str,
               code_str="','".join(icd_dx)))
#     comorbidities = dtu.redshift_to_df("""
#         Select * from {schema}.condition
#         where chai_patient_id IN ('{pat_str}')
#         """.format(schema=schema,
#                    pat_str=pat_str))

#     comorbidities = comorbidities[comorbidities['curation_indicator'] == 0]
    comor_list = list(comorbidities.diagnosis_code_standard_name.unique())
    comor_df = pd.DataFrame(columns=['chai_patient_id'] + comor_list)
    comor_df['chai_patient_id'] = lot_table['chai_patient_id']
    for chai_id in comorbidities.chai_patient_id.unique():
        pat_df = comorbidities[(comorbidities['chai_patient_id'] == chai_id)]
        comor_list_pat = list(pat_df.diagnosis_code_standard_name.unique())
        comor_df.loc[comor_df['chai_patient_id'] == chai_id, comor_list_pat] = 1
    comor_df = comor_df.drop_duplicates()
    comor_df = comor_df.fillna(0)
    return comor_df


def get_surgery(schema, lot_table, cohort):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    if cohort in nlp_tables:
        if cohort == 'Lung':
            surgery_table = nlp_tables[cohort]['surgery']
        else:
            surgery_table = nlp_tables[cohort]['schema'] + '.surgery'
        surgery_nlp = dtu.redshift_to_df("""select * from {surgery_table}
                                        where chai_patient_id in ('{pat_str}')""".format(surgery_table = surgery_table,
                                                pat_str = pat_str))
        surgery_table = schema + '.surgery'
        surgery_pt360 = dtu.redshift_to_df("""select * from {surgery_table} where
                                         chai_patient_id in ('{pat_str}')""".format(surgery_table = surgery_table,
                                                pat_str = pat_str))
#         surgery_pt360 = surgery_pt360[surgery_pt360['curation_indicator']==0]
        surgery_pt360 = surgery_pt360[~surgery_pt360['chai_patient_id'].isin(surgery_nlp['chai_patient_id'])]
        surgery = pd.concat([surgery_nlp, surgery_pt360])
    else:
        surgery = dtu.redshift_to_df("""select * from {surgery_table} where
                                         chai_patient_id in ('{pat_str}')""".format(surgery_table = schema + '.surgery',
                                                pat_str = pat_str))
#         surgery = surgery[surgery['curation_indicator']==0]
        
    surgery = surgery[['chai_patient_id','surgery_date','surgery_standard_name']]
    surgery = surgery.drop_duplicates()
    surgery = surgery.merge(lot_table,on=['chai_patient_id'],how='inner')
#     surgery = surgery[surgery.apply(lambda x:x['surgery_date']<=x['line_end'],axis=1)]
#     surgery = surgery[surgery.apply(lambda x:x['surgery_date']>=x['line_start'],axis=1)]
    surgery = surgery[surgery.apply(lambda x:(x['random_point']-90)<=(x['surgery_date']-x['line_start']).days<=x['random_point'],axis=1)]
    surgery['surgery_standard_name'] = surgery['surgery_standard_name'].str.lower()
    surgery_list = list(surgery.surgery_standard_name.unique())
    surgery_df = pd.DataFrame(columns=['chai_patient_id', 'line_start'] + surgery_list)
    surgery_df[['chai_patient_id', 'line_start']] = lot_table[['chai_patient_id', 'line_start']]
    for i,j in surgery_df.iterrows():
        pat_df = surgery[(surgery['chai_patient_id']==j['chai_patient_id'])&(surgery['line_start']==j['line_start'])]
        surgery_list_pat = list(pat_df.surgery_standard_name.unique())
        surgery_df.loc[(surgery_df['chai_patient_id']==j['chai_patient_id'])&(surgery_df['line_start']==j['line_start']),surgery_list_pat]=1
    surgery_df = surgery_df.drop_duplicates()
    surgery_df = surgery_df.fillna(0)
#     surgery_df = surgery_df.drop(np.nan, axis = 1)
    return surgery_df


def get_labs(schema, cond_table):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
#     lab_tests = pd.read_excel('Different_lab_tests_codes_Sep_2023.xlsx')
#     codes = list(set(lab_tests['Code'].tolist()))
    patient_ids = list(set(cond_table['chai_patient_id']))
    print(f"len pat_ids : {len(patient_ids)}")
    pat_str = "','".join(patient_ids)
    run_labs_str = "','".join(run_labs)
    
    patient_test_df = dtu.redshift_to_df('''SELECT 
    chai_patient_id,curation_indicator,test_name_standard_name,test_name_standard_code,
    test_date,test_value_numeric_standard,test_unit_standard_name FROM {schema}.patient_test
    WHERE ((chai_patient_id IN ('{pat_str}')) and (test_name_standard_code IN ('{run_labs_str}')))'''.format(schema=schema, pat_str=pat_str,run_labs_str=run_labs_str))

    patient_test_df = patient_test_df[~((pd.isna(patient_test_df['test_date'])) | (pd.isna(patient_test_df['test_value_numeric_standard'])))]
    
    print(f"patient_test_df shap 1 : {patient_test_df.shape}")
    # print(f"lab(patient_test) curation value count : {patient_test_df['curation_indicator'].value_counts()}")
    # # patient_test_df = patient_test_df[patient_test_df['curation_indicator'] == 0]
    # patient_test_df = patient_test_df[~((patient_test_df['curation_indicator']==1) & (patient_test_df['chai_patient_id'].isin(non_curation_pat_list)))]  
    
#     patient_test_df['test_value_numeric_standard'] = patient_test_df['test_value_numeric_standard'].str.replace("+", ).replace(">","").replace("SUPPRESSED","")
    patient_test_df['test_value_numeric_standard'] = patient_test_df['test_value_numeric_standard'].apply(lambda x: x.replace('+','').replace('>','').replace('SUPPRESSED','0').replace('<','').replace('< ','').replace('= ','').replace(',','').replace(' - 20','').replace(' -  20','').replace('..2','.2').replace('â‰¥ ','') if isinstance(x,str) else x)
    patient_test_df['test_value_numeric_standard'] = patient_test_df['test_value_numeric_standard'].astype(float)
#     patient_test_df = patient_test_df.apply(do_unit_conversion, axis = 1)
    
    print(f"patient_test_df shap 2 : {patient_test_df.shape}")
    patient_test_df = patient_test_df[
        ['chai_patient_id', 'test_name_standard_name',
         'test_name_standard_code','test_date', 'test_value_numeric_standard','test_unit_standard_name']].drop_duplicates()
#     merged_df = patient_test_df[(patient_test_df['test_name_standard_name'].isin(run_labs)) & 
#                                 (patient_test_df['test_name_standard_code'].isin(codes))].merge(lot_table,
#                                                                                                 on='chai_patient_id',
#                                                                                                 how='inner')

    print(f"patient_test_df shap 3 : {patient_test_df.shape}")
    # merged_df = patient_test_df[(patient_test_df['test_name_standard_code'].isin(run_labs))].merge(cond_table,
    #                                                                                         on='chai_patient_id',
    #                                                                                         how='inner')
    merged_df = patient_test_df.merge(cond_table,on='chai_patient_id',how='inner')
    
    print(f"merged_df shap 1 : {merged_df.shape}")
    
    merged_df = convert_unit(merged_df)
    print(f"merged_df shap 2 : {merged_df.shape}")
    merged_df = merged_df[~((pd.isna(merged_df['test_date'])) | (pd.isna(merged_df['test_value_numeric_standard'])))]
    print(f"merged_df shap 3 : {merged_df.shape}")
    merged_df.to_csv('data_backup/m2_mds_2024-05-31/inference/mds_labs_with_std_unite.csv')
    # merged_df = pd.read_csv('data_backup/m2_mds_2024-05-29_with_biomarker/mds_labs_with_std_unite.csv')
    # merged_df['test_date'] = merged_df['test_date'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
    # merged_df['random_date'] = merged_df['random_date'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
    
#     random_df = merged_df[['chai_patient_id', 'line_start', 'line_end']].drop_duplicates()
#     random_df['day_diff'] = random_df.apply(lambda x: (x['line_end'] - x['line_start']).days, axis=1)
#     random_df['random_point'] = random_df.apply(
#         lambda x: 0 if x['day_diff'] in [0, 1] else random.randint(1, x['day_diff'] - 1), axis=1)
#     random_df = random_df[['chai_patient_id', 'random_point','day_diff', 'line_start']]
    
#     merged_df = merged_df.merge(random_df, on=['chai_patient_id', 'line_start'], how='left')
    merged_df['day_diff'] = merged_df.apply(lambda x: (x['random_date'] - x['test_date']).days, axis=1)
    merged_df.loc[:, 'rand_wind'] = '-1'
    
#     merged_df.loc[(merged_df['day_diff'] > (merged_df['random_point'] - 90)) & (
#             merged_df['day_diff'] < merged_df['random_point']), 'rand_wind'] = 3
#     merged_df.loc[(merged_df['day_diff'] > (merged_df['random_point'] - 90)) & (
#                 merged_df['day_diff'] < merged_df['random_point']), 'rand_wind'] = 2
    merged_df.loc[(merged_df['day_diff'] >=45) & (
                merged_df['day_diff'] < 180), 'rand_wind'] = '45_180'
    merged_df.loc[(merged_df['day_diff'] >=0) & (
                merged_df['day_diff'] < 45), 'rand_wind'] = '0_45'
    # merged_df.loc[(merged_df['day_diff'] >=45) & (
    #             merged_df['day_diff'] < 180), 'rand_wind'] = 1
    # merged_df.loc[(merged_df['day_diff'] >=0) & (
    #             merged_df['day_diff'] < 45), 'rand_wind'] = 0
#     merged_df = merged_df.dropna(how='any')

    
    merged_df['test_value_numeric_standard'] = pd.to_numeric(merged_df['test_value_numeric_standard'])
    final_stat_df = cond_table[['chai_patient_id']].drop_duplicates()
    
    unique_labs = []
    for lab in run_labs:
        unique_labs.append(lab_name_dict[lab].split('_')[0])
        statistics_df = calculate_statistics_pl(merged_df[merged_df["test_name_standard_code"]==lab], lab_name_dict[lab], indices=['0_45','45_180'])
        # statistics_df = calculate_statistics(merged_df[merged_df["test_name_standard_code"]==lab], lab_name_dict[lab], cond_table)
        # statistics_df = calculate_statistics(merged_df[merged_df["test_name_standard_code"]==lab], lab, cond_table)
        final_stat_df = final_stat_df.merge(statistics_df, on=["chai_patient_id"], how="left")
#     final_stat_df = final_stat_df.merge(random_df, on = ['chai_patient_id', 'line_start'])

    final_stat_df_lab = cond_table[['chai_patient_id']].drop_duplicates()
    unique_labs = list(set(unique_labs))
    for lab in unique_labs:
        columns = [x for x in final_stat_df.columns if lab in x]
        codes = list(set([x.split('_')[-3] for x in columns]))
        df_list = []
        for code in codes:
            sub_col = [x for x in final_stat_df.columns if code in x]
            sub_final_stat_df = final_stat_df[sub_col+['chai_patient_id']]
            for col in list(set(sub_final_stat_df.columns)-set(['chai_patient_id'])):
                sub_final_stat_df = sub_final_stat_df.rename(columns={col:'_'.join(col.split('_')[:-3]+col.split('_')[-2:])})
            df_list.append(sub_final_stat_df)
        with open('df_list','wb') as f:
            pickle.dump(df_list,f)
        df_lab = pd.concat(df_list,ignore_index=True)
        df_lab = df_lab.groupby(by='chai_patient_id').mean()
        final_stat_df_lab = final_stat_df_lab.merge(df_lab,on='chai_patient_id',how='left')
        
    return final_stat_df_lab

def get_censored_patients(schema, lot_table):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    patient = dtu.redshift_to_df(
        '''SELECT chai_patient_id,date_of_death FROM {schema}.patient WHERE chai_patient_id IN ('{pat_str}')'''.format(schema=schema,
                                                                                           pat_str=pat_str))
    patient = patient[~pd.isna(patient['date_of_death'])][['chai_patient_id', 'date_of_death']]
#     lot_table = lot_table.merge(patient, on='chai_patient_id', how='left')
#     lot_table['censored'] = 'no'
#     lot_table.loc[lot_table.apply(lambda x: x['line_end'] == x['date_of_death'], axis=1), 'censored'] = 'yes'
 
    TABLE_WISE_DATE_COLUMN = {
        'medication': 'med_start_date',
        'patient_test': 'test_date',
        'tumor_exam': 'exam_date',
        'care_goal': 'treatment_plan_start_date',
        'surgery': 'surgery_date',
        'radiation': 'rad_start_date',
        'condition': 'diagnosis_date',
        'adverse_event': 'adverse_event_date',
        'encounter': 'encounter_date',
        'disease_status': 'assessment_date',
        'staging': 'stage_date'
    }
    date_columns = list(TABLE_WISE_DATE_COLUMN.values())
    date_df = pd.DataFrame({'chai_patient_id': patient_ids})
    for table, date_column in TABLE_WISE_DATE_COLUMN.items():
        date_df_ = dtu.redshift_to_df(
            '''SELECT chai_patient_id, max({date_column}) as {date_column} FROM {schema}.{table} 
            WHERE chai_patient_id IN ('{pat_str}') group by chai_patient_id'''.format(schema=schema,
                                                             pat_str=pat_str,
                                                             table = table,
                                                             date_column = date_column))
        if date_df_[date_column].dtype==np.object:
            date_df_[date_column] = date_df_[date_column].astype(np.datetime64).fillna(pd.NaT)
        date_df = date_df.merge(date_df_, on = ['chai_patient_id'], how = 'left')
    date_df['last_available_date'] = date_df.loc[:, date_columns].max(axis = 1)
    
    date_df = date_df.merge(patient,on='chai_patient_id',how='outer')
    date_df['last_available_date'] = date_df.apply(lambda x:x['date_of_death'] if (x['date_of_death']<x['last_available_date']) else x['last_available_date'],axis=1)
    
    lot_table = lot_table.merge(date_df[['chai_patient_id', 'last_available_date']],
                            on = ['chai_patient_id'], how = 'left')
#     lot_table.loc[lot_table.apply(lambda x: x['line_end'] == x['last_available_date'], axis=1), 'censored'] = 'yes'
    lot_table = lot_table[['chai_patient_id', 'last_available_date']].drop_duplicates()
    return lot_table

def get_demographics(schema, lot_table):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    demographics = dtu.redshift_to_df('''SELECT chai_patient_id, date_of_birth,
                                        ethnicity, gender, race FROM {schema}.patient 
                                        WHERE chai_patient_id IN ('{pat_str}')'''.format(schema=schema,
                                                                                     pat_str=pat_str))
    demographics = demographics.merge(lot_table[['chai_patient_id','line_start']],on='chai_patient_id',how='inner')
    demographics['age'] = demographics.line_start.apply(lambda x: x.year) - \
                            demographics.date_of_birth.apply(lambda x: x.year)
    return demographics.drop(columns = ['date_of_birth'], axis = 1).drop_duplicates()


def get_radiation(schema, lot_table, cohort):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)  
    if cohort in nlp_tables:
        if cohort == 'Lung':
            radiation_table = nlp_tables[cohort]['radiation']
        else:
            radiation_table = nlp_tables[cohort]['schema'] + '.radiation'
        radiation_nlp = dtu.redshift_to_df("""select * from {radiation_table}
                                        where chai_patient_id in ('{pat_str}')""".format(radiation_table = radiation_table,
                                                pat_str = pat_str))
        radiation_table = schema + '.radiation'
        radiation_pt360 = dtu.redshift_to_df("""select * from {radiation_table} where
                                         chai_patient_id in ('{pat_str}')""".format(radiation_table = radiation_table,
                                                pat_str = pat_str))
#         radiation_pt360 = radiation_pt360[radiation_pt360['curation_indicator']==0]
        radiation_pt360 = radiation_pt360[~radiation_pt360['chai_patient_id'].isin(radiation_nlp['chai_patient_id'].tolist())]
        rad = pd.concat([radiation_pt360, radiation_pt360])
    else:
        rad = dtu.redshift_to_df("""select * from {radiation_table} where
                                         chai_patient_id in ('{pat_str}')""".format(radiation_table = schema + '.radiation',
                                                pat_str = pat_str))
#         rad = rad[rad['curation_indicator']==0]
    
    rad = rad.merge(lot_table,on=['chai_patient_id'],how='inner')
    rad = rad[rad.apply(lambda x:(x['random_point']-90)<=(x['rad_start_date']-x['line_start']).days<=x['random_point'],axis=1)]
    rad_pats = rad[~rad.modality_standard_name.isnull()].chai_patient_id.unique()
    non_rad = lot_table[~lot_table.chai_patient_id.isin(rad.chai_patient_id.unique())].chai_patient_id.unique()
    rad_df = pd.DataFrame({'chai_patient_id':rad_pats,'radiation':[1 for i in range (0,len(rad_pats))]})
    non_rad_df = pd.DataFrame({'chai_patient_id':non_rad,'radiation':[0 for i in range (0,len(non_rad))]})
    all_rad = pd.concat([rad_df, non_rad_df])
    return all_rad[['chai_patient_id', 'radiation']]


def get_stage(schema, cond_table, cohort):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
    patient_ids = list(set(cond_table['chai_patient_id']))
    df_list = []
    if cohort in nlp_tables:
        if cohort == 'Lung':
            stage_table = nlp_tables[cohort]['staging']
        else:
            stage_table = nlp_tables[cohort]['schema'] + '.staging'
        stage_df_nlp = dtu.redshift_to_df('''SELECT chai_patient_id, stage_group_standard_name, tstage_standard_name, 
            mstage_standard_name, nstage_standard_name,stage_date, curation_indicator from {stage_table}
            WHERE chai_patient_id in ({pats_list})'''.format(stage_table = stage_table,
                                                         pats_list=','.join(["'" + str(pat) + "'" for pat in patient_ids]
                                                        )))
        stage_table = schema + '.staging'
        stage_df_pt360 = dtu.redshift_to_df('''SELECT chai_patient_id, stage_group_standard_name, tstage_standard_name, 
            mstage_standard_name, nstage_standard_name,stage_date, curation_indicator from {stage_table}
            WHERE chai_patient_id in ({pats_list})'''.format(stage_table = stage_table,
                                                         pats_list=','.join(["'" + str(pat) + "'" for pat in patient_ids]
                                                        )))
        
        stage_df_pt360 = stage_df_pt360[~stage_df_pt360['chai_patient_id'].isin(stage_df_nlp['chai_patient_id'].tolist())]
        stage_df = pd.concat([stage_df_pt360, stage_df_nlp]) 
    else:
        stage_table = schema + '.staging'
        stage_df = dtu.redshift_to_df('''SELECT chai_patient_id, stage_group_standard_name, tstage_standard_name, 
            mstage_standard_name, nstage_standard_name,stage_date, curation_indicator from {stage_table}
            WHERE chai_patient_id in ({pats_list})'''.format(stage_table = stage_table,
                                                         pats_list=','.join(["'" + str(pat) + "'" for pat in patient_ids]
                                                        )))
    
    # print(f"stage curation value count : {stage_df['curation_indicator'].value_counts()}")
    # # stage_df = stage_df[stage_df['curation_indicator']==0]
    # stage_df = stage_df[~((stage_df['curation_indicator']==1) & (stage_df['chai_patient_id'].isin(non_curation_pat_list)))]
        
    cond_table_merged = cond_table.merge(stage_df,on='chai_patient_id',how='inner').drop_duplicates()
    stage = ['X', '0', '1', '2', '3', '4']
    s_count = -1
    cond_table_merged['stage'] = np.nan
    cond_table_merged['tstage'] = np.nan
    cond_table_merged['nstage'] = np.nan
    cond_table_merged['mstage'] = np.nan
    for s in stage:
        s_count += 1
        cond_table_merged.loc[
            cond_table_merged.stage_group_standard_name.str.contains(s,case=False,na=False),'stage'] = s_count
        cond_table_merged.loc[
            cond_table_merged.tstage_standard_name.str.contains(s,case=False,na=False),'tstage'] = s_count
        cond_table_merged.loc[
            cond_table_merged.nstage_standard_name.str.contains(s,case=False,na=False),'nstage'] = s_count
        cond_table_merged.loc[
            cond_table_merged.mstage_standard_name.str.contains(s,case=False,na=False),'mstage'] = s_count
    cond_table_merged=cond_table_merged[~cond_table_merged['stage_date'].isnull()].drop_duplicates()
#     lot_all_dates_merged = lot_all_dates_merged[lot_all_dates_merged.apply(lambda x:x['stage_date']<=x['line_end'],axis=1)]
#     lot_all_dates_merged = lot_all_dates_merged[lot_all_dates_merged.apply(lambda x:x['stage_date']>=x['line_start'],axis=1)]

    # cond_table_merged = cond_table_merged[cond_table_merged.apply(lambda x:(0<=(x['random_date']-x['stage_date']).days<90),axis=1)]
    # cond_table_merged = cond_table_merged[cond_table_merged.apply(lambda x:(0<=(x['random_date']-x['stage_date']).days<180),axis=1)]
    
    cond_table_merged_stage = cond_table_merged.dropna(subset=['stage', 'tstage', 'mstage', 'nstage'], how = 'all')
    stage_df = cond_table_merged_stage[['chai_patient_id', 'stage', 'tstage', 'mstage', 'nstage']]
    stage_df['stage'] = stage_df.groupby(['chai_patient_id'])['stage'].apply(lambda x: x.fillna(x.max()))
    stage_df['tstage'] = stage_df.groupby(['chai_patient_id'])['tstage'].apply(lambda x: x.fillna(x.max()))
    stage_df['nstage'] = stage_df.groupby(['chai_patient_id'])['nstage'].apply(lambda x: x.fillna(x.max()))
    stage_df['mstage'] = stage_df.groupby(['chai_patient_id'])['mstage'].apply(lambda x: x.fillna(x.max()))
    stage_df = stage_df.drop_duplicates(subset = ['chai_patient_id'])
    return stage_df



def get_tumor_exam(schema, lot_table, cohort):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    if cohort in nlp_tables:
        if cohort == 'Lung':
            tumor_exam_table = nlp_tables[cohort]['tumor_exam']
        else:
            tumor_exam_table = nlp_tables[cohort]['schema'] + '.tumor_exam'
        tumor_exam_nlp = dtu.redshift_to_df("""select chai_patient_id, exam_date, tumor_grade_standard_name, 
                                                tumor_histology_standard_name from {tumor_exam_table}
                                                where chai_patient_id in ('{pat_str}')""".format(tumor_exam_table = tumor_exam_table,
                                                pat_str = pat_str))
        tumor_exam_table = schema + '.tumor_exam'
        tumor_exam_pt360 = dtu.redshift_to_df("""select chai_patient_id, exam_date, tumor_grade_standard_name,
                                                tumor_size, tumor_histology_standard_name from {tumor_exam_table}
                                                where chai_patient_id in ('{pat_str}')""".format(tumor_exam_table = tumor_exam_table,
                                                pat_str = pat_str))
        # tumor_exam_pt360 = tumor_exam_pt360[tumor_exam_pt360['curation_indicator']==0]
        tumor_exam_nlp = tumor_exam_nlp.merge(tumor_exam_pt360[['chai_patient_id', 'tumor_size']], 
                                              on = ['chai_patient_id'], how = 'left')
        tumor_exam_pt360 = tumor_exam_pt360[~tumor_exam_pt360['chai_patient_id'].isin(tumor_exam_nlp['chai_patient_id'])]
        grade = pd.concat([tumor_exam_nlp, tumor_exam_pt360])
    else:
        tumor_exam_table = schema + '.tumor_exam'
        grade = dtu.redshift_to_df("""select chai_patient_id, exam_date, tumor_grade_standard_name, tumor_size,
                                        tumor_histology_standard_name from {tumor_exam_table} where
                                         chai_patient_id in ('{pat_str}')""".format(tumor_exam_table = schema + '.tumor_exam',
                                                pat_str = pat_str))
        # grade = grade[grade['curation_indicator']==0]
    
    lot_all_dates_merged = lot_table.merge(grade,on='chai_patient_id',how='inner').drop_duplicates()
    stage = ['X', '1', '2', '3', '4']
    s_count = -1
    lot_all_dates_merged['grade'] = np.nan
    for s in stage:
        s_count += 1
        lot_all_dates_merged.loc[
            lot_all_dates_merged.tumor_grade_standard_name.str.contains(s,case=False,na=False),'grade'] = s_count
    lot_all_dates_merged=lot_all_dates_merged[~lot_all_dates_merged['exam_date'].isnull()].drop_duplicates()
#     lot_all_dates_merged = lot_all_dates_merged[lot_all_dates_merged.apply(lambda x:x['exam_date']<=x['line_end'],axis=1)]
#     lot_all_dates_merged = lot_all_dates_merged[lot_all_dates_merged.apply(lambda x:x['exam_date']>=x['line_start'],axis=1)]
    lot_all_dates_merged = lot_all_dates_merged[lot_all_dates_merged.apply(lambda x:(x['random_point']-90)<=(x['exam_date']-x['line_start']).days<=(x['random_point']),axis=1)]
    lot_all_dates_stage = lot_all_dates_merged#.dropna(subset=['grade', 'tumor_size'], how = 'all')
    tumor_exam_df = lot_all_dates_stage[['chai_patient_id','line_start','grade',
                                         'tumor_size', 'tumor_histology_standard_name']]
    tumor_exam_df['tumor_size'] = tumor_exam_df['tumor_size'].astype(float)
    tumor_exam_df['grade'] = tumor_exam_df.groupby(['chai_patient_id','line_start'])['grade'].apply(lambda x: x.fillna(x.max()))
    tumor_exam_df['tumor_size'] = tumor_exam_df.groupby(['chai_patient_id','line_start'])['tumor_size'].apply(lambda x: x.fillna(x.max()))
    tumor_exam_df = tumor_exam_df.drop_duplicates(subset = ['chai_patient_id', 'line_start'])
    #histology
    histology_list = list(tumor_exam_df.tumor_histology_standard_name.unique())
    histology_df = pd.DataFrame(columns=['chai_patient_id', 'line_start'] + histology_list)
    histology_df[['chai_patient_id', 'line_start']] = lot_table[['chai_patient_id', 'line_start']]
    for i,j in histology_df.iterrows():
        pat_df = tumor_exam_df[(tumor_exam_df['chai_patient_id']==j['chai_patient_id'])&(tumor_exam_df['line_start']==j['line_start'])]
        histology_list_pat = list(pat_df.tumor_histology_standard_name.unique())
        histology_df.loc[(histology_df['chai_patient_id']==j['chai_patient_id'])&(histology_df['line_start']==j['line_start']),histology_list_pat]=1
    histology_df = histology_df.drop_duplicates()
    histology_df = histology_df.fillna(0)
    
    tumor_exam_df = tumor_exam_df.merge(histology_df, on = ['chai_patient_id', 'line_start'], how = 'left')
    tumor_exam_df = tumor_exam_df.drop(['tumor_histology_standard_name'], axis = 1)
    return tumor_exam_df.drop([np.nan], axis = 1)

def get_encounter(schema, lot_table):
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    encounter = dtu.redshift_to_df('''select chai_patient_id, encounter_date, encounter_status,
                                        encounter_name_standard_name, encounter_value_standard_name
                                        from {schema}.encounter where chai_patient_id in ('{pat_str}')'''.format(schema = schema, 
                                                                                                              pat_str = pat_str))
#     encounter = encounter[encounter['curation_indicator']==0]
    encounter = encounter[encounter['encounter_date'] >= '2020-01-01']
    df = pd.DataFrame(encounter.groupby(by = ['chai_patient_id','encounter_name_standard_name'], as_index=False)['encounter_date'].count())
    encounter_list = list(df['encounter_name_standard_name'].unique())
    encounter_df = df.pivot(index='chai_patient_id', columns='encounter_name_standard_name', values='encounter_date')
    encounter_df.fillna(0, inplace=True)
    encounter_df['Total_encounters'] = encounter_df.loc[:,encounter_list].sum(axis=1)
    return encounter[['chai_patient_id', 'Total_encounters']]

def add_additional_lines(lot_table_name):
    lot_table = dtu.redshift_to_df('''select * from {lot_table_name}'''.format(lot_table_name = lot_table_name))
    lot_table = lot_table[lot_table['index_date']>='2020-01-01']
    line_0to1 = pd.DataFrame()
    line_0to1[['chai_patient_id', 'line_start', 'line_end']] = lot_table[lot_table['line_number']==1][['chai_patient_id', 'index_date', 'line_start']]
    line_0to1['day_diff'] = line_0to1['line_end'] - line_0to1['line_start']
    line_0to1 = line_0to1[(line_0to1['day_diff'].dt.days<=30)&(line_0to1['day_diff'].dt.days>0)].drop(columns = ['day_diff'], axis = 1)
    line_0to1['line_number'] = 0.5
    
    line_1to2 = pd.DataFrame()
    chai_id2 = lot_table[lot_table['line_number']==2]['chai_patient_id'].unique()
    temp = lot_table[(lot_table['chai_patient_id'].isin(chai_id2))&(lot_table['line_number'].isin([1,2]))].drop_duplicates()
    line_1to2[['chai_patient_id', 'line_start']] = temp[temp['line_number']==1][['chai_patient_id', 'line_end']].drop_duplicates().dropna()
    line_1to2 = line_1to2.merge(temp[temp['line_number']==2][['chai_patient_id', 'line_start']],
                    on = ['chai_patient_id'], how = 'left',
                    suffixes = ("", "_y")).drop_duplicates().rename(columns = {'line_start_y': 'line_end'})
    line_1to2['line_number'] = 1.5
    line_2to3 = pd.DataFrame()
    chai_id3 = lot_table[lot_table['line_number']==3]['chai_patient_id'].unique()
    temp = lot_table[(lot_table['chai_patient_id'].isin(chai_id3))&(lot_table['line_number'].isin([2,3]))].drop_duplicates()
    line_2to3[['chai_patient_id', 'line_start']] = temp[temp['line_number']==2][['chai_patient_id', 'line_end']].drop_duplicates().dropna()
    line_2to3 = line_2to3.merge(temp[temp['line_number']==3][['chai_patient_id', 'line_start']],
                                           on = ['chai_patient_id'],
                    how = 'left',
                    suffixes = ("", "_y")).drop_duplicates().rename(columns = {'line_start_y': 'line_end'})
    line_2to3['line_number'] = 2.5
    
    return pd.concat([lot_table[['chai_patient_id', 'line_start', 'line_end', 'line_number']],
                     line_0to1, line_1to2, line_2to3]).drop_duplicates()
    

def get_medication_for_additional_lines(schema, lot_table):
    medication_df = pickle.load(open('data/medication_df.pkl', 'rb'))
    med_list = list(medication_df.columns)[2:]
    patient_ids = list(set(lot_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    meddf = dtu.redshift_to_df('''select chai_patient_id, med_name_standard_name, med_start_date
    from {schema}.medication where med_name_standard_name in ('{med_list}')
    and chai_patient_id in ('{pat_str}')
     '''.format(schema = schema, pat_str = pat_str,
               med_list = "','".join(med_list))).drop_duplicates()
#     meddf = meddf[meddf['curation_indicator']==0]
    meddf['med_name_standard_name'] = meddf['med_name_standard_name'].str.lower()
    meddf = meddf.merge(lot_table, on = ['chai_patient_id'], how = 'inner')
    meddf = meddf[~meddf['med_start_date'].isna()]
    meddf = meddf[meddf.apply(lambda x:(abs(x['line_start']-x['med_start_date'])).days<30,axis=1)]
    
    #med_list = list(meddf.med_name_standard_name.unique())
    med_df = pd.DataFrame(columns=['chai_patient_id', 'line_start'] + med_list)
    med_df[['chai_patient_id', 'line_start']] = lot_table[['chai_patient_id', 'line_start']]
    for i,j in med_df.iterrows():
        pat_df = meddf[(meddf['chai_patient_id']==j['chai_patient_id'])&(meddf['line_start']==j['line_start'])]
        med_list_pat = list(pat_df.med_name_standard_name.unique())
        med_df.loc[(med_df['chai_patient_id']==j['chai_patient_id'])&(med_df['line_start']==j['line_start']),med_list_pat]=1
    med_df = med_df.drop_duplicates()
    med_df = med_df.fillna(0)
    return med_df    
    
def get_disease_status_prog(schema, cond_table):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
    patient_ids = list(set(cond_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    disease_status = dtu.redshift_to_df('''SELECT chai_patient_id,assessment_date,
                            assessment_value_standard_name, assessment_name_standard_name from
                                 {schema}.disease_status
                                WHERE chai_patient_id IN ('{pat_str}')'''.format(schema=schema, 
                                                         pat_str=pat_str))
    disease_status = disease_status[disease_status['assessment_name_standard_name']=='Finding related to therapeutic response']
    disease_status = disease_status[disease_status['assessment_value_standard_name']=='Tumor progression']
    return disease_status
    
def get_disease_status(schema, cond_table):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
    patient_ids = list(set(cond_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    disease_status = dtu.redshift_to_df('''SELECT chai_patient_id,assessment_date,
                            assessment_value_standard_name, assessment_name_standard_name from
                                 {schema}.disease_status
                                WHERE chai_patient_id IN ('{pat_str}')'''.format(schema=schema, 
                                                         pat_str=pat_str))
    
    # print(f"disease status curation indicator : 1")
    # disease_status = disease_status[~disease_status['chai_patient_id'].isin(non_curation_pat_list)]
    # disease_status = disease_status[disease_status['assessment_name_standard_name']=='Finding related to therapeutic response']
#     disease_status = disease_status[disease_status['assessment_value_standard_name']!='Tumor progression']

    disease_status = disease_status.drop_duplicates()
    disease_status = disease_status[~disease_status['assessment_value_standard_name'].isna()].drop('assessment_name_standard_name', axis = 1)
    disease_status = disease_status[~disease_status['assessment_date'].isna()]
    assessement_value_list = list(disease_status['assessment_value_standard_name'].unique())
    col_renamed = {col:'stable_disease_'+ col for col in assessement_value_list}
    ds_cond = disease_status.merge(cond_table, on = ['chai_patient_id'], how = 'inner')
#     ds_lot = ds_lot[ds_lot['assessment_date']>=ds_lot['line_start']]
    ds_cond['diff'] = ds_cond.apply(lambda x: (x['random_date'] - x['assessment_date']).days, axis=1)
    ds_cond.loc[:, 'rand_wind'] = 'na'
    ds_cond.loc[(ds_cond['diff'] >= 0) & (ds_cond['diff'] < 45), 'rand_wind'] = '0_45'
    ds_cond.loc[(ds_cond['diff'] >= 45) & (ds_cond['diff'] < 180), 'rand_wind'] = '45_180'
#     ds_cond.loc[(ds_cond['diff'] >= (ds_cond['random_point'] - 60)) & (
#             ds_lot['diff'] < ds_lot['random_point'] - 30), 'rand_wind'] = '30_60'
    # ds_cond.loc[(ds_cond['diff'] >= (ds_cond['random_point'] - 90)) & (
    #             ds_cond['diff'] < ds_cond['random_point'] - 60), 'rand_wind'] = '60_90'
    final_df = ds_cond[['chai_patient_id']].drop_duplicates()
    for wind in ['0_45', '45_180']:
        df_temp = ds_cond[ds_cond['rand_wind']==wind]
        df_temp = df_temp.groupby(['chai_patient_id',
                               'assessment_value_standard_name'], as_index = False)['assessment_date'].count()
        df_temp = df_temp.pivot( index = ['chai_patient_id'],
                      columns = 'assessment_value_standard_name',
                      values = ['assessment_date'])
        df_temp.columns = [col[1] for col in df_temp.columns.values]
        df_temp.reset_index(inplace = True)
        df_temp.rename(columns = {'Complete therapeutic response': 'Complete therapeutic response_wind_' + wind,
                                 'Partial therapeutic response': 'Partial therapeutic response_wind_' + wind,
                                  'Stable': 'Stable_wind_' + wind,
                                  'Tumor progression': 'Tumor progression_wind_' + wind
                                 }, inplace = True)
        
        df_temp = df_temp.fillna(0)
        col_list = []
        for ele in ['Complete therapeutic response_wind_' + wind,'Partial therapeutic response_wind_' + wind,'Stable_wind_' + wind,'Tumor progression_wind_' + wind]:
            if ele in df_temp.columns:
                col_list.append(ele)
        # df_temp[col_list] = 0
        df_temp = df_temp[['chai_patient_id']+col_list]
        final_df = final_df.merge(df_temp, on = ['chai_patient_id'], how = 'left').fillna(0)
    return final_df

## Considers only TP53 
def get_biomarker(gn_schema,pt_schema,cond_table,cohort):
    biomarker_df = dtu.redshift_to_df('''SELECT chai_patient_id,curation_indicator,biomarker_name_name,biomarker_variant_type_name,test_result_name, report_date,specimen_date FROM {schema}.biomarker'''.format(schema=gn_schema))
    
    # biomarker_df = biomarker_df[biomarker_df['curation_indicator']==0]
    
    biomarker_df['biomarker_variant_type_name'] = biomarker_df['biomarker_variant_type_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
    biomarker_df = biomarker_df[(biomarker_df['biomarker_name_name']=='TP53') & 
                            (biomarker_df['biomarker_variant_type_name'].isin(['gene mutation', 'small indel']))]
    biomarker_df = biomarker_df.rename(columns={'chai_patient_id':'chai_pt360_id'})
    Crosswalk = dtu.redshift_to_df('''SELECT * FROM {schema}.master_chai_id_map'''.format(schema='c3_gen2_cdm_202308'))
    biomarker_df = biomarker_df.merge(Crosswalk,on='chai_pt360_id',how='inner')
    
    genome_df = dtu.redshift_to_df('''SELECT chai_patient_id,curation_indicator, biomarker_name_name,biomarker_variant_type_name, test_result_name, report_date,specimen_date FROM {schema}.genetic_test'''.format(schema=gn_schema))
    
    # genome_df = genome_df[genome_df['curation_indicator']==0]
    
    genome_df['biomarker_variant_type_name'] = genome_df['biomarker_variant_type_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
    genome_df = genome_df[(genome_df['biomarker_name_name']=='TP53') & 
                          (genome_df['biomarker_variant_type_name'].isin(['gene mutation', 'small indel']))]
    genome_df = genome_df.rename(columns={'chai_patient_id':'chai_pt360_id'})
    genome_df = genome_df.merge(Crosswalk,on='chai_pt360_id',how='inner')
    
    biomarker_df = pd.concat([biomarker_df,genome_df]).drop_duplicates()
    
    if cohort=='Lung':
        biomarker_df_pt360 = dtu.redshift_to_df('''SELECT chai_patient_id,curation_indicator,biomarker_name_standard_name,\
biomarker_variant_type_standard_name,test_result_standard_name,report_date FROM {schema}.biomarker'''.format(schema=pt_schema))
        
        # biomarker_df_pt360 = biomarker_df_pt360[biomarker_df_pt360['curation_indicator']==0]
        
        biomarker_df_pt360['biomarker_variant_type_standard_name'] = biomarker_df_pt360['biomarker_variant_type_standard_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)     
        biomarker_df_pt360 = biomarker_df_pt360[(biomarker_df_pt360['biomarker_name_standard_name']=='TP53') & 
                            (biomarker_df_pt360['biomarker_variant_type_standard_name'].isin(['gene mutation', 'small indel']))]
        biomarker_df_pt360 = biomarker_df_pt360.rename(columns={'biomarker_name_standard_name':'biomarker_name_name',
                                                      'biomarker_variant_type_standard_name':'biomarker_variant_type_name',
                                                               'test_result_standard_name':'test_result_name'})
        biomarker_df = biomarker_df[['chai_patient_id','biomarker_name_name','biomarker_variant_type_name','test_result_name', 'report_date']]
        biomarker_df = pd.concat([biomarker_df,biomarker_df_pt360]).drop_duplicates()
    
    
    
    def convert_to_class(x):
        if x=='Positive':
            return 1
        elif x=='Negative':
            return -1
        else:
            return 0
    biomarker_df['tp53_mutation'] = biomarker_df['test_result_name'].apply(convert_to_class)
    
    cond_table = cond_table.merge(biomarker_df,on='chai_patient_id',how='left')
    cond_table['diff'] = cond_table.apply(lambda x:(x['random_date']-x['report_date']).days,axis=1)
    cond_table = cond_table[cond_table['diff']>=0].sort_values(by=['report_date'],ascending=False)
    cond_table = cond_table.drop_duplicates(subset=['chai_patient_id'],keep='first')
    return cond_table[['chai_patient_id','tp53_mutation']]

    
## Considers multiple biomarker
def get_all_biomarker(pt_schema,cond_table,cohort):
#     non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
#     biomarker_df = dtu.redshift_to_df('''SELECT chai_patient_id,biomarker_name_name,biomarker_variant_type_name,test_result_name, report_date,specimen_date,curation_indicator FROM {schema}.biomarker'''.format(schema=gn_schema))
    
#     print(f"genome biomarker curation value count : {biomarker_df['curation_indicator'].value_counts()}")
#     biomarker_df = biomarker_df[~((biomarker_df['curation_indicator']==1) & (biomarker_df['chai_patient_id'].isin(non_curation_pat_list)))]
    
#     # biomarker_df['biomarker_variant_type_name'] = biomarker_df['biomarker_variant_type_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
# #     biomarker_df = biomarker_df[(biomarker_df['biomarker_name_name']=='TP53') & 
# #                             (biomarker_df['biomarker_variant_type_name'].isin(['gene mutation', 'small indel']))]
#     biomarker_df = biomarker_df.rename(columns={'chai_patient_id':'chai_pt360_id'})
#     Crosswalk = dtu.redshift_to_df('''SELECT * FROM {schema}.master_chai_id_map'''.format(schema='c3_gen2_cdm_202308'))
#     biomarker_df = biomarker_df.merge(Crosswalk,on='chai_pt360_id',how='inner')
    
#     genome_df = dtu.redshift_to_df('''SELECT chai_patient_id,biomarker_name_name,biomarker_variant_type_name,test_result_name, report_date,specimen_date,curation_indicator FROM {schema}.genetic_test'''.format(schema=gn_schema))
    
#     print(f"genome genetic_test curation value count : {genome_df['curation_indicator'].value_counts()}")
#     genome_df = genome_df[~((genome_df['curation_indicator']==1) & (genome_df['chai_patient_id'].isin(non_curation_pat_list)))]
    
#     # genome_df['biomarker_variant_type_name'] = genome_df['biomarker_variant_type_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
# #     genome_df = genome_df[(genome_df['biomarker_name_name']=='TP53') & 
# #                           (genome_df['biomarker_variant_type_name'].isin(['gene mutation', 'small indel']))]
#     genome_df = genome_df.rename(columns={'chai_patient_id':'chai_pt360_id'})
#     genome_df = genome_df.merge(Crosswalk,on='chai_pt360_id',how='inner')
    
#     biomarker_df = pd.concat([biomarker_df,genome_df]).drop_duplicates()
    
    biomarker_df_pt360 = dtu.redshift_to_df('''SELECT chai_patient_id,biomarker_name_standard_name,\
biomarker_variant_type_standard_name,test_result_standard_name,report_date,curation_indicator FROM {schema}.biomarker'''.format(schema=pt_schema))

    # print(f"biomarker curation value count : {biomarker_df_pt360['curation_indicator'].value_counts()}")
    # biomarker_df_pt360 = biomarker_df_pt360[~((biomarker_df_pt360['curation_indicator']==1) & (biomarker_df_pt360['chai_patient_id'].isin(non_curation_pat_list)))]

    biomarker_df_pt360['biomarker_variant_type_standard_name'] = biomarker_df_pt360['biomarker_variant_type_standard_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
#         biomarker_df_pt360 = biomarker_df_pt360[(biomarker_df_pt360['biomarker_name_standard_name']=='TP53') & 
#                             (biomarker_df_pt360['biomarker_variant_type_standard_name'].isin(['gene mutation', 'small indel']))]
    biomarker_df_pt360 = biomarker_df_pt360.rename(columns={'biomarker_name_standard_name':'biomarker_name_name',
                                                  'biomarker_variant_type_standard_name':'biomarker_variant_type_name',
                                                           'test_result_standard_name':'test_result_name'})
    
    biomarker_df_pt360 = biomarker_df_pt360[['chai_patient_id','biomarker_name_name','biomarker_variant_type_name','test_result_name', 'report_date']]
    biomarker_df = biomarker_df_pt360.drop_duplicates()
    # biomarker_df = pd.concat([biomarker_df,biomarker_df_pt360]).drop_duplicates()
    
    
    def convert_to_class(x):
        if x in ['Deficient','Detection','High','Messenger RNA Overexpression','Messenger RNA Underexpression','Positive']:
            return 1
        elif x in ['Low','Negative','Negative for promoter variants','Negative for selected exons','Negative/Unclassified','Normal','Not Detected','Proficient','Stable']:
            return -1
        elif x in ['Equivocal','Borderline','Inconclusive (qualifier value)','Indeterminate','Indeterminate (qualifier value)',
  'Insufficient Sample','Intermediate','Medium','Not Recorded','Pending','Technical problems','Test Not Done',
  'Unknown (qualifier value)','veristrat good','veristrat poor']:
            return 0
        else:
            return -99
    
    biomarker_df['biomarker_name_name'] = biomarker_df['biomarker_name_name'].apply(lambda x:'TMB' if (x=='Tumor Mutation Burden') else x)
    biomarker_feat_df = cond_table[['chai_patient_id']]
    for biomarker_ele in biomarker_list:
        temp_df = biomarker_df[biomarker_df['biomarker_name_name']==biomarker_ele]
        if temp_df.shape[0]>0:
            temp_df[f'biomarker_{biomarker_ele}'] = temp_df['test_result_name'].apply(convert_to_class)
            temp_df = temp_df.merge(cond_table[['chai_patient_id','random_date']],on='chai_patient_id',how='inner')
            temp_df['diff'] = temp_df.apply(lambda x:(x['random_date']-x['report_date']).days,axis=1)
            temp_df = temp_df[temp_df['diff']>=0].sort_values(by=['report_date'],ascending=False)
            temp_df = temp_df.drop_duplicates(subset=['chai_patient_id'],keep='first')
            biomarker_feat_df = biomarker_feat_df.merge(temp_df[['chai_patient_id',f'biomarker_{biomarker_ele}']],on=['chai_patient_id'],how='left')
        else:
            biomarker_feat_df[f'biomarker_{biomarker_ele}'] = biomarker_feat_df.apply(lambda x:np.nan)
    
    return biomarker_feat_df
    
    
def get_imaging(schema, cond_table):
    # non_curation_pat_list = list(cond_table[cond_table['curation']==0]['chai_patient_id'].unique())
    patient_ids = list(set(cond_table['chai_patient_id']))
    pat_str = "','".join(patient_ids)
    imaging = dtu.redshift_to_df('''select chai_patient_id, curation_indicator, report_date, imaging_type_standard_name
from {schema}.imaging  where chai_patient_id IN ('{pat_str}')'''.format(schema = schema,
                                                                       pat_str=pat_str))
    
    # print(f"imaging curation value count : {imaging['curation_indicator'].value_counts()}")
    # # imaging = imaging[imaging['curation_indicator']==0]
    # imaging = imaging[~((imaging['curation_indicator']==1) & (imaging['chai_patient_id'].isin(non_curation_pat_list)))]
    if imaging.shape[0]>0:
        imaging = imaging.merge(cond_table,on='chai_patient_id',how='inner')
        imaging['diff'] = imaging.apply(lambda x:(x['random_date']-x['report_date']).days,axis=1)
        imaging = imaging[(imaging['diff']>=0) & (imaging['diff']<180)]

        imaging['imaging_type'] = 'other'
        imaging.loc[(imaging['imaging_type_standard_name']=='Computed tomography'), 'imaging_type']='CT'

        imaging.loc[(imaging['imaging_type_standard_name']=='Positron emission tomography with computed tomography'), 'imaging_type']='PET'
        imaging.loc[(imaging['imaging_type_standard_name']=='Positron emission tomography'), 'imaging_type']='PET'
        imaging.loc[(imaging['imaging_type_standard_name']=='MRI with contrast'), 'imaging_type' ]='MRI'
        imaging.loc[(imaging['imaging_type_standard_name']=='Magnetic resonance imaging'), 'imaging_type']='MRI'
        imaging.loc[(imaging['imaging_type_standard_name']=='MRI without contrast'), 'imaging_type']='MRI'
        imaging.loc[(imaging['imaging_type_standard_name']=='Radioisotope scan of bone'), 'imaging_type']='Radioisotope'
    else:
        imaging['imaging_type'] = 'other'
    imaging_df = cond_table[['chai_patient_id']]
    for img in ['CT','PET','MRI','Radioisotope']:
        img_pat_list = imaging[imaging['imaging_type']==img]['chai_patient_id'].unique()
        imaging_df[img] = imaging_df['chai_patient_id'].apply(lambda x:1 if x in img_pat_list else 0)
    
    return imaging_df

    
    