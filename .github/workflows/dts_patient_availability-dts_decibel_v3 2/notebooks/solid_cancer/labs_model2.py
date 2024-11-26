import pandas as pd
import json
import random
import copy
import numpy as np
from data_transfer_utility.application import DataTransferUtility
import yaml

with open('keys_labs.yaml','r') as file:
    lab_kb = yaml.safe_load(file)
    
lab_kb = {c['key']:c for c in lab_kb}

with open('units.yaml','r') as file:
    unite_conv_map = yaml.safe_load(file)
    
unite_conv_map = {c['base']:c for c in unite_conv_map}

# run_labs = ['Heart rate',
#              'Diastolic blood pressure',
#              'Hemoglobin [Mass/volume] in Blood',
#              'Platelets [#/volume] in Blood',
#              'Leukocytes [#/volume] in Blood by Automated count',
#              'ECOG Performance Status score',
#              'Lactate dehydrogenase [Enzymatic activity/volume] in Serum or Plasma',
#              'Calcium [Mass/volume] in Serum or Plasma',
#              'Albumin [Mass/volume] in Serum or Plasma',
#              'Albumin/Globulin [Mass Ratio] in Serum or Plasma',
#              'Beta globulin [Mass/volume] in Serum or Plasma by Electrophoresis',
#              'Beta-2-Microglobulin [Mass/volume] in Serum or Plasma'
#              'Body temperature',
#              'Systolic blood pressure',
#              'Body weight',
#              'Respiratory rate',
#              'Body height',
#              'Erythrocyte distribution width [Ratio]',
#              'Creatinine [Mass/volume] in Serum or Plasma',
#              'Pain severity - Reported',
#              'MCHC [Mass/volume]',
#              'Hematocrit [Volume Fraction] of Blood',
#              'MCV [Entitic volume]',
#              'Erythrocytes [#/volume] in Blood',
#              'Glucose [Mass/volume] in Serum or Plasma',
#              'MCH [Entitic mass]',
#              'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
#              'Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma',
#              'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
#              'Carbon dioxide, total [Moles/volume] in Serum or Plasma',
#              'Protein [Mass/volume] in Serum or Plasma',
#              'Bilirubin.total [Mass/volume] in Serum or Plasma',
#              'Potassium [Moles/volume] in Serum or Plasma',
#              'Chloride [Moles/volume] in Serum or Plasma',
#              'Sodium [Moles/volume] in Serum or Plasma',
#              'Lymphocytes [#/volume] in Blood',
#              'Lymphocytes/100 leukocytes in Blood',
#              'Urea nitrogen [Mass/volume] in Serum or Plasma',
#              'Monocytes/100 leukocytes in Blood',
#              'Neutrophils [#/volume] in Blood',
#              'Monocytes [#/volume] in Blood',
#              'Basophils [#/volume] in Blood',
#              'Platelet mean volume [Entitic volume] in Blood',
#              'Eosinophils [#/volume] in Blood',
#              'Tobacco smoking status',
#              'Nausea [Presence]',
#              'Prostate specific Ag [Mass/volume] in Serum or Plasma',
#              'Karnofsky Performance Status score',
#              'Prothrombin time (PT)',
#              'pH of Urine',
#              'Bilirubin.total [Presence] in Urine',
#              'Glucose [Presence] in Urine',
#              'Appearance of Urine',
#              'Ketones [Presence] in Urine',
#              'Protein [Presence] in Urine',
#              'Leukocytes [Presence] in Urine']

run_labs = ['89247-1']
lab_name_dict = {'89247-1':'ecog'}

# run_labs = [ 'ECOG Performance Status score',
#             'Histology grade [Identifier] in Cancer specimen',
#              'Erythrocyte distribution width [Ratio]',
#              'Oxygen saturation in Arterial blood by Pulse oximetry',
#              'CD274',
#              'Cytoplasmic Ig cells/100 cells in Blood',
#              'EGFR Gene Mutation',
#              'ALK Gene Rearrangement',
#              'ROS1 Gene Rearrangement',
#              'Carcinoembryonic Ag [Mass/volume] in Serum or Plasma',
#              'BRAF Gene Mutation',
#              'Cells.programmed cell death ligand 1/100 viable tumor cells in Tissue by Immune stain',
#              'RET/PTC Rearrangement',
#              'KRAS Gene Mutation']
            
#              'Creatinine [Mass/volume] in Serum or Plasma',
#              'Hemoglobin [Mass/volume] in Blood',
#              'Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma',
#              'Calcium [Mass/volume] in Serum or Plasma',
#              'Albumin [Mass/volume] in Serum or Plasma',
#              'Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
#              'Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma',
#              'Bilirubin.total [Mass/volume] in Serum or Plasma',
#              'Urea nitrogen [Mass/volume] in Serum or Plasma',
#              'Platelets [#/volume] in Blood',
#              'Hematocrit [Volume Fraction] of Blood',
#              'MCV [Entitic volume]',
#              'Estrogen Receptor Status',
#              'Progesterone Receptor Status',
#              'HER2 [Presence] in Tissue by Immunoassay',
#              'Neutrophils [#/volume] in Blood',
#              'Neutrophils/100 leukocytes in Blood',
#              'Histologic grade [Score] in Breast cancer specimen Qualitative by Nottingham',
#              'Glomerular filtration rate/1.73 sq M.predicted among non-blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)',
#              'Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood',
#              'Glomerular filtration rate/1.73 sq M.predicted among blacks [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (MDRD)'
#            ]


lab_dict = {'m_protein_in_serum':['33358-3','51435-6','35559-4','94400-9','33647-9','50796-2','56766-9','44932-2','50792-1'],
            'ecog':['89247-1'],
            'histology':['33732-9']
           }


def convert_unit(df):
    
    def get_key(code):
        key = None
        if code in lab_dict['m_protein_in_serum']:
            key = 'm_protein_in_serum'
        return key
  
    def update_unite(row,key,std_unit):
        if row['key']==key:
            if ((row['test_unit_standard_name']==std_unit) or (row['test_unit_standard_name'].is_in(unite_conv_map[std_unit]['convert'].keys()))):
                row['test_unit_standard_name']=std_unit
            else:
                row['test_unit_standard_name']=None
        return row
    
    df['key'] = df['test_name_standard_code'].apply(get_key)
    df['test_unit_standard_name'] = df['test_unit_standard_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
    df = df.dropna(subset=['test_value_numeric_standard'])
    df.loc[df.apply(lambda x:(x['test_name_standard_code'] in (lab_dict['ecog']+lab_dict['histology'])) and (pd.isna(x['test_unit_standard_name'])),axis=1),'test_unit_standard_name']='valid'

    
    lab_test = df['key'].dropna().unique()
    for key in lab_test:
        if key not in lab_kb.keys():
            raise KeyError(f'Lab test {key} not registered')
        std_unit = lab_kb[key]['attributes']['units']
        unit_keys = list(unite_conv_map[std_unit]['convert'].keys())
        df['test_value_numeric_standard'] = df.apply(lambda x: eval(unite_conv_map[std_unit]['convert'][x['test_unit_standard_name']].split('.')[-1])(x['test_value_numeric_standard'])\
                                 if (x['key']==key) & (x['test_unit_standard_name'] in (unit_keys))\
                                 else x['test_value_numeric_standard'])
        
        df = df.apply(update_unite,axis=1)
        df = df.dopna(subset='test_unit_standard_name')
        
    return df


# def do_unit_conversion(row):
#     if row['test_name_standard_name'] == 'Hemoglobin [Mass/volume] in Blood':
#         if row['test_unit_standard_name'] == 'g/L':
#             row['test_value_numeric_standard'] = row['test_value_numeric_standard'] / 10
#             row['test_unit_standard_name'] = 'g/dL'
#         elif row['test_unit_standard_name'] == 'mg/dL':
#             row['test_value_numeric_standard'] = row['test_value_numeric_standard']*1000
#             row['test_unit_standard_name'] = 'g/dL'
#         else: 
#             row['test_unit_standard_name'] = 'invalid_unit'
#     if row['test_name_standard_name'] in ['Kappa light chains [Mass/volume] in Serum or Plasma',
#                                           'Kappa light chains.free [Mass/volume] in Serum',
#                                           'Kappa light chains.free/Lambda light chains.free [Mass Ratio] in Serum',
#                                           'Lambda light chains [Mass/volume] in Serum or Plasma',
#                                           'Lambda light chains.free [Mass/volume] in Serum or Plasma']:
#         if row['test_unit_standard_name'] == 'mg/dL':
#             row['test_value_numeric_standard'] = row['test_value_numeric_standard'] * 10
#             row['test_unit_standard_name'] = 'mg/L'
#         else: 
#             row['test_unit_standard_name'] = 'invalid_unit'
#     if row['test_name_standard_name'] in ['Lymphocytes [#/volume] in Blood',
#                                           'Lymphocytes [#/volume] in Blood by Automated count',
#                                           'Lymphocytes [#/volume] in Blood by Manual count',
#                                           'Neutrophils [#/volume] in Blood by Automated count',
#                                           'Neutrophils [#/volume] in Blood by Manual count',
#                                           'Platelets [#/volume] in Blood',
#                                           'Platelets [#/volume] in Blood by Automated count',
#                                           'Protein.monoclonal band 1 [Mass/volume] in Serum or Plasma by Electrophoresis',
#                                           'Protein.monoclonal/Protein.total in Serum or Plasma by Electrophoresis',
#                                           'Segmented neutrophils [#/volume] in Blood']:
#         if row['test_unit_standard_name'] in ('10*9/L', '10*9 cells/L'):
#             row['test_unit_standard_name'] = 'x10(3)/mcL'
#         elif row['test_unit_standard_name'] in ('/mL', 'cells/uL'):
#             row['test_value_numeric_standard'] = row['test_value_numeric_standard']/1000
#             row['test_unit_standard_name'] = 'x10(3)/mcL'
#         else: 
#             row['test_unit_standard_name'] = 'invalid_unit'
#     return row

# def calculate_statistics(df, lab, lot_table):
#     final_df = lot_table[['chai_patient_id', 'line_start']].drop_duplicates()
    
#     # Group the data by patient ID and calculate the desired statistics
#     for index in [0, 1, 2, -1]:
#         if index in [-1]:
#             grouped_df = df[df["rand_wind"] == 1].groupby(['chai_patient_id', 'line_start'])['test_value_numeric_standard']
            
#             temp_df = copy.deepcopy(df)
#             temp_df['day_diff'] = temp_df['day_diff'] - 10000
#             grouped_df_slope = temp_df[temp_df["rand_wind"] == 1].groupby(['chai_patient_id', 'line_start'])[
#                 'test_value_numeric_standard'].apply(list).reset_index()
#             grouped_df_slope = grouped_df_slope.merge(
#                 temp_df[temp_df["rand_wind"] == 1].groupby(['chai_patient_id', 'line_start'])['day_diff'].apply(list).reset_index(),
#                 on=['chai_patient_id', 'line_start'], how='inner')

#         else:
#             grouped_df = df[df["wind_num"] == index].groupby(['chai_patient_id', 'line_start'])['test_value_numeric_standard']
            
#             temp_df = copy.deepcopy(df)
#             temp_df['day_diff'] = temp_df['day_diff'] - 10000
#             grouped_df_slope = temp_df[temp_df["wind_num"] == index].groupby(['chai_patient_id', 'line_start'])[
#                 'test_value_numeric_standard'].apply(list).reset_index()
#             grouped_df_slope = grouped_df_slope.merge(
#                 temp_df[temp_df["wind_num"] == index].groupby(['chai_patient_id', 'line_start'])['day_diff'].apply(list).reset_index(),
#                 on=['chai_patient_id', 'line_start'], how='inner')
#         lab_param = lab + '_' + str(index)
#         # Calculate the statistics for each patient
#         statistics_df = pd.DataFrame()
#         statistics_df[f'Mean_{lab_param}'] = grouped_df.mean()
#         statistics_df[f'Median_{lab_param}'] = grouped_df.median()
#         statistics_df[f'SD_{lab_param}'] = grouped_df.std()
#         statistics_df[f'Minimum_{lab_param}'] = grouped_df.min()
#         statistics_df[f'Maximum_{lab_param}'] = grouped_df.max()
#         statistics_df[f'Skewness_{lab_param}'] = grouped_df.apply(lambda x: x.skew())
#         statistics_df[f'Kurtosis_{lab_param}'] = grouped_df.apply(lambda x: x.kurtosis())
#         statistics_df[f'25th_Percentile_{lab_param}'] = grouped_df.apply(lambda x: np.percentile(x, 25))
#         statistics_df[f'75th_Percentile_{lab_param}'] = grouped_df.apply(lambda x: np.percentile(x, 75))
#         statistics_df[f'Range_{lab_param}'] = grouped_df.apply(lambda x: x.max() - x.min())
#         if index in [0, -1]:
#             if grouped_df_slope.shape[0] > 0:
#                 statistics_df[f'slope_{lab_param}'] = grouped_df_slope.apply(
#                     lambda x: np.polyfit(x['day_diff'], x['test_value_numeric_standard'], 1)[0], axis=1)
#         if statistics_df.shape[0] > 0:
#             final_df = final_df.merge(statistics_df, on=["chai_patient_id", 'line_start'], how="left")

#     return final_df
def calculate_statistics(df, lab, cond_table):
    final_df = cond_table[['chai_patient_id', 'line_start']].drop_duplicates()
    df = df.sort_values(by = ['chai_patient_id', 'test_date'])
    df['day_diff'] = df['day_diff'] - 10000
    # Group the data by patient ID and calculate the desired statistics
    # print(lab)
    for rand_wind in list(['0_45','45_180']):
    # for rand_wind in list([0,1]):    
        # print(rand_wind)
        df_temp = df[df["rand_wind"] == rand_wind].sort_values(by = ['chai_patient_id', 'test_date'])
        if df_temp.shape[0]>0:
            df_temp['value_diff'] = df_temp.groupby(['chai_patient_id', 'line_start'])['test_value_numeric_standard'].diff()
#             print(df_temp.head())
            df_temp['time_diff'] = df_temp.groupby(['chai_patient_id', 'line_start'])['test_date'].diff().dt.days
            df_temp.loc[df_temp['time_diff']==0] = np.nan
            df_temp['delta'] = df_temp['value_diff'] / df_temp['time_diff']

    #         df[value + "_delta_max_" +key] = df_temp.groupby(['chai_patient_id', 'line_start'])['delta'].transform('max')

            grouped_df = df_temp.groupby(['chai_patient_id', 'line_start'])['test_value_numeric_standard']
            grouped_df_delta = df_temp.groupby(['chai_patient_id', 'line_start'])['delta']

            grouped_df_slope = df_temp.groupby(['chai_patient_id', 'line_start'])[
                'test_value_numeric_standard'].apply(list).reset_index()
            grouped_df_slope = grouped_df_slope.merge(
                df_temp.groupby(['chai_patient_id', 'line_start'])['day_diff'].apply(list).reset_index(),
                on=['chai_patient_id', 'line_start'], how='inner')


            lab_param = lab + '_' + rand_wind
            # Calculate the statistics for each patient
            statistics_df = pd.DataFrame()


            statistics_df[f'Delta_mean_{lab_param}'] = grouped_df_delta.mean()
            statistics_df[f'Delta_min_{lab_param}'] = grouped_df_delta.min()
            statistics_df[f'Delta_max_{lab_param}'] = grouped_df_delta.max()

            statistics_df[f'Mean_{lab_param}'] = grouped_df.mean()
            statistics_df[f'Median_{lab_param}'] = grouped_df.median()
            statistics_df[f'SD_{lab_param}'] = grouped_df.std()
            statistics_df[f'Minimum_{lab_param}'] = grouped_df.min()
            statistics_df[f'Maximum_{lab_param}'] = grouped_df.max()
            statistics_df[f'Skewness_{lab_param}'] = grouped_df.apply(lambda x: x.skew())
            statistics_df[f'Kurtosis_{lab_param}'] = grouped_df.apply(lambda x: x.kurtosis())
            statistics_df[f'25th_Percentile_{lab_param}'] = grouped_df.apply(lambda x: np.percentile(x, 25))
            statistics_df[f'75th_Percentile_{lab_param}'] = grouped_df.apply(lambda x: np.percentile(x, 75))
            statistics_df[f'Range_{lab_param}'] = grouped_df.apply(lambda x: x.max() - x.min())

            if grouped_df_slope.shape[0] > 0:
                statistics_df[f'slope_{lab_param}'] = grouped_df_slope.apply(
                    lambda x: np.polyfit(x['day_diff'], x['test_value_numeric_standard'], 1)[0], axis=1)
            if statistics_df.shape[0] > 0:
                final_df = final_df.merge(statistics_df, on=["chai_patient_id", 'line_start'], how="left")

    return final_df