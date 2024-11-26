import pandas as pd
import json
import random
import copy
import numpy as np
from data_transfer_utility.application import DataTransferUtility
import yaml
from convert import *
import polars as pl

with open('keys_labs.yaml','r') as file:
    lab_kb = yaml.safe_load(file)
    
lab_kb = {c['key']:c for c in lab_kb}

with open('units.yaml','r') as file:
    unite_conv_map = yaml.safe_load(file)
    
unite_conv_map = {c['base']:c for c in unite_conv_map}


lab_name_dict = {'6690-2':'wbc_6690-2',
                 '26464-8':'wbc_26464-8',
                 '49498-9':'wbc_49498-9',
                 '33256-9':'wbc_33256-9',
                 '12227-5':'wbc_12227-5',
                 '804-5':'wbc_804-5',
                 
                 '718-7':'hemoglobin_718-7',
                 '30313-1':'hemoglobin_30313-1',
                 '20509-6':'hemoglobin_20509-6',
                 '30350-3':'hemoglobin_30350-3',
                 '30351-1':'hemoglobin_30351-1',
                 '48725-6':'hemoglobin_48725-6',
                 '93846-4':'hemoglobin_93846-4',
                 
                 '26499-4':'neutrophils_26499-4',
                 '751-8':'neutrophils_751-8',
                 '753-4':'neutrophils_753-4',
                 '768-2':'neutrophils_768-2',
                 '30451-9':'neutrophils_30451-9',
                 
                 '777-3':'platelets_777-3',
                 '26515-7':'platelets_26515-7',
                 '49497-1':'platelets_49497-1',
                 '778-1':'platelets_778-1',
                 '26516-5':'platelets_26516-5',
                 '53800-9':'platelets_53800-9',
                 '74464-9':'platelets_74464-9',
                 '13056-7':'platelets_13056-7',
                 
                 '24373-3':'ferritin_24373-3',
                 '2276-4':'ferritin_2276-4',
                 '489004':'ferritin_489004',
                 
                 '708-8':'blastscount_708-8',
                 '30376-8':'blastscount_30376-8',
                 '709-6':'blastspercent_709-6',
                 '26446-5':'blastspercent_26446-5',
                }
# run_labs = list(lab_name_dict.keys())
run_labs = list(set(lab_name_dict.keys())-set(['489004','26516-5','53800-9','74464-9','13056-7','48725-6','93846-4','33256-9','30350-3',
                                               '804-5','49497-1','20509-6','30313-1','778-1','30351-1','49498-9']))


lab_dict = {'wbc':['6690-2','26464-8','49498-9','33256-9','12227-5','804-5'],
            'hemoglobin_in_blood':['718-7','30313-1','20509-6','30350-3','30351-1','48725-6','93846-4'],
            'neutrophils_count':['26499-4','751-8','753-4','768-2','30451-9'],
            'platelets':['777-3','26515-7','49497-1','778-1','26516-5','53800-9','74464-9','13056-7'],
            'ferritin':['24373-3','2276-4','489004'],
            'blasts_percent':['709-6','26446-5'],
            'blasts_count':['708-8','30376-8']
           }


def convert_unit(df):
    
    def get_key(code):
        key = None
        if code in lab_dict['wbc']:
            key = 'wbc'
        elif code in lab_dict['hemoglobin_in_blood']:
            key = 'hemoglobin_in_blood'
        elif code in lab_dict['neutrophils_count']:
            key = 'neutrophils_count'
        elif code in lab_dict['platelets']:
            key = 'platelets'
        elif code in lab_dict['ferritin']:
            key = 'ferritin'
        elif code in lab_dict['blasts_percent']:
            key = 'blasts_percent'
        elif code in lab_dict['blasts_count']:
            key = 'blasts_count'
        return key
  
    def update_unite(row,key,std_unit):    
        if row['key']==key:
            if ((row['test_unit_standard_name']==std_unit) or (row['test_unit_standard_name'] in (unite_conv_map[std_unit]['convert'].keys()))):
                row['test_unit_standard_name']=std_unit
            else:
                row['test_unit_standard_name']=None
        return row
    
    def convert_unite(row):
        row['key'] = get_key(row['test_name_standard_code'])
        if isinstance(row['test_unit_standard_name'],str):
            row['test_unit_standard_name']=row['test_unit_standard_name'].lower()
            
        if row['key'] not in lab_kb.keys():
            raise KeyError(f"Lab test {row['key']} not registered")
            
        std_unit = lab_kb[row['key']]['attributes']['units']
        unit_keys = list(unite_conv_map[std_unit]['convert'].keys()) 
        if row['test_unit_standard_name'] in unit_keys:
            row['test_value_numeric_standard'] = eval(unite_conv_map[std_unit]['convert'][row['test_unit_standard_name']].split('.')[-1])(row['test_value_numeric_standard'])
            row['test_unit_standard_name'] = std_unit
        elif row['test_unit_standard_name']!=std_unit:
            row['test_unit_standard_name'] = None
        return row
    
    df = df.dropna(subset=['test_value_numeric_standard'])
    # df['key'] = df['test_name_standard_code'].apply(get_key)
    # df['test_unit_standard_name'] = df['test_unit_standard_name'].apply(lambda x:x.lower() if isinstance(x,str) else x)
    
    # df.loc[df.apply(lambda x:(x['test_name_standard_code'] in (lab_dict['ecog']+lab_dict['histology'])) and (pd.isna(x['test_unit_standard_name'])),axis=1),'test_unit_standard_name']='valid'

    df = df.apply(lambda x:convert_unite(x),axis=1)
    
#     lab_test = df['key'].dropna().unique()
#     for key in lab_test:
#         if key not in lab_kb.keys():
#             raise KeyError(f'Lab test {key} not registered')
#         std_unit = lab_kb[key]['attributes']['units']
#         unit_keys = list(unite_conv_map[std_unit]['convert'].keys())
#         df['test_value_numeric_standard'] = df.apply(lambda x: eval(unite_conv_map[std_unit]['convert'][x['test_unit_standard_name']].split('.')[-1])(x['test_value_numeric_standard'])\
#                                  if (x['key']==key) & (x['test_unit_standard_name'] in (unit_keys))\
#                                  else x['test_value_numeric_standard'],axis=1)
        
#     df = df.apply(lambda x:update_unite(x,key,std_unit),axis=1)
    df = df.dropna(subset=['test_unit_standard_name'])
        
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
    final_df = cond_table[['chai_patient_id']].drop_duplicates()
    df = df.sort_values(by = ['chai_patient_id', 'test_date'])
    df['day_diff'] = df['day_diff'] - 10000
    # Group the data by patient ID and calculate the desired statistics
    # print(lab)
    
    for rand_wind in list(['0_45','45_180']):
    # for rand_wind in list([0,1]):    
        # print(rand_wind)
        df_temp = df[df["rand_wind"] == rand_wind].sort_values(by = ['chai_patient_id', 'test_date'])
        if df_temp.shape[0]>0:
            df_temp['value_diff'] = df_temp.groupby(['chai_patient_id'])['test_value_numeric_standard'].diff()
            df_temp['time_diff'] = df_temp.groupby(['chai_patient_id'])['test_date'].diff().dt.days
            df_temp.loc[df_temp['time_diff']==0] = np.nan
            df_temp['delta'] = df_temp['value_diff'] / df_temp['time_diff']

    #         df[value + "_delta_max_" +key] = df_temp.groupby(['chai_patient_id'])['delta'].transform('max')

            grouped_df = df_temp.groupby(['chai_patient_id'])['test_value_numeric_standard']
            grouped_df_delta = df_temp.groupby(['chai_patient_id'])['delta']

            grouped_df_slope = df_temp.groupby(['chai_patient_id'])[
                'test_value_numeric_standard'].apply(list).reset_index()
            grouped_df_slope = grouped_df_slope.merge(
                df_temp.groupby(['chai_patient_id'])['day_diff'].apply(list).reset_index(),
                on=['chai_patient_id'], how='inner')


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
                final_df = final_df.merge(statistics_df, on=["chai_patient_id"], how="left")

    return final_df
    



def calculate_statistics_pl(df, lab,schema_type=None, indices=[0,1,2]):
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
    df = pl.from_pandas(df)
    final_df = pl.DataFrame({'chai_patient_id': list(set(df['chai_patient_id']))})
    
    # Group the data by patient ID and calculate the desired statistics
    for index in indices:
        df_window = df.filter(df['rand_wind']==index)

        df_window = df_window.groupby(['chai_patient_id']).agg(
                                        pl.col('test_value_numeric_standard').diff().alias('delta_test_value_numeric'),
                                        pl.col('test_date').diff().dt.days().alias('delta_test_date'),
                                        pl.col('random_date'),
                                        pl.col('rand_wind'),
                                        pl.col('test_value_numeric_standard'),
                                        pl.col('day_diff')
                                ).explode(columns=['delta_test_value_numeric', 
                                                   'delta_test_date',
                                                   'random_date',
                                                   'rand_wind',
                                                   'test_value_numeric_standard','day_diff']
                                ).with_columns(
                                        delta = (pl.col('delta_test_value_numeric')/(pl.col('delta_test_date')+0.001))
                                )
        
        grouped_df =  df_window.groupby('chai_patient_id').agg(pl.col('test_value_numeric_standard'))
        grouped_df_delta = df_window.groupby('chai_patient_id').agg(pl.col('delta'))
        
        df_window = df_window.with_columns(pl.col('day_diff')-10000)
        grouped_df_slope = df_window.sort(by=['day_diff'],descending=False).groupby('chai_patient_id').agg(pl.col('test_value_numeric_standard'))
        
        if grouped_df_slope.shape[0]>0:
            grouped_df_slope = grouped_df_slope.filter(pl.col('test_value_numeric_standard').list.lengths() > 1)
        if grouped_df_slope.shape[0]>0:
            grouped_df_slope = grouped_df_slope.join(
                                        df_window.sort(by=['day_diff'],descending=False)\
                                                .groupby('chai_patient_id').agg(pl.col('day_diff')),
                                        on='chai_patient_id',
                                        how='inner')
            
        lab_param = lab + '_' + index
        if grouped_df.shape[0]>0:
            # Calculate the statistics for each patient
            statistics_df = grouped_df.with_columns(
                    # pl.col('test_value_numeric_standard').list.mean().alias(f'Mean_{lab_param}'),
                    # pl.col('test_value_numeric_standard').list.eval(pl.element().median()).list.first().alias(f'Median_{lab_param}'),
                    pl.col('test_value_numeric_standard').list.min().alias(f'Minimum_{lab_param}'),
                    pl.col('test_value_numeric_standard').list.max().alias(f'Maximum_{lab_param}'),
                    # pl.col("test_value_numeric_standard").list.eval(pl.element().quantile(0.25, 'linear')).list.first().alias(f'25th_Percentile_{lab_param}'),
                    # pl.col("test_value_numeric_standard").list.eval(pl.element().quantile(0.75, 'linear')).list.first().alias(f'75th_Percentile_{lab_param}'),
            #         (pl.col('test_value_numeric_standard').list.min() - pl.col('test_value_numeric_standard').list.max()).alias(f'Range_{lab_param}')                                                 
            )
            grouped_df_delta = grouped_df_delta.with_columns(
                    pl.col('delta').list.mean().alias(f'Mean_delta_{lab_param}')
                    # pl.col('delta').list.min().alias(f'Minimum_delta_{lab_param}'),
                    # pl.col('delta').list.max().alias(f'Maximum_delta_{lab_param}')
            )
            
            statistics_df = statistics_df.join(grouped_df_delta,on='chai_patient_id',how='left')
            statistics_df = statistics_df.drop(['test_value_numeric_standard','delta'])
           
            grouped_df = grouped_df.filter(pl.col('test_value_numeric_standard').list.lengths() > 1)
            if grouped_df.shape[0]>0:
                # stats_SD_df = grouped_df.with_columns(pl.col('test_value_numeric_standard').apply(lambda x:x.std()).alias(f'SD_{lab_param}'))
                # statistics_df = statistics_df.join(stats_SD_df.select(['chai_patient_id',f'SD_{lab_param}']),on='chai_patient_id',how='left')
              
                grouped_df = grouped_df.filter(pl.col('test_value_numeric_standard').list.lengths()>2)
                if grouped_df.shape[0]>0:
                    stats_skew_df = grouped_df.with_columns(pl.col('test_value_numeric_standard').apply(lambda x:x.skew()).alias(f'Skewness_{lab_param}'))              
                    statistics_df = statistics_df.join(stats_skew_df.select(['chai_patient_id',f'Skewness_{lab_param}']),on='chai_patient_id',how='left')

                    grouped_df = grouped_df.filter(pl.col('test_value_numeric_standard').list.lengths()>3)
                    if grouped_df.shape[0]>0:
                        stats_kurtosis_df = grouped_df.with_columns(pl.col('test_value_numeric_standard').apply(lambda x:x.kurtosis()).alias(f'Kurtosis_{lab_param}')) 
                        statistics_df = statistics_df.join(stats_kurtosis_df.select(['chai_patient_id',f'Kurtosis_{lab_param}']),on='chai_patient_id',how='left')
                    else:
                        statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'))
                else:
                    statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                                               pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'))
            else:
                statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                                           pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'),
                                                           pl.lit(None).cast(pl.Float64).alias(f'SD_{lab_param}'))
            # print("Polyfitting Begins")
            # print("Grouped DF Slope", grouped_df_slope)
            if grouped_df_slope.shape[0]>0:
                
                grouped_df_slope = grouped_df_slope.with_columns(pl.struct(['day_diff','test_value_numeric_standard']).apply(lambda x:np.polyfit(x['day_diff'],x['test_value_numeric_standard'],1)[0]).alias(f'slope_{lab_param}'))          
                statistics_df = statistics_df.join(grouped_df_slope.select(['chai_patient_id',f'slope_{lab_param}']),on = 'chai_patient_id',how='left')
            else:
                statistics_df = statistics_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f'slope_{lab_param}'))
                
            
            final_df = final_df.join(statistics_df, on="chai_patient_id", how="left")
        else:
            final_df = final_df.with_columns(
                                        # pl.lit(None).cast(pl.Float64).alias(f'Mean_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'Median_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Minimum_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Maximum_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'25th_Percentile_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'75th_Percentile_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'Range_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Mean_delta_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'Minimum_delta_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'Maximum_delta_{lab_param}'),
                                        # pl.lit(None).cast(pl.Float64).alias(f'SD_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Skewness_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'Kurtosis_{lab_param}'),
                                        pl.lit(None).cast(pl.Float64).alias(f'slope_{lab_param}')
                                        )
    if len(indices) > 2:
        for stat in ["Mean_", "Minimum_", "Maximum_"]:
            f0 = f"{stat}{lab}_{indices[0]}"
            f1 = f"{stat}{lab}_{indices[1]}"
            f2 = f"{stat}{lab}_{indices[2]}"


            if final_df.filter(pl.col(f0).is_not_null() & pl.col(f1).is_not_null()).shape[0]>0:
                try:
                    final_df = final_df.with_columns(
                    pl.when(pl.col(f0).is_not_null() & pl.col(f1).is_not_null())\
                    .then(
                        (pl.col(f0)- pl.col(f1))/(pl.col(f1)+0.001)
                    )\
                    .otherwise(None)
                    .alias(f"{stat}{lab}_{indices[1]}_to_{indices[0]}_per_change"))
                except Exception as e:
                    traceback.print_exc()

            if f"{stat}{lab}_{indices[1]}_to_{indices[0]}_per_change" not in list(final_df.columns):
                final_df = final_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{stat}{lab}_{indices[1]}_to_{indices[0]}_per_change"))

            if final_df.filter(pl.col(f1).is_not_null() & pl.col(f2).is_not_null()).shape[0]>0:
                try:
                    final_df = final_df.with_columns(
                        pl.when(pl.col(f1).is_not_null() & pl.col(f2).is_not_null())\
                        .then(
                            (pl.col(f1)- pl.col(f2))/(pl.col(f2)+0.001)
                        )\
                        .otherwise(None)
                        .alias(f"{stat}{lab}_{indices[2]}_to_{indices[1]}_per_change"))
                except Exception as e:
                    traceback.print_exc()

            if f"{stat}{lab}_{indices[2]}_to_{indices[1]}_per_change" not in list(final_df.columns):
                final_df = final_df.with_columns(pl.lit(None).cast(pl.Float64).alias(f"{stat}{lab}_{indices[2]}_to_{indices[1]}_per_change"))
        
    final_df = final_df.with_columns(
                    pl.col('chai_patient_id').cast(pl.Utf8)
                )
    final_df = final_df.to_pandas()
    return final_df



