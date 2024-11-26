import os
import sys
import great_expectations as ge
import pandas as pd
import numpy as np
from datetime import datetime
import json
import xlsxwriter
sys.path.insert(0, os.getcwd())
# from great_expectations_code import ROOT_DIR
from PA_GE_test.rules import *
# from great_expectations_code.rule_config_nlp import *
# from PA_GE_test.rule_config import *
# from data_transfer_utility.application import DataTransferUtility
# from clinical_artifacts.app import ClinicalArtifactsManagement
import argparse
import time
import traceback
import psycopg2
from unit_converter.convert import *
import yaml

# excel_file =  'Inbound_surveillance_results' + '.xlsx'
# excel_file_model = ROOT_DIR + '/output_files/' + 'model_counts' + '.xlsx'
KEYS_LABS_FILENAME = "unit_converter/keys_labs.yaml"
UNITS_FILENAME = "unit_converter/units.yaml"
with open('./PA_GE_test/rule_config.yaml','r') as file:
    GE_config = yaml.safe_load(file)
    
# GE_config = {c['key']:c for c in GE_config}

with open(KEYS_LABS_FILENAME,'r') as file:
    lab_kb = yaml.safe_load(file)
    
lab_kb = {c['key']:c for c in lab_kb}

with open(UNITS_FILENAME,'r') as file:
    unite_conv_map = yaml.safe_load(file)
    
unite_conv_map = {c['base']:c for c in unite_conv_map}

lab_dict = {'m_protein_in_serum':['33358-3','51435-6','35559-4','94400-9','33647-9','50796-2','56766-9','44932-2','50792-1'],
            'm_protein_in_urine':['42482-0','40661-1','35560-2'],
            'ca':['17861-6','49765-1'],
            'serum_free_light':['36916-5','33944-0','11051-0','11050-2'],
            'hemoglobin_in_blood':['718-7','20509-6','30313-1','48725-6'],
            'neutrophils_count':['751-8','26499-4','768-2','30451-9','753-4'],
            'lymphocytes_count':['26474-7','732-8','731-0'],
            'platelets':['777-3','26515-7','53800-9','49497-1','778-1'],
            'na':['2951-2','2955-3'],
            'mg':['21377-7','19123-9'],
            'cl':['2075-0'],
            'phos' : ['2777-1'],
            'hr' : ['8867-4'],
            'dbp' : ['8462-4'],
            'ecog' : ['89262-0'],
            'k' : ['2823-3','2828-2']
           }
lab_list = list(lab_dict.keys())
lab_list = [ele for ele in lab_list if ele not in ['hr','dbp','ecog']]

def get_lab_group(code):
    for lab in lab_dict.keys():
        if code in lab_dict[lab]:
            return lab

def value_counts_check(df, tab, worksheet, cell_format1,schema,row):
        """ Function used to get column level  value counts.
        :param tab : On table we are chekcing the counts .
        :param worksheet: worksheet object
        :param cell_format1 : Excel sheet cell formatting details.
        """
        val = GE_config['value_counts_config'][tab]
        row = row+2
        col = 0
        if tab=='patient_test_raw':      
            df['lab_group'] = df['test_name_standard_code'].apply(get_lab_group)
            val+= ['lab_group']

        for co in val:
            cnt_dict = get_column_value_counts(df, tab, co)
            worksheet.write(row, col, co, cell_format1)
            worksheet.write(row, col + 1, f'count (schema : {schema})', cell_format1)
            for i, j in cnt_dict.items():
                worksheet.write(row + 1, col, i)
                worksheet.write(row + 1, col + 1, j)
                row = row + 1
            row = row + 2


def rules_check(df,table,schema,workbook):
        """
            Execution of rules mentioned in rules_config file .
            We will be rotating through rules_table and execute rules for each table .
        """
        l1=[]
#         df = ge.dataset.PandasDataset(df,columns= df.columns)
        df = ge.dataset.PandasDataset(df.to_pandas())
        
        print('Processing for table : ', table)
        column_rule = GE_config['rules_table'][table]
        for column, rule_list in column_rule.items():
            for rule in rule_list:
                if isinstance(rule, dict):
                    rule_name = list(rule.keys())[0]
                    if rule_name=='limited_values_rule':
                        if (table=='patient_test_raw') or (table=='patient_test_std'):
                            for lab in lab_list:
                                lab_df = df[df['test_name_standard_code'].isin(lab_dict[lab])]
                                if table=='patient_test_raw':
                                    limited_values = set(list(unite_conv_map[lab_kb[lab]['attributes']['units']]['convert'].keys())+[lab_kb[lab]['attributes']['units']])
                                elif table=='patient_test_std':
                                    limited_values = [lab_kb[lab]['attributes']['units']]
                                print(f'processing for table : {table} lab {lab} and column : {column} ')
                                result, par_unexp_val, unexp_cnt, unexp_per = eval(rule_name + f"(lab_df,table,column,limited_values )")
                                l1.append([f'rule : {rule_name} for lab {lab} with limited values as {limited_values}', table, column, result,
                                           par_unexp_val, unexp_cnt, unexp_per])
                        else: 
                            limited_values = rule.get(rule_name)
                            print(f'processing for table : {table} and column : {column} ')
                            result, par_unexp_val, unexp_cnt, unexp_per = eval(rule_name + f"(df,table,column,limited_values )")
                            l1.append([f'rule : {rule_name} with limited values as {limited_values}', table, column, result,
                                       par_unexp_val, unexp_cnt, unexp_per])
                    elif rule_name=='column_datatype_check':
                        data_type = rule.get(rule_name)
                        print(f'processing for table : {table} and column : {column} ')
                        result, par_unexp_val, unexp_cnt, unexp_per = eval(rule_name + f"(df,table,column,data_type)")
                        l1.append([f'rule : {rule_name} with data type {data_type}', table, column, result,
                                    par_unexp_val, unexp_cnt, unexp_per])
                    elif rule_name=='value_in_range_test_value':
                        for lab in lab_list:
                            lab_df = df[df['test_name_standard_code'].isin(lab_dict[lab])]
                            min_value = lab_kb[lab]['attributes']['valid']['min']
                            max_value = lab_kb[lab]['attributes']['valid']['max']
                            print(f'processing for table : {table} and column : {column} ')
                            result, par_unexp_val, unexp_cnt, unexp_per = eval('value_in_range' + f"(lab_df,table,column,min_value,max_value,False)")
                            l1.append([f'rule : {rule_name} with lab {lab} and value range {(min_value,max_value)}', table, column, result,
                                    par_unexp_val, unexp_cnt, unexp_per])
                    elif rule_name=='value_in_range':
                        (min_value,max_value,parse_strings_as_datetimes) = rule.get(rule_name)
                        print(f'processing for table : {table} and column : {column} ')
                        result, par_unexp_val, unexp_cnt, unexp_per = eval(rule_name + f"(df,table,column,min_value,max_value,parse_strings_as_datetimes)")
                        l1.append([f'rule : {rule_name} with value range {(min_value,max_value)}', table, column, result,
                                par_unexp_val, unexp_cnt, unexp_per])

                else:
                    print(f'processing for table : {table} and column : {column} ')
                    result, par_unexp_val, unexp_cnt, unexp_per = eval(rule + f"(df, table, column)")
                    l1.append([f'rule : {rule} ', table, column, result, par_unexp_val, unexp_cnt, unexp_per])

        print('Finished processing for table :',table)

        df_pd = pd.DataFrame(l1, columns=['Rule_name', 'table', 'column', 'Status', 'par_unexp_val', 'unexp_cnt',
                                          'unexp_percent'])
        df_pd = df_pd.fillna('')
        df_pd['par_unexp_val'] = df_pd['par_unexp_val'].astype(str)
        df_pd = df_pd.replace('[]', '')
        #workbook = xlsxwriter.Workbook(excel_file, {'nan_inf_to_errors': True})
        cell_format1 = workbook.add_format({'bold': True, 'bg_color': '#002060', 'font_color': '#ffffff'})
        cell_format_green = workbook.add_format({'bold': True, 'bg_color': '#008000', 'font_color': '#ffffff'})
        cell_format_red = workbook.add_format({'bold': True, 'bg_color': '#FF0000', 'font_color': '#ffffff'})
        cell_format_summary = workbook.add_format({'bold': True, 'bg_color': '#808080', 'font_color': '#ffffff'})
        result_table_ = df_pd['table'].unique()

        for table in result_table_:
            print(f'Adding results for table : {table}')
            df_new = df_pd.loc[df_pd['table'] == table]
            worksheet = workbook.add_worksheet(table)
            row = 0
            col = 0
            total_columns = list(df_new.columns)
            columns_length = len(total_columns)
            rows_length = len(df_new.index)

            ##Adding columns to the excel sheet.
            for columns in total_columns:
                worksheet.write(row, col, columns, cell_format1)
                col = col + 1
            ##Adding rows to excel sheet .
            row = 1
            col = 0
            for i in range(0, rows_length):
                for j in range(0, columns_length):
                    if str(df_new[total_columns[j]].iloc[i]) == "True":
                        worksheet.write(row, col, str(df_new[total_columns[j]].iloc[i]), cell_format_green)
                    elif str(df_new[total_columns[j]].iloc[i]) == "False":
                        worksheet.write(row, col, str(df_new[total_columns[j]].iloc[i]), cell_format_red)
                    else:
                        worksheet.write(row, col, str(df_new[total_columns[j]].iloc[i]))
                    col = col + 1
                row = row + 1
                col = 0
            worksheet.write(row+1, col, "Schema :", cell_format_summary)
            worksheet.write(row+1, col + 1, f"{schema}", cell_format_summary)

            if table in GE_config['value_counts_config'].keys():
                print(f'Adding column level counts for table {table}')
                value_counts_check(df,table, worksheet, cell_format1,schema,row)

        #workbook.close()
