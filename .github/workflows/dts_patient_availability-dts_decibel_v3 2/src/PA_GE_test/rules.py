##Rules file
################################################New Rules ################################

def one_value_one_record_rule(df,table_name,column_name):
    res_unique = df.expect_column_values_to_be_unique(column_name)
    if res_unique['success'] == True:
        print(f'one_value_one_record_rule passed for column {column_name } in table : {table_name} ')
        return True,None,res_unique['result']['unexpected_count'],res_unique['result']['unexpected_percent']
    else:
        print(f'one_value_one_record_rule failed for column {column_name } in table : {table_name} ')
        return False,None,res_unique['result']['unexpected_count'],res_unique['result']['unexpected_percent']

def multi_column_unique_record_rule(df,table_name,column_list):
    res_unique = df.expect_compound_columns_to_be_unique(column_list=column_list)
    if res_unique['success'] == True:
        print(f'multi_column_unique_record_rule passed for tabe {table_name}')
        return True,None,res_unique['result']['unexpected_count'],res_unique['result']['unexpected_percent']
    else:
        print(f'multi_column_unique_record_rule failed for table {table_name }')
        return False,None,res_unique['result']['unexpected_count'],res_unique['result']['unexpected_percent']


def not_null_rule(df,table_name,column_name):
    res_not_null=df.expect_column_values_to_not_be_null(column_name)
    if res_not_null['success'] == True:
        print(f'not_null rule passed for column {column_name } in table : {table_name} ')
        return True,res_not_null['result']['partial_unexpected_list'],res_not_null['result']['unexpected_count'],res_not_null['result']['unexpected_percent']
    else:
        print(f'not_null rule failed for column {column_name} in table : {table_name} ')
        return False,res_not_null['result']['partial_unexpected_list'],res_not_null['result']['unexpected_count'],res_not_null['result']['unexpected_percent']

def limited_values_rule(df,table_name,column_name,values):
    res_lim_val=df.expect_column_values_to_be_in_set(column_name,value_set=values)
    if res_lim_val['success'] == True:
        print(f'limited values rule passed for column {column_name}  with values {values} in table : {table_name} ')
        return True,set(res_lim_val['result']['partial_unexpected_list']),res_lim_val['result']['unexpected_count'],res_lim_val['result']['unexpected_percent']
    else:
        print(f'limited values rule failed for column {column_name}  with values {values} in table : {table_name} ')
        return False,set(res_lim_val['result']['partial_unexpected_list']),res_lim_val['result']['unexpected_count'],res_lim_val['result']['unexpected_percent']

def column_datatype_check(df,table_name,column_name,datatype):
    df_col_datatype=df.expect_column_values_to_be_of_type(column_name,datatype)
    if df_col_datatype['success'] == True:
        print(f'Data type validtion rule passed for column {column_name}  with data type {datatype} in table : {table_name} ')
        return True,None,None,None
    else:
        print(f'Data type validtion rule failed for column {column_name}  with data type {datatype} in table : {table_name} ')
        return False,None,None,None
    
def value_in_range(df,table_name,column_name,min_value,max_value,parse_strings_as_datetimes):
    df_col_range=df.expect_column_values_to_be_between(column_name, min_value=min_value,
                                              max_value=max_value,
                                              parse_strings_as_datetimes=parse_strings_as_datetimes)
    if df_col_range['success'] == True:
        print(f'Value in range rule passed for column {column_name}  with range {(min_value,max_value)} in table : {table_name} ')
        return True,df_col_range['result']['partial_unexpected_list'],df_col_range['result']['unexpected_count'],df_col_range['result']['unexpected_percent']
    else:
        print(f'Value in range rule failed for column {column_name}  with range {(min_value,max_value)} in table : {table_name} ')
        return False,df_col_range['result']['partial_unexpected_list'],df_col_range['result']['unexpected_count'],df_col_range['result']['unexpected_percent']

    
def date_column_rule(df,table_name,column_name):
    df_col_type=df.expect_column_values_to_be_of_type(column_name,'datetime64')
    if df_col_type['success'] == True:
        #Check the date not more than 2025
        df_res=df.expect_column_values_to_be_between(column_name, min_value='1900-01-01',
                                              max_value='2025-01-01',
                                              parse_strings_as_datetimes=True)
        if df_res['success'] == True:
            print(f'date column check rule passed for column {column_name}  with values in table : {table_name} ')
            return True,df_res['result']['partial_unexpected_list'], df_res['result']['unexpected_count'], df_res['result']['unexpected_percent']
        else:
            print(f'limited values rule failed for column {column_name}  with values  in table : {table_name} ')
            return False,df_res['result']['partial_unexpected_list'], df_res['result']['unexpected_count'], df_res['result']['unexpected_percent']
    else:
        print(f'date column check rule failed for column {column_name} with non datetime column in table : {table_name} ')
        return False,None, None,None


def table_columns_check(metadata,df,table):
    expected_coumns=metadata[table]
    result = {}
    result_fail = {}
    for i in expected_coumns:
        res = df.expect_column_to_exist(i)
        result[res['expectation_config']['kwargs']['column']] = res['success']
        if res['success'] is False:
            result_fail[res['expectation_config']['kwargs']['column']] = res['success']
    if bool(result_fail):
        print(f'column rule failed & few columns {result_fail} are missing in table {table}')
        return False,result_fail
    else:
        print(f'columns Rule passed for table : {table}')
        return True,None


def table_columns_diff(df,df_previous,table):
    expected_coumns=list(df_previous.columns)
    new_columns = list(df.columns)
    result = {}
    result_fail = {}
    for i in expected_coumns:
        res = df.expect_column_to_exist(i)
        result[res['expectation_config']['kwargs']['column']] = res['success']
        if res['success'] is False:
            result_fail[res['expectation_config']['kwargs']['column']] = 'Column dropped'
    for i in new_columns:
        res = df_previous.expect_column_to_exist(i)
        result[res['expectation_config']['kwargs']['column']] = res['success']
        if res['success'] is False:
            result_fail[res['expectation_config']['kwargs']['column']] = 'New column Added'

    if bool(result_fail):
        print(f'column diff check rule failed for table {table} ,  few columns {result_fail} ')
        return False, result_fail
    else:
        print(f'columns diff check Rule passed for table : {table}')
        return True,None

def get_column_value_counts(df,table,column_name):
    res=df.get_column_value_counts(column_name,sort='count')
    return res.to_dict()
