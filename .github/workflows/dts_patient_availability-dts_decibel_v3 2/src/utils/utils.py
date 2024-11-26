import datetime
import logging
import logging.config
import json
import os
import requests
import yaml
import smart_open
import polars as pl

from typing import List
from configs.configs import *


def load_synonym_data(data):
    result = requests.post(SYNONYM_URL, json = data)
    return result.json()

def extend_stage_code_mapper(concept_mapper, configurations=None):
    concept_code_mapper_df = pl.DataFrame(concept_mapper)
    synonym_payload = [ {"concept_code": i["concept_code"], 'vocabulary_id': i['vocabulary_id']}for i in concept_mapper]
    synonym_api_response = load_synonym_data(synonym_payload)
    if synonym_api_response["status"] == "completed":
        synonym_df = pl.DataFrame(synonym_api_response["result"]).explode("synonyms")
        if configurations is None:
            return synonym_df["synonyms"].to_list()
        
        synonym_df.join(concept_code_mapper_df, on=["concept_code", "vocabulary_id"])
        ref_key_mapping = concept_code_mapper_df.join(synonym_df, on=["concept_code", "vocabulary_id"]).select(["ref_key", "synonyms"]).unique()
        extended_configurations = configurations.copy()
        for row in ref_key_mapping.iter_rows(named=True):
            if row["ref_key"] in configurations and extended_configurations is not None:
                extended_configurations[row["synonyms"]] = extended_configurations[row["ref_key"]]
    else:
        print(synonym_payload, synonym_api_response)
        extended_configurations = []
    return extended_configurations

def setup_logger(name, default_level = logging.INFO):
    config_path = "configs/log_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        logger = logging.getLogger(name)
    # if file not found, use basic config to avoid exceptions
    else:
        print("Setting up basic config")
        logging.basicConfig(level=default_level)
        logger = logging.getLogger("root")

    return logger

logger = setup_logger("server")

def update_req_status(status, message, request):
    request_id = request["req_id"]
    s3_path = f"s3://{s3_schema_bucket}/{s3_prefix}/inference_output/{request_id}.json"
    logger.info(f"Updating the request - {request_id} with status - {status} to {s3_path}")
    try:
        status_response = dict()
        status_response["Request"] = request
        if (status == "Inprogress"):
            status_response["Status"] = status
        elif (status == "Error"):
            status_response["Status"] = status
            status_response["Error Message"] = message
        elif (status == "Completed"):
            status_response["Status"] = status
            status_response["Result"] = message
        else:
            status_response["Status"] = status
            status_response["Mesasge"] = message
            
        with smart_open.open(s3_path, "w") as s3_writer:
            json.dump(status_response, s3_writer, default=str)
            
    except Exception as e:
        logger.error("Unable to save the message")
        logger.error(str(e))

def filter_by_daterange(df, date_df, ref_date_column, start_date_column, end_date_column, join_on=None, left_join_on=None, right_join_on=None):
    """
    Method to filter the data based on a specific date_range
    @Args:
        df: Polars.DataFrame: Main Dataframe object to filter the data
        date_df: Polars.DataFrame: Reference Dataframe to get the start and end dates to filter
        ref_date_column: str: column to compare the date range and should be the column of df
        start_date_column: str:  column to compare the start date and should be the column of date_df
        end_date_column: str: column to compare the end date and should be the column of date_df
        join_on: str: column name to join the dataframes that should be in both the data frames
        left_join_on: str: column name to join the dataframes that should be in main data frame
        right_join_on: str: column name to join the dataframes that should be in date data frame
    
    @Returns:
        df: Polars.DataFrame
    """

    main_columns = df.columns
    date_df_columns = date_df.columns

    if ref_date_column not in main_columns:
        raise KeyError(f"Invalid Key Input, ref_date_colum: {ref_date_column}")
    
    if start_date_column not in date_df_columns:
        raise KeyError(f"Invalid Key Input, start_date_colum: {start_date_column}")

    if end_date_column not in date_df_columns:
        raise KeyError(f"Invalid Key Input, start_date_colum: {end_date_column}")
      
    if join_on:
        if left_join_on:
            raise KeyError("join_on input should not be with the combination of left_join_on or right_join_on")
        if right_join_on:
            raise KeyError("join_on input should not be with the combination of left_join_on or right_join_on")
        
        left_join_on = join_on
        right_join_on = join_on
    else:
        if left_join_on is None:
            raise KeyError("join_on or left_join_on and right_join_on has to be provided. No inputs are provided")
        if right_join_on is None:
            raise KeyError("join_on or left_join_on and right_join_on has to be provided. No inputs are provided")
    
    if left_join_on not in main_columns:
        raise KeyError(f"{left_join_on} column in not available in dataframe")
    
    if right_join_on not in date_df_columns:
        raise KeyError(f"{right_join_on} column in not available in dataframe")
    
    if not date_df[right_join_on].shape[0] == date_df[right_join_on].unique().shape[0]:
        logger.warning("date_df has duplicate elements. this disrupt the behaviour")

    merge_df = df.join(date_df,
                       left_on=left_join_on,
                       right_on=right_join_on,
                       how='left')
    merge_df = merge_df.filter(
        (pl.col(ref_date_column) >= pl.col(start_date_column) ) & 
        (pl.col(ref_date_column) <= pl.col(end_date_column) )
    )

    if not (merge_df[left_join_on].unique().shape[0] == df[left_join_on].unique().shape[0]):
        logger.warning(f"unique {left_join_on} records have reduced with this filter")
        print(merge_df[left_join_on].unique().shape[0],  df[left_join_on].unique().shape[0])
        
    return merge_df

def window_tagger(df, ref_date_column, window_list, ref_date=None, window_tag_column_name="window_tag", filter_outside_range=False ):
    """
        Method to tag rows based on the window list from -1 to length of window_list.
            -1: After Refresh Date
            0..n-1: specific to window_list
            n: before the last considered date
        @Args:
            df: Polars.DataFrame: Main DataFrane to tag
            ref_date_column: str: reference date column name for which the windows are calculated
            window_list: List(int): Incremental Numbers which decides window range
            ref_date: datetime.date: Refresh/reference date for tagging
                            Default: None: considers datetime.date.today() as refresh date
            window_tag_column_name: str: resultant columnname to save the tags
                            Default: `window_tag`
            filter_outside_range: Boolean: Option to filter the records with the tag -1 and n
                            Default: False
    """

    main_columns = df.columns
    if not ref_date_column in main_columns:
        raise ValueError(f"Invalid Input ref_date_column,{ref_date_column} not part of {main_columns}")
    
    #validate window_list:
    if not isinstance(window_list, List):
        raise ValueError("Invalid Input, window_list")

    n = len(window_list)
    if n==0:
        raise ValueError("Empty window_list")
    else:
        if not isinstance(window_list[0], int):
            raise ValueError(f"Invalid Input, {window_list[0]}")
        
        if window_list[0]<0:
            raise ValueError(f"Invalid Input, {window_list[0]}. window_list elements should be Positive and incremental")
        
        for i in range(1,n):
            if not isinstance(window_list[i], int):
                raise ValueError(f"Invalid Input window_list[{i}], {window_list[i]}")
        
            if window_list[i]<window_list[i-1]:
                raise ValueError(f"Invalid Input, {window_list[i]}. window_list elements should be Positive and incremental")

    #validate ref_date
    if ref_date is None:
        ref_date = datetime.date.today()

    if not isinstance(ref_date, datetime.date):
        raise ValueError(f"Invalid Data type for field ref_date. Expecting datetime.date. Got {type(ref_date)} ")
    
    #validate window_tag_column_name
    if window_tag_column_name in main_columns:
        logger.warning(f"Window tagging overrides the content of the column{window_tag_column_name}")
    
    df = df.with_columns(
        pl.when(pl.col(ref_date_column) > ref_date)
            .then(pl.lit(-1))
            .otherwise(pl.lit(n))
            .alias(window_tag_column_name)
    )

    for i in range(n):
        df = df.with_columns(
                pl.when(
                    (pl.col(window_tag_column_name) == n) &
                    ((ref_date - pl.col(ref_date_column)).dt.days() <= window_list[i])
                )
                    .then(pl.lit(i))
                    .otherwise(pl.col(window_tag_column_name))
                    .alias(window_tag_column_name)
        )
    
    if filter_outside_range:
        df = df.filter(
            pl.col(window_tag_column_name).is_in(list(range(n)))
        )
    return df

def get_latest_data(df, group_by_columns, sort_by_columns, descending=True):
    """
    Method to get the latest data(or first in sorted data)
    ARGUMENTS:
        df: polars.DataFrame: Main Dataframe to sort and filter
        group_by_columns: str,List(str): column names to groupby, if it is a one column, then string or in case of multiple columns, it is a list object
        sort_by_columns: str: Column name through which the dataframe is sorted
        descending: bool:   Sort in descending or ascending order. By default: descending = True
    """

    main_columns = df.columns
    if type(group_by_columns) == str:
        if group_by_columns not in main_columns:
            raise ValueError("Column Name doesnot exist - "+str(group_by_columns))
        group_by_columns = [group_by_columns]
    elif type(group_by_columns) == list:
        for col in group_by_columns:
            if col not in main_columns:
                raise ValueError("Column Name doesnot exist - "+str(col))
    

    if type(sort_by_columns) == str:
        if sort_by_columns not in main_columns:
            raise ValueError("Column Name doesnot exist - "+str(sort_by_columns))
    
    df = df.sort(   by=(group_by_columns + [sort_by_columns]),
                    descending=descending,
                    nulls_last=True
                ).unique(subset=group_by_columns,keep='first')
    
    return df