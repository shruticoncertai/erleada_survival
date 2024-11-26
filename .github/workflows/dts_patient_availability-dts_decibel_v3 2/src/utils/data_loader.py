import s3fs
import polars as pl
import pyarrow.dataset as ds

import pandas as pd
import utils.table_details as tables
from configs.configs import s3_schema_bucket
#S3_PARQUET_BASE_LOCATION = f"{s3_schema_bucket}/c3_cdm_site360_202112_omop_excelra_v2_qcca"
#S3_PARQUET_CONCEPT_LOCATION = "dev-eureka-rwe-spark/omop_vocabulary/concept.parquet"
SCHEMA = "omop"

def get_concept_lazy_df():
    """
    Return the concept lazy dataframe object,
    """
    s3 = s3fs.S3FileSystem()
    coreFs = s3fs.core.S3FileSystem()
    
    concept_partition_path = s3_schema_bucket + "/omop_vocabulary/concept.parquet/*.parquet"
    concept_partions = coreFs.glob(path=concept_partition_path)
    concept_lazy_df = pl.scan_pyarrow_dataset( ds.dataset(concept_partions, filesystem=s3) )
    
    return concept_lazy_df

def get_table_df(standard_table_name: str,  schema: str="omop", join_concept: bool=True, parquet_base_path = None):
    
    """
    Returns Lazy Dataframe Object given the standard table name and schema name. 
    An optional attribute whether to join with concept table or not
    
    Parameters:
    ----------
    standard_table_name: string: table name to load the parquet file
    schema: string: schema name to get the actual table name matching the parquet filename.
                    currently "omop" is supported
    join_concept: boolena: Whether to join the table with the concept table or not.
    
    
    Returns:
    --------
    lazy_df: Polars.lazyDataframe object
    """

    schema_table_name = tables.get_table(schema, standard_table_name)
    columns = tables.column_names[schema][standard_table_name].values() 
    
    s3 = s3fs.S3FileSystem()
    coreFs = s3fs.core.S3FileSystem()
    
    if schema == 'omop': partition_path = parquet_base_path+"/"+schema_table_name+".parquet"+"/*.parquet"
    else: partition_path = parquet_base_path+"/"+schema_table_name+".parquet"
    
    partition_list = coreFs.glob(path=partition_path)
    dataset = ds.dataset(partition_list, filesystem=s3, format="parquet")
    
    lazy_df = pl.scan_pyarrow_dataset(dataset)

    if (standard_table_name == 'patient_test' and 'unit_concept_id' in columns):
        concept_lazy_df = get_concept_lazy_df()
        concept_lazy_df = concept_lazy_df.with_columns(pl.col("concept_id").cast(pl.Int64))
        concept_lazy_df=concept_lazy_df.select(['concept_id', 'concept_name'])
        concept_lazy_df=concept_lazy_df.rename({"concept_name": "measurement_unit_source_name"})
        lazy_df = lazy_df.join(concept_lazy_df, left_on = 'unit_concept_id',right_on = 'concept_id', how="left")
        
    if ((join_concept) and ("concept_id" in tables.column_names[schema][standard_table_name])):
        concept_lazy_df = get_concept_lazy_df()
        concept_lazy_df = concept_lazy_df.with_columns(pl.col("concept_id").cast(pl.Int64))
        lazy_df = lazy_df.join(concept_lazy_df, left_on = tables.column_names[schema][standard_table_name]['concept_id'],right_on = 'concept_id')
    else:
        columns = [c for c in columns if c in lazy_df.columns]

    return lazy_df.select(columns)
