from configs.variable_constants import *
from utils.table_details import get_schema_type
from patient_availability_model import generate_features, generate_features_sc_v2, to_pandas, generate_features_mds

class Featurizer:
    def __init__(self,
                 input_df,
                 schema_name, configurations):
        self.input_df = input_df
        self.schema_name = schema_name
        self.patient_list = self.input_df[PERSON_ID].to_list()
        self.schema_type = get_schema_type(self.schema_name)
        self.configurations = configurations

    def compute_features(self, tumor_type=None):
        raise NotImplementedError

class MyelomaFeaturizer(Featurizer):
    def __init__(self,
                input_df,
                schema_name, configurations):
        super().__init__(input_df, schema_name, configurations)

    def compute_features(self, tumor_type=None):
        dataset, evidence = generate_features(self.patient_list, self.schema_name, is_training=False, schema_type=self.schema_type, configurations=self.configurations)
        dataset_pd = to_pandas(dataset)
        evidence_pd = to_pandas(evidence)
        return dataset_pd, evidence_pd

class SolidTumorFeaturizer(Featurizer):
    def __init__(self,
                input_df,
                schema_name, configurations):
        super().__init__(input_df, schema_name, configurations)

    def compute_features(self, tumor_type=None):
        dataset, evidence = generate_features_sc_v2(tumor_type, self.patient_list, self.schema_name, schema_type=self.schema_type, configurations=self.configurations)
        dataset_pd = to_pandas(dataset)
        evidence_pd = to_pandas(evidence)
        return dataset_pd, evidence_pd

class MDSFeaturizer(Featurizer):
    def __init__(self,
                input_df,
                schema_name, configurations):
        super().__init__(input_df, schema_name, configurations)

    def compute_features(self, tumor_type=None):
        dataset, evidence = generate_features_mds(self.patient_list, self.schema_name, schema_type=self.schema_type, configurations=self.configurations)
        dataset_pd = to_pandas(dataset)
        evidence_pd = to_pandas(evidence)
        return dataset_pd, evidence_pd
