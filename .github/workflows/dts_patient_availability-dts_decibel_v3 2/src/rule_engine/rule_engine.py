from configs.configs import *
from configs.variable_constants import *
from utils import utils
import datetime
import numpy  as np
logger = utils.setup_logger("server")

class IndicationRuleEngine:
    def __init__(self, rules):
        self.rules = rules

    # Rules assume the order as the precedence defined
    def apply_rules(self, df, evidence_df):
        result_dfs = None
        evidence_dfs = None
        for rule in self.rules:
            print("RULE: ", rule)
            d,e = rule().apply_rule(df, evidence_df)
            #print("Result: ", d)
            if result_dfs is None: 
                result_dfs = d
                evidence_dfs = e
            else: 
                result_dfs = result_dfs.append(d)
                evidence_dfs = evidence_dfs.append(e)
                
        result_dfs = result_dfs.drop_duplicates(subset=[PERSON_ID], keep='first')
        return result_dfs, evidence_dfs

class Rule:
    def __init__(self):
        pass

    def apply_rule(self,
                   df,
                   evidence_df,
                   **kwargs):
        raise NotImplementedError

class MMObjectiveCriteriaRule(Rule):
    def __init__(self):
        super().__init__()

    def apply_rule(self,
                   df, evidence_df,
                   **kwargs):
        today = datetime.date.today()
        max_pr_date = today - datetime.timedelta(days=MAX_MM_CRITERIA_DAYS_RB)
        #max_pr_date = np.datetime64(max_pr_date)
        pr_data = df.dropna(subset=[LATEST_PROGRESSION_DATE], inplace=False, how='all')
        if not pr_data.empty:
            pr_data = pr_data[pr_data[LATEST_PROGRESSION_DATE].dt.date > max_pr_date][[PERSON_ID, LATEST_PROGRESSION_DATE]]
        default_pr_probability_1 = {RES_30: 0.8, RES_60: 0.8, RES_90: 0.8, RES_120: 0.8, RES_150:0.8, RES_180:0.8, SOURCE: 'RULE:ObjectiveCriteria'}
        logger.info(f"Patients with Criteria based PR date after {max_pr_date} are {pr_data.shape}")
        pr_data = pr_data.reset_index()
        pr_data = pr_data.assign(**default_pr_probability_1)
        
        pr_data = pr_data[[PERSON_ID, RES_90, RES_120, RES_150, RES_180, SOURCE, LATEST_PROGRESSION_DATE ]]
        pr_data = pr_data.rename(columns={LATEST_PROGRESSION_DATE: SOURCE_DATE})
        evidence_data = evidence_df[evidence_df[PERSON_ID].isin(pr_data[PERSON_ID].to_list())]
        evidence_data = evidence_data.loc[evidence_data["measure"].str.startswith('Criteria', na=False)]
        

        return pr_data, evidence_data

class TumorResponseStatusRule(Rule):
    def __init__(self):
        super().__init__()

    def apply_rule(self,
                   df,evidence_df,
                   **kwargs):
        today = datetime.date.today()
        max_pr_date = today - datetime.timedelta(days=MAX_PROGRESSION_DAYS_RB)
        max_pr_date = np.datetime64(max_pr_date)
        
        pr_data = df[df[NLP_PROGRESSION_DATE] > max_pr_date][[PERSON_ID, NLP_PROGRESSION_DATE ]]
        default_pr_probability_1 = {RES_90: 0.8, RES_120: 0.8, RES_150: 0.8, RES_180: 0.8, SOURCE: 'RULE:TP_NLP'}
        logger.info(f"Patients with TP detected after {max_pr_date} are {pr_data.shape}")
        pr_data = pr_data.reset_index()
        pr_data = pr_data.assign(**default_pr_probability_1)
        pr_data = pr_data[[PERSON_ID, RES_90, RES_120, RES_150, RES_180, SOURCE, NLP_PROGRESSION_DATE ]]
        pr_data = pr_data.rename(columns={NLP_PROGRESSION_DATE: SOURCE_DATE})
        
        evidence_data = evidence_df[evidence_df[PERSON_ID].isin(pr_data[PERSON_ID].to_list())]
        evidence_data = evidence_data.loc[evidence_data["measure"] == "Tumor Response Status"]
        
        return pr_data, evidence_data
    
class MetastasisRule(Rule):
    def __init__(self):
        super().__init__()
        
    def apply_rule(self, df, evidence_df,**kwargs):
        today = datetime.date.today()
        max_met_date = today - datetime.timedelta(days=MAX_MET_DAYS_RB)
        max_met_date = np.datetime64(max_met_date)
        
        met_data = df[df[MET_DATE] > max_met_date][[PERSON_ID, MET_DATE ]]
        defatult_probability = {RES_90: 0.8, RES_120: 0.8, RES_150: 0.8, RES_180: 0.8, SOURCE: 'RULE:MET_DATE'}
        logger.info(f"Patients with metastasis confirmed after {max_met_date} are {met_data.shape}")
        met_data = met_data.reset_index()
        met_data = met_data.assign(**defatult_probability)
        met_data = met_data[[PERSON_ID, RES_90, RES_120, RES_150, RES_180, SOURCE, MET_DATE ]]
        met_data = met_data.rename(columns={MET_DATE: SOURCE_DATE})
        
        evidence_data = evidence_df[evidence_df[PERSON_ID].isin(met_data[PERSON_ID].to_list())]
        evidence_data = evidence_data.loc[evidence_data["measure"].str.startswith('Metastatic', na=False)]
        
        return met_data, evidence_data

class DataPrevalanceRule:
    def __init__(self):
        pass

    def apply_rule(self,
                   df, evidence_df,
                   **kwargs):
        raise NotImplementedError