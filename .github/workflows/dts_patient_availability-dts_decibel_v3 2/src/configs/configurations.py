from configs.configs import *
from configs.constants import *
from configs.concept_code_mapper import *
from utils import utils

class Configurations():
    def __init__(self):
        pass
    
    def get_cancer_specific_variables(self):
        raise NotImplementedError

class MultipleMyelomaConfigurations(Configurations):
    
    def __init__(self):
        super().__init__()
        self.__cancer_specific_constants = cancer_specific_constants["multiple_myeloma"]
        self.stage_group_mapper = stage_group_mapper
        
    def get_cancer_specific_variables(self):
        return self.__cancer_specific_constants
    
    def update_synonyms(self):
        self.stage_group_mapper = utils.extend_stage_code_mapper(stage_group_mapper_codes, self.stage_group_mapper)
        
        for lab_test in self.__cancer_specific_constants["lab_test_codes"]:
            synonyms = utils.extend_stage_code_mapper( [ {"concept_code": i, "vocabulary_id": "LOINC"} for i in self.__cancer_specific_constants["lab_test_codes"][lab_test]["concept_codes"]  ])
            self.__cancer_specific_constants["lab_test_codes"][lab_test]["concept_names"] = synonyms
    
class MDSConfigurations(Configurations): 
    
    def __init__(self):
        super().__init__()
        self.__cancer_specific_constants = cancer_specific_constants["mds"]
    
    def get_cancer_specific_variables(self):
        return self.__cancer_specific_constants
    
    def update_synonyms(self):
        for lab_test in self.__cancer_specific_constants["lab_test_codes"]:
            synonyms = utils.extend_stage_code_mapper( [ {"concept_code": i, "vocabulary_id": "LOINC"} for i in self.__cancer_specific_constants["lab_test_codes"][lab_test]["concept_codes"]  ])
            self.__cancer_specific_constants["lab_test_codes"][lab_test]["concept_names"] = synonyms


class PanSolidsConfigurations(Configurations):   
    def __init__(self):
        super().__init__()
        self.__cancer_specific_constants = cancer_specific_constants["pan_solids"]
        self.stage_group_mapper = stage_group_mapper
        self.bm_positive_value = [i.lower() for i in bm_positive_value]
        self.bm_negative_value = [i.lower() for i in bm_negative_value]
        self.bm_unknown_value = [i.lower() for i in bm_unknown_value]
        self.ecog_grade_mapper = ecog_grade_mapper
        self.image_group_map = image_group_map
        self.biomarkers = biomarkers
        
        
    def get_cancer_specific_variables(self):
        return self.__cancer_specific_constants 
    
    def update_synonyms(self):
        self.stage_group_mapper = utils.extend_stage_code_mapper(stage_group_mapper_codes, self.stage_group_mapper)
        self.ecog_grade_mapper = utils.extend_stage_code_mapper(ecog_value_codes, self.ecog_grade_mapper)
        self.image_group_map = utils.extend_stage_code_mapper(imaging_codes, self.image_group_map)
        
        self.bm_positive_value += utils.extend_stage_code_mapper(bm_positive_value_codes)
        self.bm_negative_value += utils.extend_stage_code_mapper(bm_negative_value_codes)
        self.bm_unknown_value += utils.extend_stage_code_mapper(bm_unknown_value_codes)
        
        self.__cancer_specific_constants["lab_test_codes"]['ecog']["concept_names"] = utils.extend_stage_code_mapper( [ {"concept_code": i, "vocabulary_id": "LOINC"} for i in self.__cancer_specific_constants["lab_test_codes"]['ecog']["concept_codes"]  ])
        
        for bm in self.biomarkers["concept_codes"]:
            self.biomarkers["synonyms"][bm] = utils.extend_stage_code_mapper(self.biomarkers["concept_codes"][bm])