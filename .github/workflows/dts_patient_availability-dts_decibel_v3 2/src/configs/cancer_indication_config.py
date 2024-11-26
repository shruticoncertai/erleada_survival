from featurizer.tumor_specific_featurizer import *
from rule_engine.rule_engine import *
from configs.variable_constants import *
from configs.configs import *

tumor_indication_config = {
    TumorIndicationConstants.MULTIPLE_MYELOMA : {
        FEATURIZER : MyelomaFeaturizer,
        RULE_ENGINE : [MMObjectiveCriteriaRule, TumorResponseStatusRule],
        BASE_MODEL_NAME : 'multiple_myeloma_base_model',
        GROUPING_ID : 'multiple_myeloma',
        SHORT_CODE: 'mm'
    },
    TumorIndicationConstants.LUNG_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'lung',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.PANCREATIC_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'pancreas',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.MALIGNANT_MELANOMA: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'melanoma',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.COLORECTAL_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'colorectal',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.COLON_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'colorectal',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.PROSTRATE_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'prostate',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.BLADDER_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'bladder',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.BREAST_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'breast',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.HCC: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'hcc',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.RENAL_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'renal',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_base_model"
    },
    TumorIndicationConstants.GASTRIC_ESOPHAGEL_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'gastricesophagus',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_line1_model"
    },
    TumorIndicationConstants.OVARIAN_CANCER: {
        FEATURIZER: SolidTumorFeaturizer,
        RULE_ENGINE: [TumorResponseStatusRule, MetastasisRule ],
        GROUPING_ID: 'pan_solids',
        SHORT_CODE: 'ovarian',
        BASE_MODEL_NAME: "solid_cancer_base_model",
        LINE_1_MODEL_NAME: "solid_cancer_base_model" #Line 1 progression model not available
    },
    TumorIndicationConstants.MDS_CANCER: {
        FEATURIZER : MDSFeaturizer,
        RULE_ENGINE : [TumorResponseStatusRule],
        BASE_MODEL_NAME : 'mds_base_model',
        GROUPING_ID : 'mds',
        SHORT_CODE: 'mds'
    }
}
