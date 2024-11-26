class ModelSelection:
    def __init__(self,
                 sep,
                 cohort_json):
        self.sep = sep
        self.cohort_json = cohort_json

    def is_line1_study(self):
        for study_criteria in self.sep['study_package']:
            if study_criteria['flexibility_type'] == 'subjective':
                if 'no prior systemic' in study_criteria['protocol_criteria_narrative']:
                    return True
        return False