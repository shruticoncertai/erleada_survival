import os


model_tracker_config = {
    'solid_cancer_base_model': {
        "model_name": os.environ.get("DEFAULT_SC_MODEL_NAME", "model_04_08_2024"),
        "feature_set": [
               'Delta_mean_ecog_0_45', 'Delta_min_ecog_0_45', 'Delta_max_ecog_0_45',
               'Mean_ecog_0_45', 'Median_ecog_0_45', 'SD_ecog_0_45',
               'Minimum_ecog_0_45', 'Maximum_ecog_0_45', 'Skewness_ecog_0_45',
               'Kurtosis_ecog_0_45', '25th_Percentile_ecog_0_45',
               '75th_Percentile_ecog_0_45', 'Range_ecog_0_45', 'slope_ecog_0_45',
               'Delta_mean_ecog_45_180', 'Delta_min_ecog_45_180',
               'Delta_max_ecog_45_180', 'Mean_ecog_45_180', 'Median_ecog_45_180',
               'SD_ecog_45_180', 'Minimum_ecog_45_180', 'Maximum_ecog_45_180',
               'Skewness_ecog_45_180', 'Kurtosis_ecog_45_180',
               '25th_Percentile_ecog_45_180', '75th_Percentile_ecog_45_180',
               'Range_ecog_45_180', 'slope_ecog_45_180', 'paclitaxel', 'ramucirumab',
               'gemcitabine', 'lorlatinib', 'atezolizumab', 'cisplatin', 'necitumumab',
               'nivolumab', 'adagrasib', 'ipilimumab', 'carboplatin', 'afatinib',
               'mobocertinib', 'osimertinib', 'durvalumab', 'docetaxel',
               'pembrolizumab', 'bevacizumab', 'gefitinib', 'brigatinib',
               'amivantamab', 'pemetrexed', 'alectinib', 'crizotinib', 'sotorasib',
               'ceritinib', 'dacomitinib', 'etoposide', 'erlotinib', 'stage', 'tstage',
               'mstage', 'nstage', 'Complete therapeutic response_wind_0_45',
               'Partial therapeutic response_wind_0_45', 'Stable_wind_0_45',
               'Tumor progression_wind_0_45',
               'Complete therapeutic response_wind_45_180',
               'Partial therapeutic response_wind_45_180', 'Stable_wind_45_180',
               'Tumor progression_wind_45_180', 'biomarker_KRAS', 'biomarker_TP53',
               'biomarker_TMB', 'biomarker_PD-L1', 'biomarker_EGFR', 'biomarker_BRAF',
               'biomarker_PTEN', 'biomarker_TERT', 'biomarker_KIT', 'biomarker_PIK3CA',
               'biomarker_CDH1', 'biomarker_STK11', 'biomarker_ALK', 'CT', 'PET',
               'MRI', 'Radioisotope', 'cohort'
        ],
        "result_set": ["60", "90", "120"]
    },
    'solid_cancer_line1_model': {
        "model_name": os.environ.get("SC_L1_MODEL_NAME", "model_05_10_2024"),
        "feature_set": [
               'Delta_mean_ecog_0_45', 'Delta_min_ecog_0_45', 'Delta_max_ecog_0_45',
               'Mean_ecog_0_45', 'Median_ecog_0_45', 'SD_ecog_0_45',
               'Minimum_ecog_0_45', 'Maximum_ecog_0_45', 'Skewness_ecog_0_45',
               'Kurtosis_ecog_0_45', '25th_Percentile_ecog_0_45',
               '75th_Percentile_ecog_0_45', 'Range_ecog_0_45', 'slope_ecog_0_45',
               'Delta_mean_ecog_45_180', 'Delta_min_ecog_45_180',
               'Delta_max_ecog_45_180', 'Mean_ecog_45_180', 'Median_ecog_45_180',
               'SD_ecog_45_180', 'Minimum_ecog_45_180', 'Maximum_ecog_45_180',
               'Skewness_ecog_45_180', 'Kurtosis_ecog_45_180',
               '25th_Percentile_ecog_45_180', '75th_Percentile_ecog_45_180',
               'Range_ecog_45_180', 'slope_ecog_45_180', 'pembrolizumab',
               'bevacizumab', 'amivantamab', 'ramucirumab', 'etoposide', 'necitumumab',
               'mobocertinib', 'ipilimumab', 'crizotinib', 'osimertinib', 'alectinib',
               'erlotinib', 'afatinib', 'atezolizumab', 'durvalumab', 'sotorasib',
               'adagrasib', 'docetaxel', 'gemcitabine', 'gefitinib', 'paclitaxel',
               'lorlatinib', 'dacomitinib', 'cisplatin', 'ceritinib', 'carboplatin',
               'brigatinib', 'pemetrexed', 'nivolumab', 'stage', 'tstage', 'mstage',
               'nstage', 'Complete therapeutic response_wind_0_45',
               'Partial therapeutic response_wind_0_45', 'Stable_wind_0_45',
               'Tumor progression_wind_0_45',
               'Complete therapeutic response_wind_45_180',
               'Partial therapeutic response_wind_45_180', 'Stable_wind_45_180',
               'Tumor progression_wind_45_180', 'biomarker_KRAS', 'biomarker_TP53',
               'biomarker_EGFR', 'biomarker_ALK', 'biomarker_BRAF', 'biomarker_TMB',
               'biomarker_ROS1', 'biomarker_MET', 'biomarker_ERBB2', 'biomarker_RET',
               'biomarker_PD-L1', 'biomarker_PTEN', 'biomarker_TERT', 'biomarker_KIT',
               'biomarker_PIK3CA', 'biomarker_CDH1', 'biomarker_STK11',
               'biomarker_NTRK1', 'biomarker_NTRK2', 'biomarker_NTRK3',
               'biomarker_FGFR3', 'biomarker_FGFR4', 'CT', 'PET', 'MRI',
               'Radioisotope', 'cohort'
        ],
        "result_set": ["60", "90", "120"]
    },
    'multiple_myeloma_base_model': {
        "model_name": os.environ.get("DEFAULT_MM_MODEL_NAME", "mm_model_nadirrs_v3"),
        "feature_set": [
            'm_protein_in_serum', 
            'abs_change_from_nadir_m_protein_in_serum', 
            'perc_change_from_nadir_m_protein_in_serum', 
            'm_protein_in_urine', 
            'abs_change_from_nadir_m_protein_in_urine', 
            'perc_change_from_nadir_m_protein_in_urine', 
            'serum_free_light_kappa', 
            'abs_change_from_nadir_serum_free_light_kappa', 
            'perc_change_from_nadir_serum_free_light_kappa', 
            'serum_free_light_lambda', 
            'abs_change_from_nadir_serum_free_light_lambda', 
            'perc_change_from_nadir_serum_free_light_lambda',
            'prg_cnt_2_years'
        ],
        "result_set": ["90", "120", "150", "180"]
    },
    'mds_base_model': {
        "model_name": os.environ.get("DEFAULT_MDS_MODEL_NAME", 'mds_05_31'),
        "feature_set": ['Minimum_wbc_0_45', 'Maximum_wbc_0_45', 'Mean_delta_wbc_0_45', 'Skewness_wbc_0_45', 'Kurtosis_wbc_0_45', 'slope_wbc_0_45', 'Minimum_wbc_45_180', 'Maximum_wbc_45_180', 'Mean_delta_wbc_45_180', 'Skewness_wbc_45_180', 'Kurtosis_wbc_45_180', 'slope_wbc_45_180', 'Minimum_blastscount_0_45', 'Maximum_blastscount_0_45', 'Mean_delta_blastscount_0_45', 'Skewness_blastscount_0_45', 'Kurtosis_blastscount_0_45', 'slope_blastscount_0_45', 'Minimum_blastscount_45_180', 'Maximum_blastscount_45_180', 'Mean_delta_blastscount_45_180', 'Skewness_blastscount_45_180', 'Kurtosis_blastscount_45_180', 'slope_blastscount_45_180', 'Minimum_ferritin_0_45', 'Maximum_ferritin_0_45', 'Mean_delta_ferritin_0_45', 'Skewness_ferritin_0_45', 'Kurtosis_ferritin_0_45', 'slope_ferritin_0_45', 'Minimum_ferritin_45_180', 'Maximum_ferritin_45_180', 'Mean_delta_ferritin_45_180', 'Skewness_ferritin_45_180', 'Kurtosis_ferritin_45_180', 'slope_ferritin_45_180', 'Minimum_platelets_0_45', 'Maximum_platelets_0_45', 'Mean_delta_platelets_0_45', 'Skewness_platelets_0_45', 'Kurtosis_platelets_0_45', 'slope_platelets_0_45', 'Minimum_platelets_45_180', 'Maximum_platelets_45_180', 'Mean_delta_platelets_45_180', 'Skewness_platelets_45_180', 'Kurtosis_platelets_45_180', 'slope_platelets_45_180', 'Minimum_hemoglobin_0_45', 'Maximum_hemoglobin_0_45', 'Mean_delta_hemoglobin_0_45', 'Skewness_hemoglobin_0_45', 'Kurtosis_hemoglobin_0_45', 'slope_hemoglobin_0_45', 'Minimum_hemoglobin_45_180', 'Maximum_hemoglobin_45_180', 'Mean_delta_hemoglobin_45_180', 'Skewness_hemoglobin_45_180', 'Kurtosis_hemoglobin_45_180', 'slope_hemoglobin_45_180', 'Minimum_neutrophils_0_45', 'Maximum_neutrophils_0_45', 'Mean_delta_neutrophils_0_45', 'Skewness_neutrophils_0_45', 'Kurtosis_neutrophils_0_45', 'slope_neutrophils_0_45', 'Minimum_neutrophils_45_180', 'Maximum_neutrophils_45_180', 'Mean_delta_neutrophils_45_180', 'Skewness_neutrophils_45_180', 'Kurtosis_neutrophils_45_180', 'slope_neutrophils_45_180', 'Minimum_blastspercent_0_45', 'Maximum_blastspercent_0_45', 'Mean_delta_blastspercent_0_45', 'Skewness_blastspercent_0_45', 'Kurtosis_blastspercent_0_45', 'slope_blastspercent_0_45', 'Minimum_blastspercent_45_180', 'Maximum_blastspercent_45_180', 'Mean_delta_blastspercent_45_180', 'Skewness_blastspercent_45_180', 'Kurtosis_blastspercent_45_180', 'slope_blastspercent_45_180'],
        "result_set": ["60", "90", "120", "150"]
    }
}