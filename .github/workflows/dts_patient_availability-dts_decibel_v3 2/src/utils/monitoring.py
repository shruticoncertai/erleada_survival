import mlflow
import os
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve
from sklearn import metrics

import warnings

from utils.config_helper import getRedshiftConnectionString

warnings.filterwarnings("ignore")

# from study.db_pool import rds_engine, redshift_engine

valid_status = ['Ruled-Out', 'Identified', 'Eligible', 'Watch', 'In-Review', 'Screening/Consented',
                'Enrolled/Randomized', 'Not-Interested', 'Screen-Failed']

# SCHEMA_TABLE and QUERY
schema = os.environ.get('REDSHIFT_SCHEMA', 'eureka_rwe_dev5')
status_tbl = f"{schema}.prescreening_status"
score_tbl = f"{schema}.{os.environ.get('CONCERTAI_PRESCREENING_TABLE_NAME', 'concertai_prescreening_test')}"

status_cols = []

CAI_PRESCREENING_QUERY = "select person_id,prescreen_status,created_at,prescreen_score,progression_score,patient_match_score from {score_tbl} where study_id={study_id}"
PRESCREEN_STATUS_QUERY = "select person_id,prescreen_status,notes,created_at from {status_tbl} where study_id={study_id}"


def prepData(df, w1, w2):
    df['patient_match_score'] = w1 * df.progression_score + w2 * df.prescreen_score
    df['label'] = np.where(df.patient_match_score >= 0.8, 1, 0)
    df['actual'] = np.where(df.final_status == 'In-Review', 1, 0)
    return df


def modelPerf_v2(df):
    auc, f1_max = 0, 0
    if df['actual'].nunique() > 1:
        ## Extract AUC
        auc = roc_auc_score(df['actual'], df['patient_match_score'])

        ## Extract F1/precision/recall
        f1 = []
        for th in range(0, 100, 5):
            f1.append(f1_score(df['actual'], df['label'], average='weighted'))
        ## Extract min threshold for max F1
        f1_max = max(f1)

    return auc, f1_max


def modelPerf(df):
    ## Extract AUC
    auc = roc_auc_score(df['actual'], df['patient_match_score'])

    ## Extract F1/precision/recall
    f1_records = list()
    for th in range(0, 100, 1):
        tmp = df[['actual', 'patient_match_score']]
        tmp.loc[:, 'pred_class'] = 1
        tmp.loc[tmp.patient_match_score >= th / 100, 'pred_class'] = 0
        f1_records.append({'th': th,
                           'precision': precision_score(tmp.actual, tmp.pred_class, average='weighted'),
                           'recall': recall_score(tmp.actual, tmp.pred_class, average='weighted'),
                           'f1': f1_score(tmp.actual, tmp.pred_class, average='weighted')})
    f1 = pd.DataFrame.from_dict(f1_records)

    ## Extract min threshold for max F1
    th_min = min(f1[f1.f1 == max(f1.f1)].th)
    f1_max = max(f1[f1.th == th_min].f1)
    f1_max

    return auc, f1_max


def pat_distribution_chart(monitoring_df, file_name='/tmp/status_distribution.png'):
    """
    Generate Bar chart of patient distribution based on prescreen status
    """
    bar_plot = monitoring_df["final_status"].value_counts().filter(pl.col("final_status").is_in(valid_status))
    ax = bar_plot.to_pandas().plot(
        x="final_status",
        y='counts',
        kind="bar",
        title="Status wise Patient Distribution",
        grid=True,
        legend=False,
        logy=True,
        xlabel="Status",
        ylabel="Number of Patients"
    )
    ax.set_ylabel('Number of Patients')
    ax.set_xlabel('Status')
    ax.bar_label(ax.containers[0])
    ax.figure.savefig(file_name, bbox_inches='tight')

    return file_name


def category_distribution_chart(monitoring_df_pd, filename='/tmp/category_distribution_boxplots.png'):
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('Distributions of Scores by Categories')
    sns.boxplot(x='final_status',
                y='prescreen_score',
                data=monitoring_df_pd,
                ax=ax[0],
                showfliers=False).set(title='Distribution by Category of Eligibility Score')
    sns.boxplot(x='final_status',
                y='progression_score',
                data=monitoring_df_pd,
                ax=ax[1],
                showfliers=False).set(title='Distribution by Category of Availability Score')
    sns.boxplot(x='final_status',
                y='patient_match_score',
                data=monitoring_df_pd,
                ax=ax[2],
                showfliers=False).set(title='Distribution by Category of Combined Score')
    fig.tight_layout(pad=1)
    fig.savefig(filename, bbox_inches='tight')

    return filename


def score_status_distribution(monitoring_df_pd, filename='/tmp/score_distribution.png'):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.tight_layout(pad=5)

    ax[0].scatter(
        monitoring_df_pd[monitoring_df_pd.label == 1].prescreen_score,
        monitoring_df_pd[monitoring_df_pd.label == 1].progression_score,
        label='High Priority'
    )
    ax[0].scatter(
        monitoring_df_pd[monitoring_df_pd.label != 1].prescreen_score,
        monitoring_df_pd[monitoring_df_pd.label != 1].progression_score,
        label='Low Priority'
    )
    ax[0].set_xlabel('Patient Eligibility')
    ax[0].set_ylabel('Patient Availability')
    ax[0].set_title('Combined Score based Priority')
    ax[0].legend()

    for i in valid_status:
        ax[1].scatter(
            monitoring_df_pd[monitoring_df_pd.final_status == i].prescreen_score,
            monitoring_df_pd[monitoring_df_pd.final_status == i].progression_score,
            label=i
        )

    ax[1].set_xlabel('Patient Eligibility')
    ax[1].set_ylabel('Patient Availability')
    ax[1].set_title('Pre-screening Status by PE/PA')
    ax[1].legend()

    fig.savefig(filename)

    return filename


def fpr_precission_recall(monitoring_df_pd, filename='/tmp/F1_score.png'):
    auc = roc_auc_score(monitoring_df_pd['actual'], monitoring_df_pd['patient_match_score'])
    fpr, tpr, _ = metrics.roc_curve(monitoring_df_pd['actual'], monitoring_df_pd['patient_match_score'])

    f1_records = list()
    for th in range(0, 100, 1):
        tmp = monitoring_df_pd[['actual', 'patient_match_score']]
        tmp.loc[:, 'pred_class'] = 1
        tmp.loc[tmp.patient_match_score >= th / 100, 'pred_class'] = 0
        f1_records.append({'th': th,
                           'precision': precision_score(tmp.actual, tmp.pred_class, average='weighted'),
                           'recall': recall_score(tmp.actual, tmp.pred_class, average='weighted'),
                           'f1': f1_score(tmp.actual, tmp.pred_class, average='weighted')})
    f1 = pd.DataFrame.from_dict(f1_records)

    precision, recall, thresholds = precision_recall_curve(monitoring_df_pd['actual'],
                                                           monitoring_df_pd['patient_match_score'])

    fig, ax = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(4)
    fig.set_figwidth(12)
    fig.tight_layout(pad=5)

    ## Plot AUC
    ax[0].plot(fpr, tpr, label="data 1, auc=" + str(round(auc, 2)))
    ax[0].legend(loc=4)
    ax[0].set_xlabel('FPR', fontsize=8)
    ax[0].set_ylabel('TPR', fontsize=8)
    ax[0].set_title('AUC', fontsize=10)

    ## Plot F1/Precision/Recall w.r.t Threshold
    ax[1].plot(f1.th, f1.precision, label='Precision')
    ax[1].plot(f1.th, f1.recall, label='Recall')
    ax[1].plot(f1.th, f1.f1, label='F1')
    ax[1].legend(loc=4)
    ax[1].set_xlabel('Threshold', fontsize=8)
    ax[1].set_ylabel('F1/Precision/Recall', fontsize=8)
    ax[1].set_title('F1/Precision/Recall', fontsize=10)

    ## Plot PRC
    ax[2].plot(recall, precision)
    ax[2].set_xlabel('Recall', fontsize=8)
    ax[2].set_ylabel('Precision', fontsize=8)
    ax[2].set_title('PRC', fontsize=10)
    ax[2].axvline(0.5, linestyle='--')
    ax[2].axhline(0.5, linestyle='--')

    fig.savefig(filename)

    return filename


def generate_charts(study_id, study_name, redshift_engine):
    files = [
        "/tmp/category_distribution_boxplots.png",
        "/tmp/status_distribution.png",
        "/tmp/score_distribution.png",
        "/tmp/F1_score.png",
        "/tmp/weight_variation.png"
    ]
    for f in files:
        if os.path.exists(f):
            os.remove(f)
    cai_prescreening_df = pl.read_database(CAI_PRESCREENING_QUERY.format(study_id=study_id, score_tbl=score_tbl),
                                           getRedshiftConnectionString())
    if cai_prescreening_df.shape[0] == 0:
        print("No Scores Available to process data. Unable to generate_charts for Monitoring")
        return

    prescreen_status_df = pl.read_database(PRESCREEN_STATUS_QUERY.format(study_id=study_id, status_tbl=status_tbl),
                                           getRedshiftConnectionString()) \
        .sort(['person_id', 'created_at']) \
        .groupby(pl.col('person_id')) \
        .agg(pl.col("prescreen_status"), pl.col("notes"), pl.col('created_at')) \
        .with_columns(
        final_status=pl.col("prescreen_status").list.last()
    )
    monitoring_df = cai_prescreening_df.join(prescreen_status_df, on="person_id", how="left") \
        .with_columns(
        pl.when(pl.col('final_status').is_null())
        .then(pl.col("prescreen_status"))
        .otherwise(pl.col('final_status'))
        .alias('final_status')
    ).select(
        ["person_id",
         "prescreen_score",
         "progression_score",
         "patient_match_score",
         "final_status",
         "created_at",
         "notes"
         ]
    ).with_columns(
        label=pl.when(pl.col("patient_match_score") >= 0.5).then(pl.lit(1)).otherwise(pl.lit(0)),
        actual=pl.when(pl.col("final_status").is_in(["In-Review"])).then(pl.lit(1)).otherwise(pl.lit(0))
    )
    monitoring_df_pd = monitoring_df.to_pandas()

    ##Generate the Bar Chart
    pat_distribution_chart(monitoring_df)

    ## Generate Scatter plot
    category_distribution_chart(monitoring_df_pd)

    score_status_distribution(monitoring_df_pd)

    try:
        fpr_precission_recall(monitoring_df_pd)
    except Exception as e:
        print("Precission Recall failed with error" + str(e))

    w1_values = [index / 100 for index in range(0, 105, 5)]
    w2_values = [1 - w1 for w1 in w1_values]
    sim_records = list()

    for w1, w2 in zip(w1_values, w2_values):
        df = monitoring_df_pd.copy()
        df = prepData(df, w1, w2)
        auc, f1 = modelPerf_v2(df)
        sim_records.append({'w1': w1,
                            'w2': w2,
                            'auc': auc,
                            'f1': f1})
    sim = pd.DataFrame.from_dict(sim_records)

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    fig.tight_layout(pad=5)

    ax[0, 0].scatter(sim.w1, sim.f1)  # , label = 'F1 vs. PA Weigth')
    ax[0, 0].set_xlabel('Patient Availability Weight')
    ax[0, 0].set_ylabel('F1 Score')
    ax[0, 0].set_title('Change in Weighted F1 w.r.t Combination Weights')

    ax[0, 1].scatter(sim.w1, sim.auc)  # , label = 'AUC vs. PA Weigth')
    ax[0, 1].set_xlabel('Patient Availability Weight')
    ax[0, 1].set_ylabel('AUC')
    ax[0, 1].set_title('Change in AUC w.r.t Combination Weights')

    ax[1, 0].scatter(sim.w2, sim.f1)  # , label = 'F1 vs. PA Weigth')
    ax[1, 0].set_xlabel('Patient Eligibility Weight')
    ax[1, 0].set_ylabel('F1 Score')
    ax[1, 0].set_title('Change in Weighted F1 w.r.t Combination Weights')

    ax[1, 1].scatter(sim.w2, sim.auc)  # , label = 'AUC vs. PA Weigth')
    ax[1, 1].set_xlabel('Patient Eligibility Weight')
    ax[1, 1].set_ylabel('AUC')
    ax[1, 1].set_title('Change in AUC w.r.t Combination Weights')

    fig.savefig("/tmp/weight_variation.png")

    experiment_name = f"dts_monitoring_{study_name}_{study_id}"
    artifacts = [f for f in files if os.path.exists(f)]
    mlflow_log_artifacts(experiment_name, artifacts)


def mlflow_log_artifacts(experiment_name, files):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    for i in files:
        mlflow.log_artifact(i)

    mlflow.end_run()
