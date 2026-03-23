"""
dags/fraud_pipeline_dag.py
--------------------------
DAG principal do pipeline de deteccao de fraudes.

Fluxo:
    ingest_bronze
        └── transform_silver
                └── load_bigquery_bronze
                        └── load_bigquery_silver
                                └── train_model
                                        └── evaluate_model
                                                └── load_bigquery_gold

Projeto GCP : bankfraud-491117
Schedule    : diario as 6h (horario de Brasilia)
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# Garante que os modulos do projeto estao no path
sys.path.insert(0, "/opt/airflow/src")
sys.path.insert(0, "/opt/airflow")

# ── Configuracoes ─────────────────────────────────────────────────────────────

GCP_PROJECT = os.getenv("GCP_PROJECT", "bankfraud-491117")
DATASET     = os.getenv("FRAUD_DATASET", "ulb")   # ulb | ieee | sparkov

DEFAULT_ARGS = {
    "owner":            "data-team",
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
    "email_on_retry":   False,
}


# ── Tarefas Python ────────────────────────────────────────────────────────────

def task_ingest_bronze(**context):
    """Baixa o dataset do Kaggle e salva em data/raw/ (Bronze local)."""
    import subprocess
    dataset_map = {"ulb": "1", "ieee": "2", "sparkov": "3"}
    num = dataset_map.get(DATASET, "1")
    result = subprocess.run(
        ["python", "download_data.py", "--dataset", num],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro no download:\n{result.stderr}")
    print(result.stdout)
    context["ti"].xcom_push(key="dataset", value=DATASET)


def task_transform_silver(**context):
    """Executa o pipeline de limpeza e feature engineering (Silver local)."""
    import subprocess
    result = subprocess.run(
        ["python", "preprocess.py", "--dataset", DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro no preprocess:\n{result.stderr}")
    print(result.stdout)


def task_load_bq_bronze(**context):
    """Carrega os dados brutos para a camada Bronze do BigQuery."""
    import subprocess
    result = subprocess.run(
        ["python", "bigquery_loader.py", "--layer", "bronze", "--dataset", DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro no load bronze:\n{result.stderr}")
    print(result.stdout)


def task_load_bq_silver(**context):
    """Carrega os dados processados para a camada Silver do BigQuery."""
    import subprocess
    result = subprocess.run(
        ["python", "bigquery_loader.py", "--layer", "silver", "--dataset", DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro no load silver:\n{result.stderr}")
    print(result.stdout)


def task_train_model(**context):
    """Treina o Isolation Forest com os dados Silver."""
    import subprocess
    result = subprocess.run(
        ["python", "isolation_forest.py", "--dataset", DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro no treino:\n{result.stderr}")
    print(result.stdout)


def task_evaluate_model(**context):
    """Avalia o modelo e gera metricas."""
    import subprocess
    result = subprocess.run(
        ["python", "evaluate.py", "--dataset", DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro na avaliacao:\n{result.stderr}")
    print(result.stdout)


def task_load_bq_gold(**context):
    """Gera e carrega as tabelas Gold (KPIs e anomaly summary) no BigQuery."""
    import subprocess
    result = subprocess.run(
        ["python", "bigquery_loader.py", "--layer", "gold", "--dataset", DATASET],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Erro no load gold:\n{result.stderr}")
    print(result.stdout)


def task_notify_success(**context):
    """Loga resumo final do pipeline."""
    dag_run = context.get("dag_run")
    print(f"Pipeline concluido com sucesso!")
    print(f"DAG       : {dag_run.dag_id}")
    print(f"Run ID    : {dag_run.run_id}")
    print(f"Dataset   : {DATASET}")
    print(f"Projeto   : {GCP_PROJECT}")
    print(f"BigQuery  : console.cloud.google.com/bigquery?project={GCP_PROJECT}")


# ── DAG ───────────────────────────────────────────────────────────────────────

with DAG(
    dag_id="fraud_detection_pipeline",
    default_args=DEFAULT_ARGS,
    description="Pipeline end-to-end de deteccao de fraudes financeiras",
    schedule_interval="0 6 * * *",   # diario as 6h
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "ml", "bigquery", "production"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    ingest = PythonOperator(
        task_id="ingest_bronze",
        python_callable=task_ingest_bronze,
        doc_md="Baixa o dataset do Kaggle via API e salva em data/raw/",
    )

    transform = PythonOperator(
        task_id="transform_silver",
        python_callable=task_transform_silver,
        doc_md="Limpeza, normalizacao e feature engineering (Bronze -> Silver)",
    )

    load_bronze = PythonOperator(
        task_id="load_bigquery_bronze",
        python_callable=task_load_bq_bronze,
        doc_md="Carrega dados brutos para bankfraud-491117.bronze",
    )

    load_silver = PythonOperator(
        task_id="load_bigquery_silver",
        python_callable=task_load_bq_silver,
        doc_md="Carrega dados processados para bankfraud-491117.silver",
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        doc_md="Treina o Isolation Forest e salva os artefatos em models/",
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
        doc_md="Avalia o modelo e gera metricas em reports/figures/",
    )

    load_gold = PythonOperator(
        task_id="load_bigquery_gold",
        python_callable=task_load_bq_gold,
        doc_md="Gera KPIs e anomaly summary em bankfraud-491117.gold",
    )

    notify = PythonOperator(
        task_id="notify_success",
        python_callable=task_notify_success,
        doc_md="Loga resumo final do pipeline",
    )

    # ── Dependencias ──────────────────────────────────────────────────────────
    #
    # start
    #   └── ingest_bronze
    #         └── transform_silver
    #               ├── load_bigquery_bronze
    #               └── load_bigquery_silver
    #                     └── train_model
    #                           └── evaluate_model
    #                                 └── load_bigquery_gold
    #                                       └── notify_success
    #                                             └── end

    (
        start
        >> ingest
        >> transform
        >> load_bronze
        >> load_silver
        >> train
        >> evaluate
        >> load_gold
        >> notify
        >> end
    )