"""
bigquery_loader.py
------------------
Carrega os dados processados (Silver) para o BigQuery nas camadas
Bronze, Silver e Gold do projeto fraud-detection.

Projeto GCP : bankfraud-491117
Dataset BQ  : bronze / silver / gold

Uso:
    python bigquery_loader.py --layer bronze --dataset ulb
    python bigquery_loader.py --layer silver --dataset ulb
    python bigquery_loader.py --layer gold
    python bigquery_loader.py --all --dataset ulb

Dependencias:
    pip install google-cloud-bigquery pandas pyarrow
"""

import argparse
import os

import pandas as pd
from google.cloud import bigquery

# ── Configuracoes do projeto ──────────────────────────────────────────────────

GCP_PROJECT  = "bankfraud-491117"
BQ_LOCATION  = "US"

DATASETS_BQ = {
    "bronze": f"{GCP_PROJECT}.bronze",
    "silver": f"{GCP_PROJECT}.silver",
    "gold":   f"{GCP_PROJECT}.gold",
}

# Arquivos locais por dataset e camada
FILE_MAP = {
    "ulb": {
        "bronze": os.path.join("data", "raw",       "ulb", "creditcard.csv"),
        "silver": os.path.join("data", "processed", "ulb_clean.csv"),
        "pred":   os.path.join("data", "processed", "ulb_predictions.csv"),
    },
    "ieee": {
        "bronze_tx": os.path.join("data", "raw",       "ieee", "train_transaction.csv"),
        "bronze_id": os.path.join("data", "raw",       "ieee", "train_identity.csv"),
        "silver":    os.path.join("data", "processed", "ieee_clean.csv"),
        "pred":      os.path.join("data", "processed", "ieee_predictions.csv"),
    },
    "sparkov": {
        "bronze": os.path.join("data", "raw",       "sparkov", "fraudTrain.csv"),
        "silver": os.path.join("data", "processed", "sparkov_clean.csv"),
        "pred":   os.path.join("data", "processed", "sparkov_predictions.csv"),
    },
}


# ── Cliente BigQuery ──────────────────────────────────────────────────────────

def get_client() -> bigquery.Client:
    """Retorna cliente autenticado do BigQuery."""
    try:
        client = bigquery.Client(project=GCP_PROJECT)
        print(f"BigQuery conectado: {GCP_PROJECT}")
        return client
    except Exception as e:
        print(f"\nERRO ao conectar ao BigQuery: {e}")
        print("Verifique se executou: gcloud auth application-default login")
        raise


def create_dataset_if_not_exists(client: bigquery.Client, dataset_id: str) -> None:
    """Cria o dataset no BigQuery se nao existir."""
    full_id = f"{GCP_PROJECT}.{dataset_id}"
    dataset  = bigquery.Dataset(full_id)
    dataset.location = BQ_LOCATION
    try:
        client.get_dataset(full_id)
        print(f"  Dataset ja existe : {full_id}")
    except Exception:
        client.create_dataset(dataset, exists_ok=True)
        print(f"  Dataset criado    : {full_id}")


# ── Carga generica ────────────────────────────────────────────────────────────

def load_csv_to_bq(
    client: bigquery.Client,
    csv_path: str,
    table_id: str,
    write_mode: str = "WRITE_TRUNCATE",
    chunksize: int = 100_000,
) -> None:
    """
    Carrega um CSV para uma tabela do BigQuery.
    Para arquivos grandes, faz a carga em chunks.
    """
    if not os.path.exists(csv_path):
        print(f"  AVISO: arquivo nao encontrado: {csv_path}")
        return

    file_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"  Carregando: {csv_path} ({file_mb:.1f} MB)")
    print(f"  Destino   : {table_id}")

    # Arquivos grandes: carrega em chunks via DataFrame
    if file_mb > 50:
        print(f"  Arquivo grande — carregando em chunks de {chunksize:,} linhas...")
        first_chunk = True
        total_rows  = 0

        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            mode = "WRITE_TRUNCATE" if first_chunk else "WRITE_APPEND"
            # schema_update_options so pode ser usado com WRITE_APPEND
            cfg = bigquery.LoadJobConfig(
                write_disposition=mode,
                autodetect=True if first_chunk else False,
            )
            job = client.load_table_from_dataframe(chunk, table_id, job_config=cfg)
            job.result()
            total_rows  += len(chunk)
            first_chunk  = False
            print(f"    Chunk {i+1}: {total_rows:,} linhas enviadas...")
    else:
        # Arquivo pequeno: carga direta sem schema_update_options
        job_config = bigquery.LoadJobConfig(
            write_disposition=write_mode,
            autodetect=True,
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
        )
        with open(csv_path, "rb") as f:
            job = client.load_table_from_file(f, table_id, job_config=job_config)
        job.result()

    table = client.get_table(table_id)
    print(f"  OK — {table.num_rows:,} linhas em {table_id}\n")


# ── Camada Bronze ─────────────────────────────────────────────────────────────

def load_bronze(client: bigquery.Client, dataset: str) -> None:
    """Carrega dados brutos para a camada Bronze."""
    print("\n--- BRONZE ---")
    create_dataset_if_not_exists(client, "bronze")
    files = FILE_MAP[dataset]

    if dataset == "ulb":
        load_csv_to_bq(client, files["bronze"], f"{DATASETS_BQ['bronze']}.raw_transactions_ulb")

    elif dataset == "ieee":
        load_csv_to_bq(client, files["bronze_tx"], f"{DATASETS_BQ['bronze']}.raw_transactions_ieee")
        load_csv_to_bq(client, files["bronze_id"], f"{DATASETS_BQ['bronze']}.raw_identity_ieee")

    elif dataset == "sparkov":
        load_csv_to_bq(client, files["bronze"], f"{DATASETS_BQ['bronze']}.raw_transactions_sparkov")


# ── Camada Silver ─────────────────────────────────────────────────────────────

def load_silver(client: bigquery.Client, dataset: str) -> None:
    """Carrega dados processados para a camada Silver."""
    print("\n--- SILVER ---")
    create_dataset_if_not_exists(client, "silver")
    files = FILE_MAP[dataset]
    load_csv_to_bq(client, files["silver"], f"{DATASETS_BQ['silver']}.processed_transactions_{dataset}")

    # Carrega predicoes se existirem
    if os.path.exists(files.get("pred", "")):
        load_csv_to_bq(client, files["pred"], f"{DATASETS_BQ['silver']}.fraud_predictions_{dataset}")


# ── Camada Gold ───────────────────────────────────────────────────────────────

def load_gold(client: bigquery.Client, dataset: str) -> None:
    """
    Gera e carrega as tabelas Gold via queries SQL no BigQuery.
    Requer que a camada Silver ja esteja carregada.
    """
    print("\n--- GOLD ---")
    create_dataset_if_not_exists(client, "gold")

    silver_table = f"{DATASETS_BQ['silver']}.fraud_predictions_{dataset}"

    # Verifica se a tabela Silver existe
    try:
        client.get_table(silver_table)
    except Exception:
        print(f"  AVISO: tabela Silver nao encontrada: {silver_table}")
        print(f"  Execute --layer silver primeiro.")
        return

    queries = {
        # KPIs diarios de fraude
        f"{DATASETS_BQ['gold']}.fraud_kpis_daily_{dataset}": f"""
            SELECT
                DATE(TIMESTAMP_SECONDS(CAST(time_seconds AS INT64))) AS data,
                COUNT(*)                                              AS total_transacoes,
                SUM(CAST(is_fraud_pred AS INT64))                    AS fraudes_detectadas,
                SUM(CASE WHEN is_fraud_pred = 1 THEN amount ELSE 0 END) AS valor_bloqueado,
                ROUND(AVG(CAST(is_fraud_pred AS INT64)) * 100, 4)    AS taxa_fraude_pct,
                AVG(anomaly_score)                                    AS avg_anomaly_score
            FROM `{silver_table}`
            GROUP BY 1
            ORDER BY 1 DESC
        """ if dataset == "ulb" else f"""
            SELECT
                DATE(timestamp)                                          AS data,
                COUNT(*)                                                 AS total_transacoes,
                SUM(CAST(is_fraud_pred AS INT64))                       AS fraudes_detectadas,
                SUM(CASE WHEN is_fraud_pred = 1 THEN amount ELSE 0 END) AS valor_bloqueado,
                ROUND(AVG(CAST(is_fraud_pred AS INT64)) * 100, 4)       AS taxa_fraude_pct,
                AVG(anomaly_score)                                       AS avg_anomaly_score
            FROM `{silver_table}`
            GROUP BY 1
            ORDER BY 1 DESC
        """,

        # Score de risco por cliente (apenas Sparkov tem user_id rico)
        f"{DATASETS_BQ['gold']}.anomaly_summary_{dataset}": f"""
            SELECT
                ROUND(anomaly_score, 1)             AS score_bucket,
                COUNT(*)                            AS total,
                SUM(CAST(is_fraud_pred AS INT64))   AS fraudes,
                ROUND(AVG(amount), 2)               AS avg_amount,
                CASE
                    WHEN anomaly_score >= 0.7 THEN 'ALTO'
                    WHEN anomaly_score >= 0.4 THEN 'MEDIO'
                    ELSE 'BAIXO'
                END AS nivel_risco
            FROM `{silver_table}`
            GROUP BY 1, 5
            ORDER BY 1 DESC
        """,
    }

    for table_dest, query in queries.items():
        print(f"  Gerando: {table_dest.split('.')[-1]}")
        job_config = bigquery.QueryJobConfig(
            destination=table_dest,
            write_disposition="WRITE_TRUNCATE",
            create_disposition="CREATE_IF_NEEDED",
        )
        job = client.query(query, job_config=job_config)
        job.result()
        t = client.get_table(table_dest)
        print(f"  OK — {t.num_rows:,} linhas\n")


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run_all(dataset: str) -> None:
    client = get_client()
    print(f"\n{'='*55}")
    print(f"  CARGA COMPLETA BigQuery — {dataset.upper()}")
    print(f"  Projeto: {GCP_PROJECT}")
    print(f"{'='*55}")
    load_bronze(client, dataset)
    load_silver(client, dataset)
    load_gold(client, dataset)
    print(f"\n{'='*55}")
    print(f"  Pipeline BigQuery concluido!")
    print(f"  Acesse: console.cloud.google.com/bigquery?project={GCP_PROJECT}")
    print(f"{'='*55}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Carga de dados para o BigQuery")
    parser.add_argument("--layer",   type=str, choices=["bronze", "silver", "gold"], help="Camada a carregar")
    parser.add_argument("--dataset", type=str, choices=["ulb", "ieee", "sparkov"], default="ulb")
    parser.add_argument("--all",     action="store_true", help="Carrega todas as camadas")
    args = parser.parse_args()

    client = get_client()

    if args.all:
        run_all(args.dataset)
    elif args.layer == "bronze":
        create_dataset_if_not_exists(client, "bronze")
        load_bronze(client, args.dataset)
    elif args.layer == "silver":
        create_dataset_if_not_exists(client, "silver")
        load_silver(client, args.dataset)
    elif args.layer == "gold":
        load_gold(client, args.dataset)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()