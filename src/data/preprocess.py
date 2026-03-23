"""
preprocess.py
-------------
Pipeline Bronze -> Silver para os datasets reais de fraud detection do Kaggle.

Suporta 3 datasets com schemas diferentes:
  --dataset ulb      ULB Credit Card Fraud (creditcard.csv)
  --dataset ieee     IEEE-CIS Fraud Detection (join transaction + identity)
  --dataset sparkov  Sparkov Simulated Transactions (fraudTrain.csv)

Uso:
    python preprocess.py --dataset ulb
    python preprocess.py --dataset ieee
    python preprocess.py --dataset sparkov
    python preprocess.py --dataset ulb --output data/processed/ulb_clean.csv

Dependencias:
    pip install pandas numpy
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── Caminhos padrao por dataset ───────────────────────────────────────────────

DATASET_CONFIGS = {
    "ulb": {
        "name": "ULB Credit Card Fraud",
        "input": os.path.join("data", "raw", "ulb", "creditcard.csv"),
        "output": os.path.join("data", "processed", "ulb_clean.csv"),
        "report": os.path.join("data", "processed", "ulb_quality_report.txt"),
    },
    "ieee": {
        "name": "IEEE-CIS Fraud Detection",
        "input_transaction": os.path.join("data", "raw", "ieee", "train_transaction.csv"),
        "input_identity":    os.path.join("data", "raw", "ieee", "train_identity.csv"),
        "output": os.path.join("data", "processed", "ieee_clean.csv"),
        "report": os.path.join("data", "processed", "ieee_quality_report.txt"),
    },
    "sparkov": {
        "name": "Sparkov Simulated Transactions",
        "input": os.path.join("data", "raw", "sparkov", "fraudTrain.csv"),
        "output": os.path.join("data", "processed", "sparkov_clean.csv"),
        "report": os.path.join("data", "processed", "sparkov_quality_report.txt"),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_ulb(cfg: dict) -> pd.DataFrame:
    """
    ULB Credit Card Fraud
    Colunas: Time, V1..V28 (PCA anonimizado), Amount, Class
    """
    _check_file(cfg["input"], "ulb")
    print(f"Carregando ULB de: {cfg['input']}")
    df = pd.read_csv(cfg["input"])
    df = df.rename(columns={"Time": "time_seconds", "Amount": "amount", "Class": "is_fraud"})
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["dataset"]  = "ulb"
    _print_load_summary(df)
    return df


def load_ieee(cfg: dict) -> pd.DataFrame:
    """
    IEEE-CIS Fraud Detection
    Join entre train_transaction.csv e train_identity.csv por TransactionID.
    """
    _check_file(cfg["input_transaction"], "ieee (transaction)")
    _check_file(cfg["input_identity"],    "ieee (identity)")

    print(f"Carregando IEEE transaction: {cfg['input_transaction']}")
    print("  (arquivo grande ~400MB, pode demorar...)")
    df_tx = pd.read_csv(cfg["input_transaction"])

    print(f"Carregando IEEE identity   : {cfg['input_identity']}")
    df_id = pd.read_csv(cfg["input_identity"])

    print("Fazendo join por TransactionID (left join)...")
    df = df_tx.merge(df_id, on="TransactionID", how="left")

    df = df.rename(columns={
        "TransactionID":  "transaction_id",
        "TransactionDT":  "time_seconds",
        "TransactionAmt": "amount",
        "isFraud":        "is_fraud",
        "ProductCD":      "product_category",
        "card4":          "card_type",
        "card6":          "card_category",
        "P_emaildomain":  "purchaser_email_domain",
        "R_emaildomain":  "recipient_email_domain",
        "DeviceType":     "device_type",
        "DeviceInfo":     "device_info",
    })
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["dataset"]  = "ieee"
    _print_load_summary(df)
    return df


def load_sparkov(cfg: dict) -> pd.DataFrame:
    """
    Sparkov Simulated Transactions
    Dataset com timestamp real, merchant, category, localizacao geografica.
    """
    _check_file(cfg["input"], "sparkov")
    print(f"Carregando Sparkov de: {cfg['input']}")
    df = pd.read_csv(cfg["input"], parse_dates=["trans_date_trans_time"])
    df = df.rename(columns={
        "trans_date_trans_time": "timestamp",
        "cc_num":                "card_number",
        "amt":                   "amount",
        "trans_num":             "transaction_id",
        "merch_lat":             "merchant_lat",
        "merch_long":            "merchant_long",
    })
    df["is_fraud"] = df["is_fraud"].astype(bool)
    df["dataset"]  = "sparkov"
    _print_load_summary(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  LIMPEZA GENERICA
# ══════════════════════════════════════════════════════════════════════════════

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas, valores invalidos e corrige tipos."""
    print("\nLimpando dados...")
    before = len(df)

    # Duplicatas
    id_col = "transaction_id" if "transaction_id" in df.columns else None
    df = df.drop_duplicates(subset=[id_col]) if id_col else df.drop_duplicates()

    # Valores invalidos
    df = df[df["amount"].notna() & (df["amount"] >= 0)]
    df = df[df["is_fraud"].notna()]

    # Tipos
    df["amount"]   = df["amount"].astype(float)
    df["is_fraud"] = df["is_fraud"].astype(bool)

    removed = before - len(df)
    print(f"  Registros removidos: {removed:,}" if removed else "  Nenhum registro invalido.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — especifico por dataset
# ══════════════════════════════════════════════════════════════════════════════

def features_ulb(df: pd.DataFrame) -> pd.DataFrame:
    """
    ULB: sem timestamp real.
    Usa time_seconds (segundos desde 1a transacao do dia) para aproximar hora.
    """
    print("Gerando features ULB...")

    # Hora aproximada (ciclo de 24h = 86400 segundos)
    df["hour"]          = (df["time_seconds"] % 86400 // 3600).astype(int)
    df["is_madrugada"]  = df["hour"].between(0, 5).astype(int)

    # Transformacoes do valor
    df["log_amount"]    = np.log1p(df["amount"])
    mean_a, std_a       = df["amount"].mean(), df["amount"].std()
    df["amount_zscore"] = (df["amount"] - mean_a) / (std_a + 1e-6)

    # Flag: valor acima do percentil 95
    df["is_high_value"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)

    print("  Features: hour, is_madrugada, log_amount, amount_zscore, is_high_value")
    return df


def features_ieee(df: pd.DataFrame) -> pd.DataFrame:
    """
    IEEE-CIS: TransactionDT e timedelta em segundos (nao timestamp real).
    Features comportamentais baseadas nos campos de identidade.
    """
    print("Gerando features IEEE-CIS...")

    df["hour"]         = (df["time_seconds"] % 86400 // 3600).astype(int)
    df["day_of_week"]  = (df["time_seconds"] // 86400 % 7).astype(int)
    df["is_madrugada"] = df["hour"].between(0, 5).astype(int)
    df["log_amount"]   = np.log1p(df["amount"])

    mean_a, std_a       = df["amount"].mean(), df["amount"].std()
    df["amount_zscore"] = (df["amount"] - mean_a) / (std_a + 1e-6)

    # Sinais de dado ausente (comum em fraudes IEEE)
    df["has_email"]  = df["purchaser_email_domain"].notna().astype(int)
    df["has_device"] = df["device_type"].notna().astype(int)

    # Proporcao de campos V nulos por transacao
    v_cols = [c for c in df.columns if c.startswith("V")]
    if v_cols:
        df["v_null_ratio"] = df[v_cols].isnull().mean(axis=1)

    print("  Features: hour, day_of_week, is_madrugada, log_amount,")
    print("            amount_zscore, has_email, has_device, v_null_ratio")
    return df


def features_sparkov(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sparkov: dataset mais rico com timestamp real e localizacao geografica.
    """
    print("Gerando features Sparkov...")

    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_madrugada"] = df["hour"].between(0, 5).astype(int)
    df["log_amount"]   = np.log1p(df["amount"])

    # Valor relativo a media da categoria
    cat_avg = df.groupby("category")["amount"].transform("mean")
    df["amount_vs_category_avg"] = df["amount"] / (cat_avg + 1e-6)

    # Distancia geografica cliente <-> estabelecimento
    if all(c in df.columns for c in ["lat", "long", "merchant_lat", "merchant_long"]):
        df["geo_distance"] = np.sqrt(
            (df["lat"]  - df["merchant_lat"])  ** 2 +
            (df["long"] - df["merchant_long"]) ** 2
        )

    # Tempo desde ultima transacao do mesmo cartao (em segundos)
    df = df.sort_values(["card_number", "timestamp"])
    df["seconds_since_last"] = (
        df.groupby("card_number")["timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )

    print("  Features: hour, day_of_week, month, is_weekend, is_madrugada,")
    print("            log_amount, amount_vs_category_avg, geo_distance, seconds_since_last")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  RELATORIO E SALVAMENTO
# ══════════════════════════════════════════════════════════════════════════════

def quality_report(df: pd.DataFrame, dataset: str, stage: str) -> dict:
    nulls = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
    return {
        "dataset": dataset,
        "stage": stage,
        "rows": len(df),
        "cols": df.shape[1],
        "fraud_count": int(df["is_fraud"].sum()),
        "fraud_rate":  round(float(df["is_fraud"].mean()) * 100, 4),
        "nulls": nulls,
    }


def print_report(r: dict) -> None:
    print(f"\n--- Qualidade [{r['stage']}] ---")
    print(f"  Linhas   : {r['rows']:,}")
    print(f"  Colunas  : {r['cols']}")
    print(f"  Fraudes  : {r['fraud_count']:,} ({r['fraud_rate']}%)")
    top = dict(list(r["nulls"].items())[:4])
    print(f"  Nulos    : {top if top else 'nenhum'}")


def save_report(reports: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("RELATORIO DE QUALIDADE — BRONZE -> SILVER\n")
        f.write("=" * 50 + "\n\n")
        for r in reports:
            f.write(f"ETAPA  : {r['stage']}\n")
            f.write(f"Linhas : {r['rows']:,}\n")
            f.write(f"Fraudes: {r['fraud_count']:,} ({r['fraud_rate']}%)\n")
            f.write(f"Nulos  : {r['nulls']}\n\n")
    print(f"Relatorio salvo: {path}")


def save_silver(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Silver salvo   : {path}  ({mb:.1f} MB)")
    print(f"Shape final    : {df.shape[0]:,} linhas x {df.shape[1]} colunas")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _check_file(path: str, name: str) -> None:
    if not os.path.exists(path):
        print(f"\nERRO: arquivo nao encontrado: {path}")
        print(f"Execute primeiro: python download_data.py --dataset {name.split()[0]}")
        sys.exit(1)


def _print_load_summary(df: pd.DataFrame) -> None:
    fraud_n = df["is_fraud"].sum()
    rate    = df["is_fraud"].mean() * 100
    print(f"  Linhas   : {len(df):,}")
    print(f"  Colunas  : {df.shape[1]}")
    print(f"  Fraudes  : {fraud_n:,} ({rate:.3f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(dataset: str) -> pd.DataFrame:
    cfg = DATASET_CONFIGS[dataset]

    print("=" * 55)
    print(f"  PIPELINE BRONZE -> SILVER  |  {cfg['name']}")
    print("=" * 55)

    # 1. Carrega
    loaders = {"ulb": load_ulb, "ieee": load_ieee, "sparkov": load_sparkov}
    df = loaders[dataset](cfg)
    r_bronze = quality_report(df, dataset, "BRONZE (entrada)")
    print_report(r_bronze)

    # 2. Limpa
    df = clean(df)

    # 3. Features
    feature_fns = {"ulb": features_ulb, "ieee": features_ieee, "sparkov": features_sparkov}
    df = feature_fns[dataset](df)

    # 4. Relatorio final
    r_silver = quality_report(df, dataset, "SILVER (saida)")
    print_report(r_silver)

    # 5. Salva
    save_silver(df, cfg["output"])
    save_report([r_bronze, r_silver], cfg["report"])

    print("\nPipeline concluido com sucesso!")
    print("=" * 55)
    print(f"\nProximo passo:")
    print(f"  Abra notebooks/01_eda.ipynb e carregue: {cfg['output']}")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Bronze -> Silver para datasets de fraud detection"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["ulb", "ieee", "sparkov"], default="ulb",
        help="Dataset: ulb | ieee | sparkov  (padrao: ulb)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Caminho de saida customizado para o CSV Silver"
    )
    args = parser.parse_args()

    if args.output:
        DATASET_CONFIGS[args.dataset]["output"] = args.output

    run_pipeline(args.dataset)


if __name__ == "__main__":
    main()