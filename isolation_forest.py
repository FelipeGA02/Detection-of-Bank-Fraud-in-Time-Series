"""
Treina e aplica o modelo Isolation Forest para deteccao de anomalias.

Suporta os 3 datasets do projeto com configuracoes especificas para cada um.

Uso:
    python isolation_forest.py --dataset ulb
    python isolation_forest.py --dataset ieee
    python isolation_forest.py --dataset sparkov
    python isolation_forest.py --dataset ulb --contamination 0.002
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Configuracoes por dataset ─────────────────────────────────────────────────

DATASET_CONFIGS = {
    "ulb": {
        "input":       os.path.join("data", "processed", "ulb_clean.csv"),
        "output":      os.path.join("data", "processed", "ulb_predictions.csv"),
        "model_path":  os.path.join("models", "isolation_forest_ulb.pkl"),
        "scaler_path": os.path.join("models", "scaler_ulb.pkl"),
        "contamination": 0.002,
        "feature_selector": "ulb",
    },
    "ieee": {
        "input":       os.path.join("data", "processed", "ieee_clean.csv"),
        "output":      os.path.join("data", "processed", "ieee_predictions.csv"),
        "model_path":  os.path.join("models", "isolation_forest_ieee.pkl"),
        "scaler_path": os.path.join("models", "scaler_ieee.pkl"),
        "contamination": 0.035,
        "feature_selector": "ieee",
    },
    "sparkov": {
        "input":       os.path.join("data", "processed", "sparkov_clean.csv"),
        "output":      os.path.join("data", "processed", "sparkov_predictions.csv"),
        "model_path":  os.path.join("models", "isolation_forest_sparkov.pkl"),
        "scaler_path": os.path.join("models", "scaler_sparkov.pkl"),
        "contamination": 0.005,
        "feature_selector": "sparkov",
    },
}

# ── Selecao de features por dataset ──────────────────────────────────────────

def select_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Retorna apenas as colunas numericas relevantes para o modelo."""

    common = ["amount", "log_amount", "amount_zscore", "is_high_value", "is_madrugada"]

    if dataset == "ulb":
        v_cols = [c for c in df.columns if c.startswith("V")]
        cols = common + ["hour"] + v_cols

    elif dataset == "ieee":
        cols = common + [
            "hour", "day_of_week",
            "has_email", "has_device", "v_null_ratio",
        ]

    elif dataset == "sparkov":
        cols = common + [
            "hour", "day_of_week", "month",
            "is_weekend", "amount_vs_category_avg",
            "seconds_since_last",
        ]
        if "geo_distance" in df.columns:
            cols.append("geo_distance")

    cols = [c for c in cols if c in df.columns]
    print(f"  Features selecionadas: {len(cols)}")
    print(f"  Colunas: {cols[:8]}{'...' if len(cols) > 8 else ''}")
    return df[cols]


# ── Treinamento ───────────────────────────────────────────────────────────────

def train(df: pd.DataFrame, dataset: str, contamination: float):
    """Treina o Isolation Forest e salva os artefatos."""

    cfg = DATASET_CONFIGS[dataset]
    os.makedirs("models", exist_ok=True)

    print(f"\nPreparando features...")
    X = select_features(df, dataset).fillna(0)

    print(f"  Shape para treino: {X.shape}")
    print(f"  Contamination    : {contamination}")

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nTreinando Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_scaled)

    joblib.dump(model,  cfg["model_path"])
    joblib.dump(scaler, cfg["scaler_path"])
    print(f"  Modelo salvo  : {cfg['model_path']}")
    print(f"  Scaler salvo  : {cfg['scaler_path']}")

    return model, scaler, X.columns.tolist()


# ── Predicao ──────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Carrega o modelo treinado e gera predicoes."""

    cfg = DATASET_CONFIGS[dataset]

    if not os.path.exists(cfg["model_path"]):
        raise FileNotFoundError(
            f"Modelo nao encontrado: {cfg['model_path']}\n"
            f"Execute com --train primeiro."
        )

    model  = joblib.load(cfg["model_path"])
    scaler = joblib.load(cfg["scaler_path"])

    X        = select_features(df, dataset).fillna(0)
    X_scaled = scaler.transform(X)

    df = df.copy()
    df["prediction"]    = model.predict(X_scaled)     
    df["anomaly_score"] = -model.score_samples(X_scaled)  
    df["is_fraud_pred"] = (df["prediction"] == -1).astype(int)

    return df


# ── Pipeline completo ─────────────────────────────────────────────────────────

def run(dataset: str, contamination: float = None, skip_train: bool = False) -> pd.DataFrame:
    cfg = DATASET_CONFIGS[dataset]

    if not contamination:
        contamination = cfg["contamination"]

    print("=" * 55)
    print(f"  ISOLATION FOREST — {dataset.upper()}")
    print("=" * 55)

    print(f"\nCarregando dados: {cfg['input']}")
    if not os.path.exists(cfg["input"]):
        raise FileNotFoundError(
            f"Arquivo nao encontrado: {cfg['input']}\n"
            "Execute preprocess.py primeiro."
        )
    df = pd.read_csv(cfg["input"])
    print(f"  Shape: {df.shape}")

    if not skip_train:
        train(df, dataset, contamination)
    else:
        print("\nUsando modelo existente (--skip-train).")

    print("\nGerando predicoes...")
    df = predict(df, dataset)

    n_anomalias = df["is_fraud_pred"].sum()
    total       = len(df)
    taxa        = n_anomalias / total * 100

    print(f"\n{'='*55}")
    print(f"  RESULTADOS")
    print(f"{'='*55}")
    print(f"  Total transacoes  : {total:,}")
    print(f"  Anomalias (pred)  : {n_anomalias:,} ({taxa:.2f}%)")

    if "is_fraud" in df.columns:
        real = df["is_fraud"].sum()
        print(f"  Fraudes reais     : {real:,} ({real/total*100:.3f}%)")

    print(f"  Score medio       : {df['anomaly_score'].mean():.4f}")
    print(f"  Score maximo      : {df['anomaly_score'].max():.4f}")

    os.makedirs(os.path.dirname(cfg["output"]), exist_ok=True)
    df.to_csv(cfg["output"], index=False)
    print(f"\n  Predicoes salvas  : {cfg['output']}")
    print(f"{'='*55}")
    print("\nProximo passo: python evaluate.py --dataset " + dataset)

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Isolation Forest para deteccao de fraudes"
)
parser.add_argument(
    "--dataset", type=str, choices=["ulb", "ieee", "sparkov"], default="ulb",
    help="Dataset: ulb | ieee | sparkov  (padrao: ulb)"
)
parser.add_argument(
    "--contamination", type=float, default=None,
    help="Proporcao esperada de anomalias (ex: 0.01 = 1%%)"
)
parser.add_argument(
    "--skip-train", action="store_true",
    help="Pula o treinamento e usa o modelo ja salvo"
)
args = parser.parse_args()

run(
    dataset=args.dataset,
    contamination=args.contamination,
    skip_train=args.skip_train,
)

