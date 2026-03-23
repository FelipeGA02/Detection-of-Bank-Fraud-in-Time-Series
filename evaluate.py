"""
Avalia o desempenho do modelo Isolation Forest e realiza A/B Testing
comparando o novo modelo com um sistema de regras fixas (baseline).

Uso:
    python evaluate.py --dataset ulb
    python evaluate.py --dataset ieee
    python evaluate.py --dataset sparkov

"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="darkgrid")
plt.rcParams["figure.dpi"] = 110

# ── Configuracoes por dataset ─────────────────────────────────────────────────

DATASET_CONFIGS = {
    "ulb": {
        "predictions": os.path.join("data", "processed", "ulb_predictions.csv"),
        "silver":      os.path.join("data", "processed", "ulb_clean.csv"),
        "feature_selector": "ulb",
    },
    "ieee": {
        "predictions": os.path.join("data", "processed", "ieee_predictions.csv"),
        "silver":      os.path.join("data", "processed", "ieee_clean.csv"),
        "feature_selector": "ieee",
    },
    "sparkov": {
        "predictions": os.path.join("data", "processed", "sparkov_predictions.csv"),
        "silver":      os.path.join("data", "processed", "sparkov_clean.csv"),
        "feature_selector": "sparkov",
    },
}


# ── Selecao de features (igual ao isolation_forest.py) ───────────────────────

def select_features(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    common = ["amount", "log_amount", "amount_zscore", "is_high_value", "is_madrugada"]
    if dataset == "ulb":
        v_cols = [c for c in df.columns if c.startswith("V")]
        cols = common + ["hour"] + v_cols
    elif dataset == "ieee":
        cols = common + ["hour", "day_of_week", "has_email", "has_device", "v_null_ratio"]
    elif dataset == "sparkov":
        cols = common + ["hour", "day_of_week", "month", "is_weekend",
                         "amount_vs_category_avg", "seconds_since_last"]
        if "geo_distance" in df.columns:
            cols.append("geo_distance")
    cols = [c for c in cols if c in df.columns]
    return df[cols]


# ── Metricas do Isolation Forest ──────────────────────────────────────────────

def evaluate_isolation_forest(df: pd.DataFrame, dataset: str) -> dict:
    """Calcula metricas do Isolation Forest (requer is_fraud real)."""

    if "is_fraud" not in df.columns or "is_fraud_pred" not in df.columns:
        print("Colunas 'is_fraud' e 'is_fraud_pred' necessarias para avaliacao supervisionada.")
        return {}

    y_true = df["is_fraud"].astype(int)
    y_pred = df["is_fraud_pred"].astype(int)
    score  = df["anomaly_score"]

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, score),
        "avg_precision": average_precision_score(y_true, score),
    }

    print(f"\n{'='*55}")
    print(f"  METRICAS — Isolation Forest ({dataset.upper()})")
    print(f"{'='*55}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1-Score     : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision: {metrics['avg_precision']:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraude"]))

    return metrics


# ── Regressao Logistica (baseline supervisionado) ─────────────────────────────

def train_logistic(df: pd.DataFrame, dataset: str) -> tuple:
    """Treina Regressao Logistica como modelo supervisionado comparativo."""

    if "is_fraud" not in df.columns:
        print("Coluna 'is_fraud' necessaria para Regressao Logistica.")
        return None, None, None, None

    X = select_features(df, dataset).fillna(0)
    y = df["is_fraud"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1_score":  f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
    }

    print(f"\n{'='*55}")
    print(f"  METRICAS — Regressao Logistica ({dataset.upper()})")
    print(f"{'='*55}")
    print(f"  Precision    : {metrics['precision']:.4f}")
    print(f"  Recall       : {metrics['recall']:.4f}")
    print(f"  F1-Score     : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC      : {metrics['roc_auc']:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraude"]))

    return pipeline, metrics, (y_test, y_pred, y_proba), X_test


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, title: str, path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraude"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_roc_curves(results: dict, dataset: str) -> None:
    """Plota curvas ROC de todos os modelos no mesmo grafico."""
    plt.figure(figsize=(7, 5))
    for name, (y_true, _, y_score) in results.items():
        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Curva ROC — {dataset.upper()}")
    plt.legend()
    plt.tight_layout()
    path = f"reports/figures/roc_curve_{dataset}.png"
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_precision_recall(results: dict, dataset: str) -> None:
    """Plota curvas Precision-Recall (essencial para dados desbalanceados)."""
    plt.figure(figsize=(7, 5))
    for name, (y_true, _, y_score) in results.items():
        if y_score is not None:
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            plt.plot(rec, prec, linewidth=2, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Curva Precision-Recall — {dataset.upper()}")
    plt.legend()
    plt.tight_layout()
    path = f"reports/figures/precision_recall_{dataset}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def plot_anomaly_score_dist(df: pd.DataFrame, dataset: str) -> None:
    """Distribuicao do anomaly score por classe real."""
    if "is_fraud" not in df.columns or "anomaly_score" not in df.columns:
        return
    plt.figure(figsize=(9, 4))
    df[df["is_fraud"]==0]["anomaly_score"].hist(bins=60, alpha=0.6, color="steelblue", label="Normal", density=True)
    df[df["is_fraud"]==1]["anomaly_score"].hist(bins=60, alpha=0.7, color="crimson",   label="Fraude", density=True)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Densidade")
    plt.title(f"Distribuicao do Anomaly Score — {dataset.upper()}")
    plt.legend()
    plt.tight_layout()
    path = f"reports/figures/anomaly_score_dist_{dataset}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.show()


# ── A/B Testing ───────────────────────────────────────────────────────────────

def ab_testing(metrics_if: dict, metrics_lr: dict, dataset: str) -> None:
    """
    Compara Isolation Forest (novo modelo) vs sistema de regras fixas (baseline).
    O baseline e simulado com as metricas tipicas de sistemas baseados em regras.
    """

    baseline = {
        "precision": 0.60,
        "recall":    0.70,
        "f1_score":  0.646,
        "roc_auc":   0.00, 
    }

    print(f"\n{'='*55}")
    print(f"  A/B TESTING — {dataset.upper()}")
    print(f"  Grupo A: Regras Fixas (baseline)")
    print(f"  Grupo B: Isolation Forest (novo modelo)")
    print(f"{'='*55}")

    rows = []
    for metric in ["precision", "recall", "f1_score"]:
        rows.append({
            "Metrica":          metric.replace("_", " ").title(),
            "Regras Fixas (A)": f"{baseline[metric]:.3f}",
            "Isolation Forest (B)": f"{metrics_if.get(metric, 0):.3f}" if metrics_if else "—",
            "Logistica (B2)":   f"{metrics_lr.get(metric, 0):.3f}" if metrics_lr else "—",
        })

    result_df = pd.DataFrame(rows)
    print(result_df.to_string(index=False))

    if metrics_if:
        total     = 10000 
        fraud_tot = int(total * 0.02)
        normal    = total - fraud_tot

        detected_a  = int(fraud_tot * baseline["recall"])
        detected_b  = int(fraud_tot * metrics_if.get("recall", 0))

        tabela = [
            [detected_a,  fraud_tot - detected_a],
            [detected_b,  fraud_tot - detected_b],
        ]
        try:
            chi2, p, _, _ = chi2_contingency(tabela)
            print(f"\n  Teste Qui-quadrado (deteccao A vs B):")
            print(f"    Chi2    : {chi2:.4f}")
            print(f"    p-value : {p:.4f}")
            print(f"    {'Diferenca estatisticamente significativa ✓' if p < 0.05 else 'Diferenca nao significativa ✗'}")
        except Exception:
            pass

    if metrics_if and metrics_lr:
        labels   = ["Precision", "Recall", "F1-Score"]
        x        = np.arange(len(labels))
        width    = 0.25

        vals_a  = [baseline["precision"], baseline["recall"], baseline["f1_score"]]
        vals_b  = [metrics_if.get("precision",0), metrics_if.get("recall",0), metrics_if.get("f1_score",0)]
        vals_b2 = [metrics_lr.get("precision",0), metrics_lr.get("recall",0), metrics_lr.get("f1_score",0)]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - width, vals_a,  width, label="Regras Fixas",       color="gray",      edgecolor="white")
        ax.bar(x,         vals_b,  width, label="Isolation Forest",   color="steelblue", edgecolor="white")
        ax.bar(x + width, vals_b2, width, label="Reg. Logistica",     color="seagreen",  edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title(f"A/B Testing — Comparativo de Modelos ({dataset.upper()})")
        ax.legend()
        plt.tight_layout()
        os.makedirs("reports/figures", exist_ok=True)
        plt.savefig(f"reports/figures/ab_testing_{dataset}.png", bbox_inches="tight")
        plt.show()


# ── Pipeline principal ────────────────────────────────────────────────────────

def run(dataset: str) -> None:
    cfg = DATASET_CONFIGS[dataset]
    os.makedirs("reports/figures", exist_ok=True)

    print("=" * 55)
    print(f"  AVALIACAO DE MODELOS — {dataset.upper()}")
    print("=" * 55)

    if not os.path.exists(cfg["predictions"]):
        raise FileNotFoundError(
            f"Predicoes nao encontradas: {cfg['predictions']}\n"
            "Execute isolation_forest.py primeiro."
        )
    df = pd.read_csv(cfg["predictions"])
    print(f"\nPredicoes carregadas: {df.shape}")

    metrics_if = evaluate_isolation_forest(df, dataset)

    plot_anomaly_score_dist(df, dataset)

    if "is_fraud" in df.columns:
        y_true = df["is_fraud"].astype(int)
        y_pred = df["is_fraud_pred"].astype(int)
        plot_confusion_matrix(
            y_true, y_pred,
            f"Confusion Matrix — Isolation Forest ({dataset.upper()})",
            f"reports/figures/cm_if_{dataset}.png"
        )

    df_silver = pd.read_csv(cfg["silver"])
    _, metrics_lr, lr_results, _ = train_logistic(df_silver, dataset)

    roc_data = {}
    if "is_fraud" in df.columns and "anomaly_score" in df.columns:
        roc_data["Isolation Forest"] = (
            df["is_fraud"].astype(int),
            df["is_fraud_pred"].astype(int),
            df["anomaly_score"],
        )
    if lr_results:
        y_test, y_pred_lr, y_proba_lr = lr_results
        roc_data["Reg. Logistica"] = (y_test, y_pred_lr, y_proba_lr)

    if roc_data:
        plot_roc_curves(roc_data, dataset)
        plot_precision_recall(roc_data, dataset)

    ab_testing(metrics_if, metrics_lr or {}, dataset)

    print(f"\n{'='*55}")
    print(f"  RESUMO FINAL — {dataset.upper()}")
    print(f"{'='*55}")
    if metrics_if:
        print(f"  Isolation Forest  F1={metrics_if['f1_score']:.3f}  AUC={metrics_if['roc_auc']:.3f}")
    if metrics_lr:
        print(f"  Reg. Logistica    F1={metrics_lr['f1_score']:.3f}  AUC={metrics_lr['roc_auc']:.3f}")
    print(f"\n  Graficos salvos em: reports/figures/")
    print(f"{'='*55}")


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Avaliacao de modelos e A/B Testing"
)
parser.add_argument(
    "--dataset", type=str, choices=["ulb", "ieee", "sparkov"], default="ulb",
    help="Dataset: ulb | ieee | sparkov  (padrao: ulb)"
)
args = parser.parse_args()
run(args.dataset)

