# 🔍 Detection of Bank Fraud in Time Series

Sistema end-to-end de detecção de fraudes financeiras com Engenharia de Dados, Ciência de Dados e BI — usando dados reais de transações.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=flat&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📌 Índice

1. [Visão Geral](#1-visão-geral)
2. [Problema de Negócio](#2-problema-de-negócio)
3. [Datasets](#3-datasets)
4. [Arquitetura do Projeto](#4-arquitetura-do-projeto)
5. [Como Executar](#5-como-executar)
6. [Pipeline de Dados](#6-pipeline-de-dados)
7. [Análise Exploratória — EDA](#7-análise-exploratória--eda)
8. [Feature Engineering](#8-feature-engineering)
9. [Testes Estatísticos](#9-testes-estatísticos)
10. [Modelos de Machine Learning](#10-modelos-de-machine-learning)
11. [Validação e Métricas](#11-validação-e-métricas)
12. [A/B Testing](#12-ab-testing)
13. [Visualização e Monitoramento](#13-visualização-e-monitoramento)
14. [Stack Tecnológica](#14-stack-tecnológica)
15. [Estrutura de Pastas](#15-estrutura-de-pastas)
16. [Resultados Esperados](#16-resultados-esperados)
17. [Extensões Futuras](#17-extensões-futuras)

---

## 1. Visão Geral

Este projeto implementa um **sistema completo de detecção de fraudes financeiras** usando análise de séries temporais, estatística e machine learning — simulando o ambiente de dados usado por bancos e fintechs.

A arquitetura cobre todo o ciclo:

```
Dados reais (Kaggle) → Limpeza → EDA → Feature Engineering
→ Modelos de ML → Validação → Dashboard de monitoramento
```

**Principais entregas:**

| Entrega | Descrição |
|---|---|
| Pipeline ETL | Ingestão, limpeza e transformação automatizadas |
| EDA | Análise exploratória com visualizações temporais |
| Feature Engineering | Features comportamentais e temporais por cliente |
| Modelos de ML | Isolation Forest + Regressão Logística |
| Validação | Métricas, A/B Testing e análise estatística |
| Dashboard | KPIs de fraude para monitoramento contínuo |

---

## 2. Problema de Negócio

Fraudes financeiras geram bilhões em prejuízo anualmente. Sistemas baseados apenas em **regras fixas** têm limitações críticas:

- Dificuldade para detectar novos padrões de fraude
- Alta taxa de falsos positivos
- Baixa adaptabilidade a comportamentos emergentes

**Objetivo:** construir um modelo orientado a dados que detecte automaticamente padrões suspeitos em transações financeiras.

### Exemplos de comportamentos suspeitos detectados

| Comportamento | Sinal |
|---|---|
| Valor muito acima da média histórica | `amount / avg_historico > 3x` |
| Muitas transações em curto intervalo | `n_transacoes_24h > percentil 95` |
| Horário incomum (madrugada) | `hour between 0 and 5` |
| Mudança abrupta nos gastos | variação brusca no `amount_zscore` |
| Localização inconsistente | distância geográfica fora do padrão |

---

## 3. Datasets

O projeto suporta **3 datasets públicos reais** do Kaggle, com diferentes níveis de complexidade.

### 3.1 ULB Credit Card Fraud ⭐ Recomendado para começar

**Fonte:** [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Tamanho:** 143 MB | **Transações:** 284.807 | **Fraudes:** 492 (0,17%)

Transações reais de cartão de crédito de setembro de 2013 por portadores europeus. Features V1–V28 são resultado de transformação PCA por confidencialidade. Ideal para começar pela simplicidade: 1 arquivo CSV, sem joins, já limpo.

| Campo | Tipo | Descrição |
|---|---|---|
| `Time` | float | Segundos desde a 1ª transação |
| `V1` – `V28` | float | Componentes PCA (anonimizados) |
| `Amount` | float | Valor da transação (€) |
| `Class` | int | 0 = normal, 1 = fraude |

**Features geradas pelo pipeline:**

- `hour` — hora aproximada do dia (via time_seconds % 86400)
- `is_madrugada` — transação entre 0h e 5h
- `log_amount` — log1p do valor (corrige skewness)
- `amount_zscore` — desvio em relação à média global
- `is_high_value` — valor acima do percentil 95

---

### 3.2 IEEE-CIS Fraud Detection 🏆 Mais completo para portfólio

**Fonte:** [kaggle.com/competitions/ieee-fraud-detection](https://www.kaggle.com/competitions/ieee-fraud-detection)  
**Tamanho:** ~500 MB | **Transações:** 590.540 | **Fraudes:** 20.663 (3,5%)

Dataset de e-commerce real fornecido pela Vesta Corporation. Requer join entre dois arquivos (`transaction` + `identity`) por `TransactionID`. Features ricas incluindo dispositivo, e-mail, tipo de cartão e mais de 300 variáveis.

| Arquivo | Descrição |
|---|---|
| `train_transaction.csv` | Dados financeiros da transação |
| `train_identity.csv` | Dados de identidade e dispositivo |

**Campos principais após join:**

| Campo | Tipo | Descrição |
|---|---|---|
| `TransactionDT` | int | Timedelta em segundos (não timestamp real) |
| `TransactionAmt` | float | Valor da transação (USD) |
| `ProductCD` | string | Categoria do produto |
| `card1`–`card6` | mixed | Informações do cartão |
| `P_emaildomain` | string | Domínio de e-mail do comprador |
| `DeviceType` | string | Tipo de dispositivo |
| `V1`–`V339` | float | Features engineered pela Vesta |
| `isFraud` | int | 0 = normal, 1 = fraude |

> ⚠️ **Atenção:** é necessário aceitar os termos da competição em kaggle.com antes de baixar.

**Features geradas pelo pipeline:**

- `hour`, `day_of_week` — horário aproximado via timedelta
- `is_madrugada` — flag para transações 0h–5h
- `log_amount`, `amount_zscore` — transformações do valor
- `has_email`, `has_device` — flags de campos ausentes (sinal de fraude)
- `v_null_ratio` — proporção de campos V nulos por linha

---

### 3.3 Sparkov Simulated Transactions 🗺️ Schema mais rico

**Fonte:** [kaggle.com/datasets/kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)  
**Tamanho:** ~60 MB | **Transações:** ~150.000 | **Arquivos:** `fraudTrain.csv` + `fraudTest.csv`

Dataset simulado com o framework Sparkov. Possui timestamp real, merchant, category, coordenadas geográficas do cliente e do estabelecimento — o schema mais parecido com transações do dia a dia.

| Campo | Tipo | Descrição |
|---|---|---|
| `trans_date_trans_time` | datetime | Timestamp real da transação |
| `cc_num` | int | Número do cartão (anonimizado) |
| `merchant` | string | Nome do estabelecimento |
| `category` | string | Categoria (grocery, shopping, etc.) |
| `amt` | float | Valor da transação (USD) |
| `lat` / `long` | float | Localização do cliente |
| `merch_lat` / `merch_long` | float | Localização do estabelecimento |
| `is_fraud` | int | 0 = normal, 1 = fraude |

**Features geradas pelo pipeline:**

- `hour`, `day_of_week`, `month`, `is_weekend` — componentes temporais
- `is_madrugada` — flag para madrugada
- `log_amount` — transformação do valor
- `amount_vs_category_avg` — valor relativo à média da categoria
- `geo_distance` — distância aproximada cliente ↔ estabelecimento
- `seconds_since_last` — intervalo desde a última transação do mesmo cartão

### Comparativo dos datasets

| | ULB | IEEE-CIS | Sparkov |
|---|---|---|---|
| Dados reais | ✅ | ✅ | Simulado |
| Timestamp real | ❌ | ❌ | ✅ |
| Geolocalização | ❌ | ❌ | ✅ |
| Merchant / Categoria | ❌ | Parcial | ✅ |
| Join necessário | ❌ | ✅ | ❌ |
| Tamanho | 143 MB | ~500 MB | ~60 MB |
| Taxa de fraude | 0,17% | 3,5% | ~0,5% |
| Dificuldade | ⭐ Fácil | ⭐⭐⭐ Difícil | ⭐⭐ Médio |

---

## 4. Arquitetura do Projeto

```
┌────────────────────────────────────────────────────┐
│                  FONTES DE DADOS                   │
│     Kaggle API: ULB  /  IEEE-CIS  /  Sparkov       │
└──────────────────────┬─────────────────────────────┘
                       │ download_data.py
                       ▼
┌────────────────────────────────────────────────────┐
│               CAMADA BRONZE                        │
│          data/raw/{ulb, ieee, sparkov}/            │
│         CSVs brutos, sem modificações              │
└──────────────────────┬─────────────────────────────┘
                       │ preprocess.py
                       ▼
┌────────────────────────────────────────────────────┐
│               CAMADA SILVER                        │
│           data/processed/*_clean.csv               │
│    Limpeza + normalização + feature engineering    │
└──────────────────────┬─────────────────────────────┘
                       │ notebooks/
                       ▼
┌────────────────────────────────────────────────────┐
│          EDA  →  ML  →  VALIDAÇÃO                  │
│  Isolation Forest  /  Regressão Logística          │
│  Métricas  /  A/B Testing  /  Testes estatísticos  │
└──────────────────────┬─────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────┐
│           CAMADA GOLD — Dashboard                  │
│    KPIs diários  /  Score de risco por cliente     │
└────────────────────────────────────────────────────┘
```

---

## 5. Como Executar

### Pré-requisitos

```bash
pip install kaggle pandas numpy scikit-learn scipy matplotlib seaborn jupyter
```

### Configurar API do Kaggle (única vez)

1. Acesse [kaggle.com/settings](https://www.kaggle.com/settings)
2. Clique em **"Create New API Token"**
3. Salve o `kaggle.json` em:
   - **Windows:** `C:\Users\SEU_USUARIO\.kaggle\kaggle.json`
   - **Linux/macOS:** `~/.kaggle/kaggle.json`

### Criar estrutura de pastas

```bash
# Windows
mkdir data\raw data\processed src\data src\features src\models notebooks tests

# Linux/macOS
mkdir -p data/raw data/processed src/data src/features src/models notebooks tests
```

### Baixar os datasets

```bash
# ULB — recomendado para começar (143 MB)
python download_data.py --dataset 1

# IEEE-CIS — completo para portfólio (~500 MB)
python download_data.py --dataset 2

# Sparkov — schema rico com geolocalização (~60 MB)
python download_data.py --dataset 3

# Todos de uma vez
python download_data.py --all
```

### Executar o pipeline Bronze → Silver

```bash
# Processa o dataset escolhido
python preprocess.py --dataset ulb
python preprocess.py --dataset ieee
python preprocess.py --dataset sparkov
```

### Iniciar notebooks

```bash
jupyter notebook
# Abra: notebooks/01_eda.ipynb
```

---

## 6. Pipeline de Dados

### Bronze — Dados Brutos

Dados armazenados exatamente como baixados do Kaggle, sem nenhuma modificação.

```
data/raw/
├── ulb/
│   └── creditcard.csv
├── ieee/
│   ├── train_transaction.csv
│   └── train_identity.csv
└── sparkov/
    ├── fraudTrain.csv
    └── fraudTest.csv
```

### Silver — Dados Limpos e Enriquecidos

Após rodar `preprocess.py`, os dados passam por:

- Remoção de duplicatas e valores inválidos
- Correção de tipos de dados
- Extração de features temporais (hora, dia da semana, is_madrugada)
- Features comportamentais específicas de cada dataset
- Geração de relatório de qualidade (`quality_report.txt`)

```
data/processed/
├── ulb_clean.csv
├── ulb_quality_report.txt
├── ieee_clean.csv
├── ieee_quality_report.txt
├── sparkov_clean.csv
└── sparkov_quality_report.txt
```

---

## 7. Análise Exploratória — EDA

Objetivo: entender o comportamento temporal das transações antes da modelagem.

**Análises realizadas:**

- Distribuição de valores de transações (normal vs fraude)
- Frequência de transações por hora do dia
- Heatmap de taxa de fraude: hora × dia da semana
- Identificação de outliers via boxplot
- Sazonalidade e tendências ao longo do tempo

**Principais insights esperados:**

| Insight | Observação |
|---|---|
| Horário de pico de fraudes | Entre 0h e 5h (madrugada) |
| Valores suspeitos | Transações acima do percentil 95 têm maior taxa de fraude |
| Frequência anormal | Clientes com muitas transações em curto intervalo |
| Dataset ULB | Altamente desbalanceado (0,17% de fraudes) — exige SMOTE ou peso de classe |
| Dataset IEEE-CIS | Campos V ausentes correlacionados com fraude |
| Dataset Sparkov | Distância geográfica elevada é forte sinal de fraude |

---

## 8. Feature Engineering

Features criadas para capturar padrões temporais e comportamentais.

### Features comuns a todos os datasets

| Feature | Descrição |
|---|---|
| `hour` | Hora da transação (0–23) |
| `is_madrugada` | Flag: transação entre 0h e 5h |
| `log_amount` | Log do valor (corrige distribuição skewed) |
| `amount_zscore` | Desvio padrão em relação à média global |
| `is_high_value` | Valor acima do percentil 95 |

### Features exclusivas por dataset

**ULB:**
- Derivadas de `Time` (timedelta em segundos)

**IEEE-CIS:**
- `has_email` — comprador informou e-mail?
- `has_device` — transação tem dados de dispositivo?
- `v_null_ratio` — proporção de campos V nulos por transação

**Sparkov:**
- `amount_vs_category_avg` — valor relativo à média da categoria
- `geo_distance` — distância geográfica cliente ↔ estabelecimento
- `seconds_since_last` — intervalo desde a última transação do cartão
- `is_weekend`, `day_of_week`, `month`

---

## 9. Testes Estatísticos

Métodos para validar hipóteses antes da modelagem.

### Z-Score

Detecta valores muito distantes da média:

```
Z = (x - média) / desvio_padrão
|Z| > 3  →  outlier estatístico
```

### IQR — Interquartile Range

Define limites aceitáveis:

```
limite_inferior = Q1 - 1.5 × IQR
limite_superior = Q3 + 1.5 × IQR
```

### Testes de Hipótese

| Teste | Objetivo |
|---|---|
| Teste t | Comparar valor médio: fraudes vs normais |
| Qui-quadrado | Relação entre horário e ocorrência de fraude |
| ANOVA | Comparar médias entre categorias (Sparkov) |

---

## 10. Modelos de Machine Learning

### Isolation Forest — Detecção sem rótulos

Algoritmo principal. Não precisa de dados rotulados — ideal quando `is_fraud` não está disponível ou é escasso.

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=200,
    contamination=0.01,  # proporção esperada de fraudes
    random_state=42
)
model.fit(X_scaled)

predictions   = model.predict(X_scaled)   # 1 = normal, -1 = anomalia
anomaly_score = -model.score_samples(X_scaled)  # quanto maior, mais suspeito
```

**Configuração recomendada por dataset:**

| Dataset | `contamination` | Justificativa |
|---|---|---|
| ULB | 0.002 | 0,17% de fraudes reais |
| IEEE-CIS | 0.035 | 3,5% de fraudes reais |
| Sparkov | 0.005 | ~0,5% de fraudes simuladas |

### Regressão Logística — Classificação com rótulos

Usado quando `is_fraud` está disponível para treinamento supervisionado.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        class_weight="balanced",  # trata desbalanceamento
        max_iter=1000,
        random_state=42
    ))
])
pipeline.fit(X_train, y_train)
```

> **Nota:** Para o dataset ULB (0,17% de fraudes), considere usar `class_weight="balanced"` ou SMOTE para lidar com o desbalanceamento extremo.

---

## 11. Validação e Métricas

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

metrics = {
    "precision": precision_score(y_true, y_pred),
    "recall":    recall_score(y_true, y_pred),
    "f1_score":  f1_score(y_true, y_pred),
    "roc_auc":   roc_auc_score(y_true, y_proba),
}
```

### Por que não usar Accuracy?

Nos datasets de fraude, a acurácia é enganosa: um modelo que sempre prevê "normal" teria 99,83% de acurácia no ULB. **Use F1-Score e ROC-AUC como métricas principais.**

### Metas por dataset

| Métrica | ULB | IEEE-CIS | Sparkov |
|---|---|---|---|
| F1-Score | > 0.80 | > 0.75 | > 0.82 |
| ROC-AUC | > 0.95 | > 0.88 | > 0.90 |
| Recall | > 0.80 | > 0.70 | > 0.80 |

---

## 12. A/B Testing

Avalia se o modelo de ML supera o sistema baseado em regras fixas.

```python
from scipy.stats import chi2_contingency

# Grupo A: regras fixas  |  Grupo B: modelo ML
tabela = [[700, 300], [900, 100]]  # [deteccoes, falhas]
chi2, p_value, _, _ = chi2_contingency(tabela)
# p-value < 0.05 → diferença estatisticamente significativa
```

**Resultado esperado:**

| Métrica | Sistema Antigo (Regras) | Novo Modelo (ML) |
|---|---|---|
| Taxa de detecção | 70% | 90% |
| Falsos positivos | 15% | 8% |
| F1-Score | 0.72 | 0.91 |

---

## 13. Visualização e Monitoramento

**KPIs monitorados:**

| KPI | Descrição |
|---|---|
| Fraudes detectadas / dia | Contagem diária de anomalias |
| Taxa de fraude (%) | Proporção sobre total de transações |
| Valor bloqueado | Volume financeiro protegido |
| Clientes em risco | Usuários com score de anomalia elevado |
| Heatmap hora × dia | Mapa de calor da concentração de fraudes |

**Ferramentas sugeridas:**

- Power BI / Looker Studio — dashboards executivos
- Metabase — exploração self-service
- Plotly / Seaborn — visualizações nos notebooks

---

## 14. Stack Tecnológica

| Categoria | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Dados | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Análise estatística | SciPy, Statsmodels |
| Visualização | Matplotlib, Seaborn, Plotly |
| Download de dados | Kaggle API (`kaggle`) |
| BI e dashboards | Power BI / Looker Studio / Metabase |
| Notebooks | Jupyter |
| Qualidade de código | flake8, pytest |
| Versionamento | Git + GitHub |

---

## 15. Estrutura de Pastas

```
fraud-detection/
│
├── data/
│   ├── raw/
│   │   ├── ulb/
│   │   │   └── creditcard.csv
│   │   ├── ieee/
│   │   │   ├── train_transaction.csv
│   │   │   └── train_identity.csv
│   │   └── sparkov/
│   │       ├── fraudTrain.csv
│   │       └── fraudTest.csv
│   └── processed/
│       ├── ulb_clean.csv
│       ├── ieee_clean.csv
│       ├── sparkov_clean.csv
│       └── *_quality_report.txt
│
├── notebooks/
│   ├── 01_eda.ipynb               # Análise exploratória
│   ├── 02_feature_engineering.ipynb
│   ├── 03_statistical_analysis.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_ab_testing.ipynb
│   └── 06_results_visualization.ipynb
│
├── src/
│   ├── data/
│   │   ├──  download_data.py               # Download via Kaggle API
│   │   └── preprocess.py                   # Pipeline Bronze -> Silver
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── statistical_analysis.py
│   └── models/
│       ├── isolation_forest.py
│       ├── logistic_regression.py
│       └── evaluate.py
│
├── models/
│   ├── isolation_forest_ulb.pkl
│   ├── isolation_forest_ieee.pkl
│   └── scaler.pkl
│
├── reports/
│   └── figures/
│
├── tests/
│   ├── test_preprocess.py
│   └── test_models.py
│
├── requirements.txt
└── README.md
```

---

## 16. Resultados Esperados

| Métrica | Meta | ULB | IEEE-CIS | Sparkov |
|---|---|---|---|---|
| F1-Score | > 0.80 | — | — | — |
| ROC-AUC | > 0.90 | — | — | — |
| Recall | > 0.80 | — | — | — |
| Falsos positivos | < 10% | — | — | — |
| Cobertura de testes | > 80% | — | — | — |

> Os campos marcados com `—` serão preenchidos após execução dos modelos.

---

## 17. Extensões Futuras

- [ ] LSTM para análise avançada de séries temporais
- [ ] Autoencoders para detecção de anomalias em alta dimensão
- [ ] SMOTE para tratamento de desbalanceamento extremo (ULB)
- [ ] Detecção em tempo real com Apache Kafka
- [ ] Graph Analytics para identificar redes de fraude
- [ ] API REST com FastAPI para servir predições
- [ ] Orquestração com Apache Airflow
- [ ] Deploy no Google Cloud (BigQuery + Cloud Run)

---

## Aplicações Reais

Este tipo de sistema é amplamente utilizado em bancos digitais, fintechs, empresas de pagamento, e-commerce, telecom e plataformas de marketplace.

---

## Licença

Distribuído sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.

---

## Autor

Desenvolvido como projeto de portfólio em Engenharia de Dados e Ciência de Dados.
