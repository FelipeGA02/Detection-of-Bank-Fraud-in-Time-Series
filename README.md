# End-to-End Data Platform for Fraud Detection (Data Engineering + BI + ML)

## 1. Visão Geral

Este projeto implementa uma arquitetura completa de dados (end-to-end), desde ingestão até consumo analítico, para detecção de fraudes financeiras.

A solução inclui:

- Pipelines de dados orquestrados (ETL/ELT)
- Modelagem de dados em Data Warehouse (camadas bronze, silver, gold)
- Processamento e transformação de dados em escala
- Treinamento de modelos de machine learning
- Exposição de dados para análise e BI (dashboards)

O objetivo é simular um ambiente real de dados utilizado por empresas data-driven.**sistemas antifraude utilizados por bancos e fintechs**, onde transações são analisadas continuamente para identificar riscos.

---

# 2. Problema de Negócio

Fraudes financeiras geram bilhões de prejuízo todos os anos. Sistemas tradicionais baseados apenas em regras fixas possuem limitações, como:

- dificuldade para detectar novos padrões de fraude
- alta taxa de falsos positivos
- baixa adaptabilidade

O objetivo deste projeto é construir um **modelo baseado em dados que detecte automaticamente padrões suspeitos**.

### Exemplos de comportamentos suspeitos

- Transações muito acima da média do cliente
- Muitas transações em curto período
- Compras em horários incomuns
- Mudança abrupta no padrão de gastos

---

# 3. Dataset

O projeto pode utilizar datasets públicos de fraude financeira.

Exemplo:

**Credit Card Fraud Detection Dataset**

Estrutura típica:

| Campo | Descrição |
|-----|-----|
| timestamp | momento da transação |
| user_id | identificador do cliente |
| amount | valor da transação |
| merchant | estabelecimento |
| transaction_type | tipo de transação |
| location | localização |
| is_fraud | indicador de fraude |

---

# 4 Arquitetura do Projeto

## 4.1 Arquitetura de Dados

Pipeline de Dados

O pipeline segue uma arquitetura moderna em camadas (bronze, silver, gold), integrando engenharia de dados, análise e machine learning:

1. Ingestão de dados (APIs, arquivos, fontes externas)
2. Armazenamento em camada raw (bronze) — dados brutos e imutáveis
3. Limpeza e pré-processamento (silver) — tratamento, normalização e validação
4. Análise exploratória (EDA) — entendimento inicial dos dados e identificação de padrões
5. Feature engineering — criação de variáveis temporais e comportamentais
6. Testes estatísticos — validação de hipóteses e detecção de outliers (Z-score, IQR)
7. Treinamento de modelos — algoritmos de detecção de anomalias e regressão
8. Validação — avaliação com métricas (Precision, Recall, F1-score) e A/B testing
9. Detecção de anomalias — identificação de transações suspeitas
10. Geração de métricas e outputs analíticos
11. Disponibilização dos dados (gold) — tabelas agregadas e otimizadas para consumo
12. Visualização e monitoramento — dashboards de BI para acompanhamento contínuo

## 4.2 Orquestração de Dados (Apache Airflow)

Os pipelines são orquestrados utilizando Apache Airflow, com DAGs responsáveis por:

- Ingestão de dados
- Processamento e limpeza
- Feature engineering
- Treinamento de modelos
- Geração de outputs analíticos

Cada etapa é desacoplada e executada de forma automatizada.

##4.2 Data Warehouse (BigQuery)

Os dados são armazenados e processados utilizando BigQuery, permitindo:

- Execução de queries em larga escala
- Processamento analítico eficiente
- Integração com ferramentas de BI

Exemplos de tabelas:

- raw_transactions
- processed_transactions
- fraud_predictions

##4.3 CI/CD

O projeto utiliza GitHub Actions para:

- Execução de testes automatizados
- Validação de código (lint)
- Garantia de qualidade do pipeline

Isso simula um ambiente de produção com boas práticas de engenharia.

# 5. Análise Exploratória de Dados (EDA)

Objetivo: entender o comportamento temporal das transações.

Principais análises:

- distribuição de valores de transações
- frequência de compras por usuário
- comportamento por horário
- sazonalidade
- identificação inicial de outliers

Visualizações utilizadas:

- histogramas
- boxplots
- séries temporais
- heatmaps de horários

Exemplo de insights:

- maioria das transações ocorre entre 8h e 22h
- valores acima de determinado limite são raros
- fraudes costumam ocorrer em horários noturnos

---

# 6. Feature Engineering

Criação de variáveis que capturam padrões temporais.

### Features temporais

- média de gasto nas últimas 24h
- média de gasto nos últimos 7 dias
- número de transações nas últimas 24h
- intervalo entre transações
- horário da transação

### Features comportamentais

- razão entre valor atual e média histórica do cliente
- desvio padrão de gastos
- frequência de transações

### Exemplo

valor_relativo = valor_transacao / media_historica_cliente


Essas variáveis ajudam o modelo a entender **mudanças abruptas de comportamento**.

---

# 7. Uso de Estatística

Métodos estatísticos são utilizados para identificar anomalias e validar hipóteses.

### Z-score

Detecta valores muito distantes da média.

Z = (x - média) / desvio_padrão

Valores com:

|Z| > 3

podem ser considerados outliers.

---

### IQR (Interquartile Range)

Define limites aceitáveis:

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

---

### Testes de Hipótese

Utilizados para validar relações entre variáveis.

Exemplos:

- teste t para comparar médias
- teste qui-quadrado para independência
- análise de variância (ANOVA)

---

# 8. Modelos de Machine Learning

O projeto utiliza modelos de **detecção de anomalias**.

## Isolation Forest

Algoritmo muito usado para detectar fraudes.

Características:

- eficiente em grandes datasets
- não precisa de dados rotulados
- identifica pontos isolados no espaço de dados

Exemplo em Python:

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(contamination=0.01)
model.fit(X)

predictions = model.predict(X)
```
Saída:
1  = normal
-1 = anomalia

## 8. Regressão

Modelos de regressão podem ser usados para prever **valores esperados** em uma série temporal ou em variáveis relacionadas às transações.

### Exemplos

- **Regressão Linear**  
  Utilizada para prever valores contínuos, como o valor esperado de uma transação com base no comportamento histórico do cliente.

- **Regressão Logística**  
  Utilizada para **classificação de fraude**, estimando a probabilidade de uma transação ser fraudulenta.

A diferença entre o valor previsto e o valor observado pode indicar **anomalias ou comportamentos suspeitos**.

---

## 9. A/B Testing

**A/B Testing** é utilizado para avaliar se o novo modelo de detecção de fraudes apresenta desempenho superior ao sistema existente.

### Experimento

**Grupo A**  
Sistema antigo baseado em regras fixas.

**Grupo B**  
Novo sistema baseado em **machine learning**.

### Métricas avaliadas

- taxa de detecção de fraudes
- taxa de falsos positivos
- tempo de resposta do sistema
- impacto financeiro

### Exemplo de resultado

| Métrica | Sistema Antigo | Novo Modelo |
|--------|--------|--------|
| Detecção de fraude | 70% | 90% |
| Falsos positivos | 15% | 8% |

---

## 10. Avaliação do Modelo

Principais métricas utilizadas:

### Precision

Proporção de fraudes detectadas que **realmente eram fraudes**.

### Recall

Proporção de fraudes reais que foram **corretamente identificadas pelo modelo**.

### F1 Score

Média harmônica entre **precision** e **recall**, utilizada para balancear ambas as métricas.

F1 = 2 * (precision * recall) / (precision + recall)


---

## 11. Visualização e Monitoramento

Criação de **dashboards analíticos** para monitorar o desempenho do sistema de detecção de fraudes.

### Indicadores monitorados

- fraudes detectadas por dia
- valor financeiro bloqueado
- clientes com comportamento suspeito
- distribuição de anomalias ao longo do tempo

### Ferramentas possíveis

- Power BI
- Tableau
- Metabase

---

## 12. Stack Tecnológica

### Linguagem

- Python

### Manipulação de dados

- Pandas
- NumPy

### Machine Learning

- Scikit-learn

### Análise estatística

- SciPy
- Statsmodels

### Visualização de dados

- Matplotlib
- Seaborn
- Plotly

### Engenharia de dados (opcional)

- SQL
- Apache Spark
- Apache Kafka

### Deploy e produção (opcional)

- Docker
- Airflow
- FastAPI

---

## 13. Resultados Esperados

Com a implementação do modelo, espera-se:

- aumento da taxa de detecção de fraudes
- redução de falsos positivos
- identificação automática de comportamentos anormais
- melhoria na segurança financeira

---

## 14. Possíveis Extensões do Projeto

Melhorias futuras podem incluir:

- uso de **redes neurais LSTM** para análise de séries temporais
- **autoencoders** para detecção avançada de anomalias
- detecção de fraude em **tempo real**
- integração com **streaming de dados**
- uso de **graph analytics** para identificar redes de fraude

---

## 15. Aplicações Reais

Esse tipo de sistema é amplamente utilizado em:

- bancos digitais
- fintechs
- empresas de pagamento
- e-commerce
- telecom
- plataformas de marketplace

---

## 16. Estrutura de Pastas

- fraud-detection-timeseries/
  - README.md
  - requirements.txt
  - .gitignore

  - data/
    - raw/
      - dataset_original.csv
    - processed/
      - dataset_clean.csv
    - external/

  - notebooks/
    - 01_data_exploration.ipynb
    - 02_feature_engineering.ipynb
    - 03_statistical_analysis.ipynb
    - 04_model_training.ipynb
    - 05_ab_testing.ipynb
    - 06_results_visualization.ipynb

  - src/
    - data/
      - load_data.py
      - preprocess.py
    - features/
      - feature_engineering.py
    - models/
      - isolation_forest.py
      - regression_model.py
      - evaluate.py
    - visualization/
      - plots.py

  - reports/
    - figures/
      - anomaly_detection.png
      - time_series_plot.png
      - model_metrics.png
    - final_report.md

  - dashboards/
    - fraud_dashboard.pbix

  - tests/
    - test_features.py
    - test_models.py
    - test_data_processing.py

### Explicação das Pastas

### `data/`

Responsável pelo armazenamento de **todos os dados do projeto**, segmentados por estágio de maturação.

- **raw**: Dados originais, sem modificações (**imutáveis**).
- **processed**: Dados limpos, transformados e prontos para modelagem.
- **external**: Datasets de terceiros ou fontes complementares.

---

### `notebooks/`

Contém os **Jupyter Notebooks organizados numericamente** para refletir o fluxo lógico do projeto.

Etapas:

- **Exploração**: Análise inicial e entendimento dos dados.
- **Feature Engineering**: Criação de variáveis e tratamento de séries temporais.
- **Análise Estatística**: Validação de hipóteses e distribuições.
- **Treinamento**: Experimentação de algoritmos (ex: Isolation Forest).
- **A/B Testing**: Validação do impacto das soluções.
- **Visualização**: Consolidação e análise dos resultados finais.

---

### `src/` (Source Code)

Armazena o **código Python reutilizável e modularizado**. Separar a lógica dos notebooks em arquivos `.py` facilita a **manutenção, reprodutibilidade e deploy**.

Subpastas:

- **data**: Scripts de ETL (extração, carga e limpeza de dados).
- **features**: Lógica de cálculo de variáveis e criação de colunas.
- **models**: Scripts de treinamento, arquitetura dos modelos e cálculo de métricas.
- **visualization**: Funções para geração de gráficos padronizados.

---

### `reports/`

Destinado aos **entregáveis finais do projeto** e evidências visuais.

- **figures**: Exportação de gráficos (PNG/JPG) para apresentações e relatórios.
- **final_report**: Documentação detalhada com análises e conclusões do projeto.

---

### `dashboards/`

Armazena arquivos de **ferramentas de Business Intelligence** utilizadas para visualização interativa e monitoramento de indicadores.

Exemplos:

- Power BI
- Tableau
- Metabase

Utilizado para acompanhar **KPIs de fraude**, padrões temporais e desempenho do modelo.

---

### `tests/`

Responsável por garantir a **integridade do pipeline de dados** através de testes unitários.

Testes comuns:

- Verificação se o **cálculo das features está correto**.
- Validação se o **formato de entrada e saída dos dados segue o esquema esperado**.
- Testes de **funcionamento e performance mínima dos modelos**.
