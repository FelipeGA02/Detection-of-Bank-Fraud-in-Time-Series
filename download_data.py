"""
Baixa os datasets de fraud detection do Kaggle e organiza nas pastas do projeto.

Datasets disponiveis:
  1. ULB Credit Card Fraud     — leve, ideal para comecar (143 MB)
  2. IEEE-CIS Fraud Detection  — completo, ideal para portfolio (mais pesado)
  3. Sparkov Simulated         — colunas ricas (merchant, location, category)

Como configurar o Kaggle API (fazer UMA vez):
  1. Acesse https://www.kaggle.com/settings
  2. Clique em "Create New API Token"
  3. Salve o kaggle.json baixado em:
       Windows: C:\\Users\\SEU_USUARIO\\.kaggle\\kaggle.json
  4. Pronto! Execute este script.

Uso:
    python download_data.py              # baixa o ULB (recomendado para comecar)
    python download_data.py --dataset 2  # baixa o IEEE-CIS
    python download_data.py --dataset 3  # baixa o Sparkov
    python download_data.py --all        # baixa todos
"""

import argparse
import os
import shutil
import sys
import zipfile

# ── Datasets disponiveis ──────────────────────────────────────────────────────

DATASETS = {
    1: {
        "name": "ULB Credit Card Fraud",
        "slug": "mlg-ulb/creditcardfraud",
        "type": "dataset",
        "size": "143 MB",
        "description": "284k transacoes reais, 492 fraudes (0.17%). Features anonimizadas com PCA.",
        "files": ["creditcard.csv"],
        "output_dir": os.path.join("data", "raw", "ulb"),
    },
    2: {
        "name": "IEEE-CIS Fraud Detection",
        "slug": "ieee-fraud-detection",
        "type": "competition",
        "size": "~500 MB",
        "description": "590k transacoes e-commerce, 3.5% fraudes. Features ricas com identity + transaction.",
        "files": [
            "train_transaction.csv",
            "train_identity.csv",
            "test_transaction.csv",
            "test_identity.csv",
        ],
        "output_dir": os.path.join("data", "raw", "ieee"),
    },
    3: {
        "name": "Sparkov Simulated Transactions",
        "slug": "kartik2112/fraud-detection",
        "type": "dataset",
        "size": "~60 MB",
        "description": "150k transacoes simuladas com merchant, category, location — schema rico.",
        "files": ["fraudTrain.csv", "fraudTest.csv"],
        "output_dir": os.path.join("data", "raw", "sparkov"),
    },
}


# ── Autenticacao ──────────────────────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    """Verifica se o kaggle.json esta configurado corretamente."""
    kaggle_dir  = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")

    if os.path.exists(kaggle_json):
        print(f"Credenciais encontradas em: {kaggle_json}")
        return True

    print("\nERRO: kaggle.json nao encontrado!")
    print("=" * 55)
    print("Configure a API do Kaggle em 3 passos:")
    print()
    print("  1. Acesse: https://www.kaggle.com/settings")
    print("  2. Clique em 'Create New API Token'")
    print(f"  3. Mova o kaggle.json para: {kaggle_json}")
    print()
    print("Se a pasta .kaggle nao existir, crie manualmente:")
    print(f"     mkdir {kaggle_dir}")
    print("=" * 55)
    return False


def get_kaggle_api():
    """Retorna a instancia autenticada da API do Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except ImportError:
        print("\nERRO: pacote 'kaggle' nao instalado.")
        print("Execute: pip install kaggle")
        sys.exit(1)
    except Exception as e:
        print(f"\nERRO ao autenticar: {e}")
        print("Verifique se o kaggle.json esta no lugar certo.")
        sys.exit(1)


# ── Download ──────────────────────────────────────────────────────────────────

def download_dataset(api, dataset_info: dict) -> bool:
    """Baixa e extrai um dataset para a pasta de saida."""
    name       = dataset_info["name"]
    slug       = dataset_info["slug"]
    dtype      = dataset_info["type"]
    output_dir = dataset_info["output_dir"]

    print(f"\nBaixando: {name}")
    print(f"  Tamanho estimado : {dataset_info['size']}")
    print(f"  Destino          : {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        if dtype == "dataset":
            print("  Iniciando download (aguarde)...")
            api.dataset_download_files(
                dataset=slug,
                path=tmp_dir,
                unzip=True,
                quiet=False,
            )
        elif dtype == "competition":
            print("  Iniciando download da competicao (aguarde)...")
            print("  ATENCAO: voce precisa aceitar os termos da competicao em:")
            print(f"  https://www.kaggle.com/competitions/{slug}/rules")
            api.competition_download_files(
                competition=slug,
                path=tmp_dir,
                quiet=False,
            )
            _extract_zips(tmp_dir)

        _move_csv_files(tmp_dir, output_dir)

        shutil.rmtree(tmp_dir, ignore_errors=True)

        files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]
        print(f"\n  Arquivos disponiveis em '{output_dir}':")
        for f in files:
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            print(f"    {f}  ({size_mb:.1f} MB)")

        print(f"  Download concluido!")
        return True

    except Exception as e:
        print(f"\n  ERRO no download: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False


def _extract_zips(directory: str) -> None:
    """Extrai todos os arquivos .zip encontrados no diretorio."""
    for fname in os.listdir(directory):
        if fname.endswith(".zip"):
            zip_path = os.path.join(directory, fname)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(directory)
            os.remove(zip_path)


def _move_csv_files(src: str, dst: str) -> None:
    """Move todos os CSVs da pasta src para dst, incluindo subpastas."""
    for root, _, files in os.walk(src):
        for fname in files:
            if fname.endswith(".csv"):
                src_path = os.path.join(root, fname)
                dst_path = os.path.join(dst, fname)
                shutil.move(src_path, dst_path)


# ── Listagem ──────────────────────────────────────────────────────────────────

def list_datasets() -> None:
    """Exibe os datasets disponiveis."""
    print("\nDatasets disponiveis:")
    print("=" * 65)
    for key, info in DATASETS.items():
        print(f"  [{key}] {info['name']}")
        print(f"       Tamanho    : {info['size']}")
        print(f"       Descricao  : {info['description']}")
        print(f"       Destino    : {info['output_dir']}")
        print()
    print("Recomendacao para comecar: dataset 1 (ULB) — mais leve e simples.")
    print("Para portfolio completo  : dataset 1 + 2.")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Download de datasets de fraud detection do Kaggle"
)
parser.add_argument(
    "--dataset", type=int, choices=[1, 2, 3], default=1,
    help="Dataset a baixar: 1=ULB, 2=IEEE-CIS, 3=Sparkov (padrao: 1)"
)
parser.add_argument(
    "--all", action="store_true",
    help="Baixa todos os datasets"
)
parser.add_argument(
    "--list", action="store_true",
    help="Lista os datasets disponiveis e sai"
)
args = parser.parse_args()

if args.list:
    list_datasets()

print("=" * 55)
print("  KAGGLE DATASET DOWNLOADER")
print("  Fraud Detection Platform")
print("=" * 55)

list_datasets()

if not check_kaggle_credentials():
    sys.exit(1)

api = get_kaggle_api()
print("Autenticado com sucesso!\n")

targets = list(DATASETS.keys()) if args.all else [args.dataset]

results = {}
for key in targets:
    results[key] = download_dataset(api, DATASETS[key])

print("\n" + "=" * 55)
print("RESUMO DO DOWNLOAD")
print("=" * 55)
for key, success in results.items():
    status = "OK" if success else "FALHOU"
    print(f"  [{status}] {DATASETS[key]['name']}")

print("\nProximo passo:")
print("  python preprocess.py --dataset ulb")
print("  (ou: ieee | sparkov conforme o dataset baixado)")
print("=" * 55)