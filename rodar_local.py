"""Roda a arquitetura completa do projeto na sua maquina, sem AWS, com um comando.

    python rodar_local.py                # Docker: PostgreSQL + app (FastAPI + Streamlit), como na AWS
    python rodar_local.py --sem-docker   # sem Docker: SQLite no lugar do PostgreSQL, API e painel como processos locais
    python rodar_local.py --parar        # derruba os containers do modo Docker

Etapas (cada uma so roda se for necessaria):
    1. Dado      : `dvc pull` do remote `repositorio` (pasta dvc-publico/ do proprio repo) + conferencia do MD5
    2. ML        : `dvc repro` -> grid de 9 runs no MLflow, @champion no Model Registry, export para models/champion/
    3. Servico   : sobe o banco com a tabela `customers` (CSV -> banco) e a app que serve o @champion

Equivalencia com a AWS (README, secao 2):
    EC2 churn-db-host  (PostgreSQL em Docker)        -> servico `db` do docker-compose.local.yml
    EC2 churn-app-host (imagem do ECR + Supervisord) -> servico `app`, construido com o mesmo Dockerfile
    IAM Role / ECR                                   -> nao se aplica localmente (a imagem e construida na hora)

Opcoes:
    --pular-treino   usa o models/champion/ que ja existe (nao roda o dvc repro)
    --retreinar      forca o grid e a promocao de novo (dvc repro --force)
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parent
CSV = RAIZ / "data" / "raw" / "churn.csv"
PONTEIRO = RAIZ / "data" / "raw" / "churn.csv.dvc"
CHAMPION = RAIZ / "models" / "champion"
COMPOSE = RAIZ / "docker-compose.local.yml"
SQLITE = RAIZ / "data" / "churn_local.db"
API, PAINEL = "http://127.0.0.1:8000", "http://127.0.0.1:8501"


def etapa(texto: str) -> None:
    print(f"\n=== {texto} ===", flush=True)


def rodar(cmd: list, **kw) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=RAIZ, check=True, **kw)


def tem_dvc() -> bool:
    return importlib.util.find_spec("dvc") is not None


def md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def garantir_dado() -> None:
    etapa("1/3 Dado (DVC)")
    esperado = yaml.safe_load(PONTEIRO.read_text(encoding="utf-8"))["outs"][0]["md5"]
    if not CSV.exists():
        if tem_dvc():
            rodar([sys.executable, "-m", "dvc", "pull", str(PONTEIRO.relative_to(RAIZ))])
        else:
            # sem DVC instalado: o remote `repositorio` e uma pasta do proprio repo (layout files/md5/xx/resto)
            objeto = RAIZ / "dvc-publico" / "files" / "md5" / esperado[:2] / esperado[2:]
            print(f"dvc pull indisponivel; copiando direto de {objeto.relative_to(RAIZ)}")
            CSV.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(objeto, CSV)
    atual = md5(CSV)
    if atual != esperado:
        sys.exit(f"churn.csv com MD5 {atual}, mas o ponteiro DVC espera {esperado}. Rode `dvc checkout`.")
    print(f"churn.csv ok (md5 {atual}, mesmo hash do ponteiro {PONTEIRO.name})")


def garantir_modelo(pular: bool, forcar: bool) -> None:
    etapa("2/3 ML Pipeline (dvc repro: grid MLflow -> @champion -> models/champion/)")
    if pular:
        if not (CHAMPION / "MLmodel").exists():
            sys.exit("--pular-treino, mas models/champion/ nao existe. Rode sem essa opcao.")
        print("usando o models/champion/ existente")
        return
    if tem_dvc():
        rodar([sys.executable, "-m", "dvc", "repro"] + (["--force"] if forcar else []))
    else:
        print("DVC nao instalado; rodando as etapas do dvc.yaml diretamente")
        for modulo in ("src.experiments", "src.register_model", "src.export_champion"):
            rodar([sys.executable, "-m", modulo])


def esperar_api(segundos: int = 120) -> dict:
    import json
    fim = time.time() + segundos
    while time.time() < fim:
        try:
            with urllib.request.urlopen(f"{API}/health", timeout=3) as r:
                saude = json.loads(r.read())
            if saude.get("database", {}).get("status") == "connected":
                return saude
        except OSError:
            pass
        time.sleep(2)
    sys.exit(f"A API nao respondeu em {segundos}s. Veja os logs (docker compose -f {COMPOSE.name} logs).")


def resumo(saude: dict) -> None:
    m, db = saude["model"], saude["database"]
    print(f"\nModelo: {m['model_name']} v{m['version']} {m['alias']} ({m['run_name']}), "
          f"acuracia {m['test_accuracy']:.4f}, ROC-AUC {m.get('test_roc_auc') or float('nan'):.4f}")
    print(f"Banco : {db.get('engine', 'postgresql')} com {db.get('records')} clientes")
    print(f"\nPainel : {PAINEL}\nAPI    : {API}/docs\nTeste  : {API}/results/7015-DBER")


def modo_docker() -> None:
    etapa("3/3 Servico (docker compose: PostgreSQL + app)")
    if shutil.which("docker") is None:
        sys.exit("Docker nao encontrado. Instale o Docker Desktop ou use `python rodar_local.py --sem-docker`.")
    rodar(["docker", "compose", "-f", COMPOSE.name, "up", "--build", "-d"])
    resumo(esperar_api())
    print(f"\nPara parar: python rodar_local.py --parar")


def carregar_sqlite() -> None:
    """CSV -> SQLite com o mesmo nome de tabela e de colunas do PostgreSQL (infra/db/init_customers.sql)."""
    import pandas as pd
    SQLITE.unlink(missing_ok=True)
    df = pd.read_csv(CSV, dtype={"TotalCharges": str})
    with sqlite3.connect(SQLITE) as con:
        df.to_sql("customers", con, index=False)
        con.execute('CREATE UNIQUE INDEX ix_customer ON customers ("customerID")')
    print(f"{len(df)} clientes carregados em {SQLITE.relative_to(RAIZ)} (tabela customers)")


def modo_sem_docker() -> None:
    etapa("3/3 Servico (SQLite + uvicorn + streamlit, sem Docker)")
    carregar_sqlite()
    env = dict(os.environ, DATABASE_URL=f"sqlite:///{SQLITE.as_posix()}", API_URL=API, PYTHONUNBUFFERED="1")
    processos = [
        subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api_v2:app", "--port", "8000"], cwd=RAIZ, env=env),
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app_churn_v2.py", "--server.port", "8501",
                          "--server.headless", "true"], cwd=RAIZ, env=env),
    ]
    try:
        resumo(esperar_api())
        print("\nCtrl+C para parar.")
        while all(p.poll() is None for p in processos):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processos:
            p.terminate()


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Arquitetura completa do projeto, local e sem AWS")
    ap.add_argument("--sem-docker", action="store_true", help="SQLite + processos locais no lugar dos containers")
    ap.add_argument("--parar", action="store_true", help="derruba os containers do modo Docker")
    ap.add_argument("--pular-treino", action="store_true", help="usa o models/champion/ existente")
    ap.add_argument("--retreinar", action="store_true", help="forca o grid e a promocao (dvc repro --force)")
    a = ap.parse_args()
    os.chdir(RAIZ)
    if a.parar:
        rodar(["docker", "compose", "-f", COMPOSE.name, "down"])
        return
    try:
        garantir_dado()
        garantir_modelo(a.pular_treino, a.retreinar)
        modo_sem_docker() if a.sem_docker else modo_docker()
    except subprocess.CalledProcessError as erro:
        sys.exit(f"\nFalhou: {' '.join(map(str, erro.cmd))} (codigo {erro.returncode}). Veja a saida acima.")


if __name__ == "__main__":
    main()
