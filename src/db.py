"""Conexao com o PostgreSQL da camada de dados (2a EC2 - churn-db-host).

As credenciais NUNCA ficam no codigo (o repositorio e publico). Elas sao lidas
de variaveis de ambiente, que podem vir de um arquivo .env local (ignorado pelo
Git) ou do `docker run --env-file` na EC2 da aplicacao.

Variaveis:
    DB_HOST      IP do host do banco (privado 172.31.x.x dentro da VPC; publico fora dela)
    DB_PORT      porta do Postgres (padrao 5432)
    DB_NAME      nome do banco (padrao churn_db)
    DB_USER      usuario (padrao postgres)
    DB_PASSWORD  senha (obrigatoria, sem valor padrao)
"""
import os
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Carrega um .env simples (CHAVE=valor) sem sobrescrever o ambiente."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def db_settings() -> dict:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "churn_db"),
        "username": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD"),
    }


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    cfg = db_settings()
    if not cfg["password"]:
        raise RuntimeError("DB_PASSWORD nao definida. Configure o .env (veja .env.example).")
    url = URL.create("postgresql+psycopg2", **cfg)
    # pool_pre_ping descarta conexoes mortas (ex.: banco reiniciado);
    # connect_timeout evita que a API trave se o Security Group bloquear a porta.
    return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
