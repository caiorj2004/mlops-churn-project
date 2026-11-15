# Infraestrutura e Implantação (MLOps Churn App)

Este diretório reúne os scripts e artefatos para reproduzir a arquitetura do projeto na AWS (região `us-east-2`).

| Arquivo | Função |
| --- | --- |
| `deploy_ecr.ps1` | Build local da imagem e push para o Amazon ECR (executado no Windows) |
| `iam_trust_ec2.json` | Trust policy da role `EC2-ECR-ReadRole` (EC2 lê o ECR sem chaves fixas) |
| `setup_ec2.sh` | Deploy da **camada de aplicação** na EC2 `churn-app-host` |
| `setup_db_ec2.sh` | Provisiona a **camada de dados** na EC2 `churn-db-host` (Postgres em Docker + carga do CSV) |
| `db/init_customers.sql` | DDL da tabela `customers` + `\copy` do `churn.csv` (CSV → postgres) |

---

## 1. Evolução da arquitetura

| Etapa | App | Onde ficam os dados dos clientes | Infra |
| --- | --- | --- | --- |
| 1 | app-churn-v1 | `data/batch_predictions.parquet` embutido na imagem (só consulta, sem inferência) | 1 EC2 |
| 2 | app-churn-v2 | Parquet (consulta) + inferência online via `POST /predict` | 1 EC2 |
| **3** | **app-churn-v2** | **`churn.csv` bruto no PostgreSQL em Docker numa 2ª EC2**; a API faz `SELECT` pela rede | **2 EC2** |

A etapa 3 não cria uma versão nova do app: as funcionalidades da v2 continuam as mesmas (consulta por código + nova inferência), muda só a infraestrutura.

## 2. Topologia em duas camadas (etapa 3)

```text
[Cliente / Navegador]
        │  HTTP :8501
        ▼
┌── AWS (VPC padrão, us-east-2) ───────────────────────────────────────────────┐
│                                                                              │
│  EC2 churn-app-host (role EC2-ECR-ReadRole)     EC2 churn-db-host            │
│  ┌── container app-churn ────────────┐          ┌── container churn_postgres ┐│
│  │ Supervisord                       │  SQL     │ PostgreSQL 15              ││
│  │  ├─ Streamlit  :8501 (app_v2)     │  :5432   │ banco churn_db             ││
│  │  └─ FastAPI    :8000 (api_v2) ────┼─────────►│ tabela customers (7.043)   ││
│  │     + modelo RandomForest         │ IP priv. │ volume pgdata              ││
│  └───────────────────────────────────┘          └────────────────────────────┘│
│           ▲ docker pull (STS)                                                │
│     Amazon ECR app-churn-v1:latest                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

As duas máquinas conversam pelo **IP privado** da VPC. A credencial do banco nunca vai para a imagem nem para o Git: entra no container em tempo de execução via `--env-file`.

## 3. Endpoints da API (`src/api_v2.py`)

| Método | Rota | O que faz |
| --- | --- | --- |
| GET | `/health` | Status do modelo e da conexão com o Postgres (com contagem de registros) |
| GET | `/customers/{customerID}` | Consulta pura: registro do cliente como está no banco |
| GET | `/results/{customerID}` | Busca o cliente no banco e calcula o score do modelo sobre ele |
| POST | `/predict` | Inferência online para um cliente novo (não depende do banco) |

Se o banco estiver fora do ar, as rotas de consulta respondem `503` e o `/predict` continua funcionando.

---

## 4. Passo a passo de implantação

### 4.1 Camada de dados (`churn-db-host`)
1. EC2 `t3.micro` Ubuntu, mesma VPC da aplicação. Security Group com a porta **5432** liberada para o Security Group da aplicação (evite `0.0.0.0/0`).
2. No terminal da máquina, clone o repositório e coloque o `churn.csv` real em `data/raw/`.
3. Rode:
   ```bash
   export DB_PASSWORD='sua-senha'
   bash infra/setup_db_ec2.sh
   ```
   Saída esperada: `COPY 7043`.

### 4.2 Imagem da aplicação (máquina local, Windows)
```powershell
python -m dvc pull                 # garante data/raw/churn.csv (usado pelo lote da v1 no build)
.\infra\deploy_ecr.ps1 -AwsProfile churn   # exporta o @champion, faz o build e o push para o ECR
```
Espere aparecer `Publicacao concluida com sucesso!` antes do passo 4.3: se a EC2 baixar a imagem antes do push terminar, ela recebe a versão anterior (confira o `Digest:` do pull).

### 4.3 Camada de aplicação (`churn-app-host`)
1. Crie `~/churn.env` (modelo em `.env.example`) com `DB_HOST=<IP privado da churn-db-host>` e a senha.
2. Libere espaço (o disco tem ~7 GB), baixe o script da `main` e rode:
   ```bash
   sudo docker stop churn_app && sudo docker rm churn_app
   sudo docker system prune -a -f
   curl -sO https://raw.githubusercontent.com/caiorj2004/mlops-churn-project/main/infra/setup_ec2.sh
   bash setup_ec2.sh
   ```
   O script sobe o container com `--env-file` e imprime o `/health`, que deve mostrar o modelo `@champion` e `records: 7043`.
3. Acesse `http://<IP público da churn-app-host>:8501`.

## 5. Teste local (VS Code)
```powershell
copy .env.example .env               # ajuste DB_HOST para o IP PÚBLICO da churn-db-host e a senha
python -m src.export_champion        # copia o @champion do MLflow para models/champion/
python -m uvicorn src.api_v2:app --port 8000 --reload
streamlit run app_churn_v2.py        # em outro terminal
```
Para testar de fora da AWS, a regra da porta 5432 precisa aceitar o seu IP público.

## 6. Custos
Instâncias `t3.micro` consomem créditos enquanto estão **Executando**. Pare as duas ao final (`Estado da instância > Interromper`). Ao religar, o IP **público** muda, mas o **privado** se mantém, então o `DB_HOST` da aplicação continua válido.
