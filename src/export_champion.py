"""Exporta o modelo @champion do MLflow Model Registry para models/champion/.

Ponte entre o ML Pipeline e o deploy: o Model Registry vive no mlflow.db local,
entao, antes do `docker build`, o @champion e copiado para uma pasta que a
imagem da aplicacao inclui. Assim os apps da AWS servem exatamente o modelo
promovido no registry, com a linhagem preservada em champion_info.json.

Conteudo gerado em models/champion/:
    MLmodel, model.pkl, requirements.txt ...  pacote MLflow Models (flavor sklearn + pyfunc)
    feature_contract.json                      categorias + ordem das colunas usadas no treino
    champion_info.json                         versao, run, metricas e linhagem do modelo

Uso:  python -m src.export_champion
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "churn-model"
ALIAS = "champion"
EXPORT_DIR = Path("models") / "champion"


def export_champion(export_dir: Path = EXPORT_DIR) -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run = client.get_run(mv.run_id)
    artefatos = {a.path for a in client.list_artifacts(mv.run_id)}
    if "feature_contract.json" not in artefatos:
        raise SystemExit(
            f"O run {mv.run_id} do @champion nao tem feature_contract.json. "
            "Rode de novo `python -m src.experiments` e `python -m src.register_model`."
        )

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)

    # 1. Pacote MLflow Models da versao apontada por @champion
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}@{ALIAS}", dst_path=str(export_dir))
    # 2. Contrato de features do mesmo run
    mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="feature_contract.json", dst_path=str(export_dir))

    info = {
        "model_name": MODEL_NAME,
        "alias": f"@{ALIAS}",
        "version": int(mv.version),
        "run_id": mv.run_id,
        "run_name": run.data.tags.get("mlflow.runName"),
        "test_accuracy": run.data.metrics.get("test_accuracy"),
        "n_estimators": run.data.params.get("n_estimators"),
        "max_depth": run.data.params.get("max_depth"),
        "data_md5": run.data.tags.get("data_md5"),
        "git_commit": run.data.tags.get("git_commit"),
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    (export_dir / "champion_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


if __name__ == "__main__":
    info = export_champion()
    print(f"@champion exportado para {EXPORT_DIR}/")
    for chave, valor in info.items():
        print(f"  {chave}: {valor}")
