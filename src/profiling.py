import pandas as pd
from src.config import DATA_PATH


def run_profiling():
    """Realiza o Profiling inicial do dataset de Churn."""
    print("=== PROFILING DO DATASET ===")
    df = pd.read_csv(DATA_PATH)

    print("\n1. Estrutura e Tipos:")
    print(df.info())

    print("\n2. Resumo Estatístico (Numéricas):")
    print(df.describe().T)

    print("\n3. Contagem de Nulos por Coluna:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\n4. Distribuição da Variável Alvo (Churn):")
    print(df["Churn"].value_counts(normalize=True) * 100)

    # Tenta gerar o relatório HTML se o ydata-profiling estiver instalado
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(
            df, title="Relatório de Profiling - Churn Dataset"
        )
        profile.to_file("churn_profile_report.html")
        print(
            "\n✅ Relatório HTML gerado com sucesso: churn_profile_report.html"
        )
    except ImportError:
        print(
            "\n💡 Para gerar o relatório HTML completo, instale: pip install ydata-profiling"
        )


if __name__ == "__main__":
    run_profiling()