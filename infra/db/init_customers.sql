-- Carga CSV -> PostgreSQL (camada de dados, EC2 churn-db-host)
-- Mesmo DDL executado no container churn_postgres durante a aula.
-- Os nomes de coluna ficam entre aspas para preservar o formato do churn.csv
-- (customerID, SeniorCitizen, ...). TotalCharges fica como texto porque o CSV
-- original pode trazer valores em branco; a API converte para numero.

CREATE TABLE IF NOT EXISTS customers (
    "customerID"       VARCHAR(50) PRIMARY KEY,
    "gender"           VARCHAR(20),
    "SeniorCitizen"    INT,
    "Partner"          VARCHAR(10),
    "Dependents"       VARCHAR(10),
    "tenure"           INT,
    "PhoneService"     VARCHAR(10),
    "MultipleLines"    VARCHAR(30),
    "InternetService"  VARCHAR(30),
    "OnlineSecurity"   VARCHAR(30),
    "OnlineBackup"     VARCHAR(30),
    "DeviceProtection" VARCHAR(30),
    "TechSupport"      VARCHAR(30),
    "StreamingTV"      VARCHAR(30),
    "StreamingMovies"  VARCHAR(30),
    "Contract"         VARCHAR(30),
    "PaperlessBilling" VARCHAR(10),
    "PaymentMethod"    VARCHAR(50),
    "MonthlyCharges"   NUMERIC(10, 2),
    "TotalCharges"     VARCHAR(50),
    "Churn"            VARCHAR(10)
);

TRUNCATE TABLE customers;

\copy customers FROM '/tmp/churn.csv' WITH (FORMAT csv, HEADER true, DELIMITER ',');

SELECT count(*) AS total_registros FROM customers;
