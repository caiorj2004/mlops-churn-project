param (
    [string]$Region = "us-east-2",
    [string]$RepoName = "app-churn-v1",
    [string]$AwsProfile = ""   # ex.: -AwsProfile churn (perfil configurado com `aws configure --profile churn`)
)

$ErrorActionPreference = "Stop"

# Usa o AWS CLI v2 se estiver instalado; senao, o pacote pip `awscli` (python -m awscli)
$UseAwsV2 = [bool](Get-Command aws -ErrorAction SilentlyContinue)
$ProfileArgs = if ($AwsProfile) { @("--profile", $AwsProfile) } else { @() }

function Invoke-Aws {
    if ($UseAwsV2) { & aws @args @ProfileArgs } else { & python -m awscli @args @ProfileArgs }
}

Write-Host "--> Obtendo Account ID da AWS..." -ForegroundColor Cyan
$AccountId = (Invoke-Aws sts get-caller-identity --query Account --output text)
if (-not $AccountId) { throw "Nao foi possivel obter o Account ID. Verifique as credenciais AWS." }
$AccountId = $AccountId.Trim()
$EcrUri = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$LocalImage = "$($RepoName):latest"
$ImageFullName = "$EcrUri/$($RepoName):latest"

if (-not (Test-Path "data/raw/churn.csv")) { throw "data/raw/churn.csv ausente. Rode 'python -m dvc pull' antes do build." }

Write-Host "--> 0. Exportando o modelo @champion do MLflow Model Registry..." -ForegroundColor Cyan
python -m src.export_champion
if ($LASTEXITCODE -ne 0) { throw "Falha ao exportar o @champion. Rode src.experiments e src.register_model antes." }

Write-Host "--> 1. Autenticando Docker no Amazon ECR ($Region)..." -ForegroundColor Cyan
Invoke-Aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $EcrUri

Write-Host "--> 2. Compilando imagem Docker local..." -ForegroundColor Cyan
docker build -t $LocalImage .

Write-Host "--> 3. Tagueando imagem: $ImageFullName..." -ForegroundColor Cyan
docker tag $LocalImage $ImageFullName

Write-Host "--> 4. Enviando camadas para o ECR..." -ForegroundColor Cyan
docker push $ImageFullName

Write-Host "`nPublicacao concluida com sucesso! URI: $ImageFullName" -ForegroundColor Green
