param (
    [string]$Region = "us-east-2",
    [string]$RepoName = "app-churn-v1"
)

$ErrorActionPreference = "Stop"

Write-Host "--> Obtendo Account ID da AWS..." -ForegroundColor Cyan
$AccountId = (python -m awscli sts get-caller-identity --query Account --output text).Trim()
$EcrUri = "$AccountId.dkr.ecr.$Region.amazonaws.com"
$ImageFullName = "$EcrUri/$($RepoName):latest"

Write-Host "--> 1. Autenticando Docker no Amazon ECR ($Region)..." -ForegroundColor Cyan
python -m awscli ecr get-login-password --region $Region | docker login --username AWS --password-stdin $EcrUri

Write-Host "--> 2. Compilando imagem Docker local (com batch scoring real)..." -ForegroundColor Cyan
docker build -t "$RepoName:latest" .

Write-Host "--> 3. Tagueando imagem: $ImageFullName..." -ForegroundColor Cyan
docker tag "$RepoName:latest" $ImageFullName

Write-Host "--> 4. Enviando camadas para o ECR..." -ForegroundColor Cyan
docker push $ImageFullName

Write-Host "`nPublicacao concluida com sucesso! URI: $ImageFullName" -ForegroundColor Green
