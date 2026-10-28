# Copies the local .env (GROQ_API_KEY etc.) to the instance and restarts the
# container so it picks the values up. The key never enters git or the image.
#
# Usage: .\deploy\aws-env.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$awsJson = Join-Path $PSScriptRoot "aws.json"
$cfg = Get-Content $awsJson | ConvertFrom-Json
$ip = $cfg.publicIp
if (-not $ip) { throw "no publicIp in deploy/aws.json - run deploy/aws-start.ps1 first" }
$localEnv = Join-Path $repo ".env"
if (-not (Test-Path $localEnv)) { throw "no .env in the repo root - copy .env.example and fill it in" }

scp -i $cfg.keyPath -o StrictHostKeyChecking=accept-new $localEnv "ubuntu@${ip}:/home/ubuntu/RepoRag/.env"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }

ssh -i $cfg.keyPath -o StrictHostKeyChecking=accept-new "ubuntu@$ip" `
    "cd /home/ubuntu/RepoRag && docker compose -f deploy/docker-compose.yml up -d --force-recreate app"
if ($LASTEXITCODE -ne 0) { throw "container restart failed" }

Start-Sleep -Seconds 5
$health = Invoke-RestMethod "http://${ip}:8000/api/health" -TimeoutSec 30
Write-Host "env applied - healthy: chunks=$($health.chunks) model=$($health.model)"
