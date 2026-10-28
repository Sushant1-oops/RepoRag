# Uploads the local Qdrant index, chat DB and repo registry to the instance
# so the deployed app starts with everything already embedded. The app
# container is stopped during the copy (Qdrant local mode takes an exclusive
# file lock) and restarted afterwards.
#
# Usage: .\deploy\aws-upload-data.ps1

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$awsJson = Join-Path $PSScriptRoot "aws.json"
$cfg = Get-Content $awsJson | ConvertFrom-Json
$ip = $cfg.publicIp
if (-not $ip) { throw "no publicIp in deploy/aws.json - run deploy/aws-start.ps1 first" }
$remote = "ubuntu@${ip}"

ssh -i $cfg.keyPath -o StrictHostKeyChecking=accept-new $remote `
    "cd /home/ubuntu/RepoRag && docker compose -f deploy/docker-compose.yml stop app; sudo mkdir -p data && sudo chown -R ubuntu:ubuntu data"
if ($LASTEXITCODE -ne 0) { throw "could not stop the container" }

scp -i $cfg.keyPath -o StrictHostKeyChecking=accept-new -r `
    (Join-Path $repo "data/qdrant") "${remote}:/home/ubuntu/RepoRag/data/"
if ($LASTEXITCODE -ne 0) { throw "qdrant upload failed" }
scp -i $cfg.keyPath -o StrictHostKeyChecking=accept-new `
    (Join-Path $repo "data/app.db") (Join-Path $repo "data/repos.json") `
    "${remote}:/home/ubuntu/RepoRag/data/"
if ($LASTEXITCODE -ne 0) { throw "db/registry upload failed" }

ssh -i $cfg.keyPath -o StrictHostKeyChecking=accept-new $remote `
    "cd /home/ubuntu/RepoRag && docker compose -f deploy/docker-compose.yml up -d app"
if ($LASTEXITCODE -ne 0) { throw "container restart failed" }

Start-Sleep -Seconds 8
$health = Invoke-RestMethod "http://${ip}:8000/api/health" -TimeoutSec 60
Write-Host "data uploaded - healthy: chunks=$($health.chunks)"
