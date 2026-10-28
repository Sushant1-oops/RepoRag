# Opens an interactive SSH session on the backend instance, e.g. to watch
# logs: docker compose -f deploy/docker-compose.yml logs -f app
#
# Usage: .\deploy\aws-ssh.ps1

$ErrorActionPreference = "Stop"
$awsJson = Join-Path $PSScriptRoot "aws.json"
$cfg = Get-Content $awsJson | ConvertFrom-Json
$ip = $cfg.publicIp
if (-not $ip) { throw "no publicIp in deploy/aws.json - run deploy/aws-start.ps1 first" }

ssh -i $cfg.keyPath -o StrictHostKeyChecking=accept-new "ubuntu@$ip"
