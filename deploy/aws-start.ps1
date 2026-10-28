# Starts the RepoRag backend instance and waits until the app answers.
# The public IP changes on every start unless the instance was created with
# -AllocateElasticIp, so the fresh IP is written back to deploy/aws.json.
#
# Usage: .\deploy\aws-start.ps1 [-SkipHealthCheck]

param([switch]$SkipHealthCheck)

$ErrorActionPreference = "Stop"
$awsJson = Join-Path $PSScriptRoot "aws.json"
if (-not (Test-Path $awsJson)) {
    throw "deploy/aws.json not found - run deploy/aws-create.ps1 first"
}
$cfg = Get-Content $awsJson | ConvertFrom-Json

function Aws { aws @Args --region $cfg.region --output json }

$state = (Aws ec2 describe-instances --instance-ids $cfg.instanceId |
    ConvertFrom-Json).Reservations[0].Instances[0]
if ($state.State.Name -eq "terminated") { throw "instance was terminated - create a new one" }

if ($state.State.Name -ne "running") {
    Write-Host "starting $($cfg.instanceId) ..."
    Aws ec2 start-instances --instance-ids $cfg.instanceId | Out-Null
    Aws ec2 wait instance-running --instance-ids $cfg.instanceId | Out-Null
    Write-Host "instance is running"
} else {
    Write-Host "instance already running"
}

# refresh the public IP (it rotates across stop/start without an Elastic IP)
$state = (Aws ec2 describe-instances --instance-ids $cfg.instanceId |
    ConvertFrom-Json).Reservations[0].Instances[0]
$ip = if ($cfg.elasticIp) { $cfg.elasticIp } else { $state.PublicIpAddress }
if (-not $ip) { throw "instance has no public IP yet - retry in a few seconds" }
$cfg | Add-Member -Force -NotePropertyName publicIp -NotePropertyValue $ip
$cfg | ConvertTo-Json | Set-Content $awsJson -Encoding utf8
Write-Host "public ip: $ip"

if ($SkipHealthCheck) { Write-Host "http://${ip}:8000"; return }

# a cold start reloads docker + models; the very first boot also builds the
# image, which can take 5-10 minutes
Write-Host "waiting for http://${ip}:8000/api/health (first boot takes a while) ..."
$deadline = (Get-Date).AddMinutes(15)
while ((Get-Date) -lt $deadline) {
    try {
        $health = Invoke-RestMethod "http://${ip}:8000/api/health" -TimeoutSec 10
        Write-Host "healthy: chunks=$($health.chunks) model=$($health.model)"
        Write-Host "app ready at http://${ip}:8000"
        return
    } catch {
        Start-Sleep -Seconds 15
        Write-Host "  ... still booting ($(Get-Date -Format HH:mm:ss))"
    }
}
Write-Warning "no health response within 15 minutes - check logs:"
Write-Warning "  .\deploy\aws-ssh.ps1  then: docker compose -f deploy/docker-compose.yml logs app"
