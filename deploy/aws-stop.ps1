# Stops the RepoRag backend instance. Compute billing stops immediately;
# the EBS volume (index, chats, repo registry) stays put and costs roughly
# $2.4/month for the default 30GB gp3 volume while the instance is stopped.
#
# Usage: .\deploy\aws-stop.ps1

$ErrorActionPreference = "Stop"
$awsJson = Join-Path $PSScriptRoot "aws.json"
if (-not (Test-Path $awsJson)) { throw "deploy/aws.json not found - nothing to stop" }
$cfg = Get-Content $awsJson | ConvertFrom-Json

aws ec2 stop-instances --instance-ids $cfg.instanceId --region $cfg.region --output json | Out-Null
Write-Host "stop requested, waiting for the instance to settle ..."
aws ec2 wait instance-stopped --instance-ids $cfg.instanceId --region $cfg.region
Write-Host "instance $($cfg.instanceId) is stopped - data persists on the EBS volume"
Write-Host "start it again any time with: .\deploy\aws-start.ps1"
