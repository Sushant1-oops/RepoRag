# One-time AWS setup for the RepoRag backend.
#
# Creates (idempotently): an EC2 key pair, a security group with SSH (limited
# to this machine's IP when it can be detected) and port 8000 open, and an
# Ubuntu 24.04 instance that boots the compose stack via deploy/ec2-user-data.sh.
# Everything the app persists lives on the instance's EBS volume, so the
# instance can be stopped (deploy/aws-stop.ps1) and started again later
# (deploy/aws-start.ps1) without losing the index or chat history.
#
# Usage:
#   .\deploy\aws-create.ps1                              # t3.medium, us-east-1
#   .\deploy\aws-create.ps1 -InstanceType t3.small -Region eu-central-1
#
# Requires: aws CLI configured once (aws configure) with credentials that can
# create EC2 resources.

param(
    [string]$Region = $(if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }),
    [string]$InstanceType = "t3.medium",
    [string]$KeyName = "reporag-key",
    [string]$SgName = "reporag-sg",
    [int]$VolumeSize = 30,
    # keep the same public IP across stop/start cycles (~$3.6/mo while stopped,
    # free while the instance runs)
    [switch]$AllocateElasticIp
)

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$awsJson = Join-Path $PSScriptRoot "aws.json"
$keyPath = Join-Path $PSScriptRoot "$KeyName.pem"

function Aws { aws @Args --region $Region --output json }

Write-Host "== AWS credentials check =="
$identity = Aws sts get-caller-identity | ConvertFrom-Json
Write-Host "account: $($identity.Account)"

# --- key pair -----------------------------------------------------------------
if (-not (Test-Path $keyPath)) {
    Write-Host "== creating key pair $KeyName =="
    $material = (Aws ec2 create-key-pair --key-name $KeyName) | ConvertFrom-Json
    [IO.File]::WriteAllText($keyPath, $material.KeyMaterial)
    # OpenSSH on Windows refuses world-readable keys
    icacls $keyPath /inheritance:r /grant:r "$env:USERNAME:R" | Out-Null
} else {
    Write-Host "== key pair file already present: $keyPath =="
}
try { Aws ec2 describe-key-pairs --key-names $KeyName | Out-Null }
catch { throw "key file exists but the AWS key pair is missing - use another -KeyName" }

# --- security group -------------------------------------------------------------
Write-Host "== security group $SgName =="
$sgId = try {
    (Aws ec2 describe-security-groups --group-names $SgName |
        ConvertFrom-Json).SecurityGroups[0].GroupId
} catch {
    $vpc = (Aws ec2 describe-vpcs |
        ConvertFrom-Json).Vpcs | Where-Object IsDefault | Select-Object -First 1
    (Aws ec2 create-security-group --group-name $SgName `
        --description "RepoRag backend (ssh + http app)" `
        --vpc-id $vpc.VpcId).GroupId
}
Write-Host "sg: $sgId"

# SSH from this machine's IP when it can be detected, else anywhere
$myIp = try { (Invoke-RestMethod "https://api.ipify.org?format=json").ip + "/32" }
        catch { "0.0.0.0/0" }
$ingress = @(
    "IpProtocol=tcp,FromPort=22,ToPort=22,CidrIp=$myIp",
    "IpProtocol=tcp,FromPort=8000,ToPort=8000,CidrIp=0.0.0.0/0"
)
foreach ($spec in $ingress) {
    try { Aws ec2 authorize-security-group-ingress --group-id $sgId --ip-permissions "[$spec]" | Out-Null }
    catch { Write-Host "  ingress rule already present: $spec" }
}

# --- AMI: Ubuntu 24.04 via Canonical's public SSM parameter ---------------------
Write-Host "== resolving Ubuntu 24.04 AMI in $Region =="
$ssmNames = @(
    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/amd64/ami-id",
    "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp2/amd64/ami-id"
)
$ami = $null
foreach ($name in $ssmNames) {
    $hit = Aws ssm get-parameters --names $name | ConvertFrom-Json
    if ($hit.Parameters -and $hit.Parameters[0].Value) { $ami = $hit.Parameters[0].Value; break }
}
if (-not $ami) { throw "could not resolve an Ubuntu 24.04 AMI in $Region" }
Write-Host "ami: $ami"

# --- instance -------------------------------------------------------------------
Write-Host "== launching $InstanceType instance =="
$userData = Join-Path $PSScriptRoot "ec2-user-data.sh"
# forward slashes: the CLI parses file:// badly with backslash paths
$launch = Aws ec2 run-instances --image-id $ami --instance-type $InstanceType `
    --key-name $KeyName --security-group-ids $sgId `
    --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$VolumeSize,VolumeType=gp3,DeleteOnTermination=true}" `
    --user-data "file://$($userData -replace '\\', '/')" `
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=reporag-backend}]" |
    ConvertFrom-Json
$instanceId = $launch.Instances[0].InstanceId
Write-Host "instance: $instanceId"

$elasticIp = $null
if ($AllocateElasticIp) {
    $alloc = (Aws ec2 allocate-address --domain vpc) | ConvertFrom-Json
    Aws ec2 associate-address --instance-id $instanceId --allocation-id $alloc.AllocationId | Out-Null
    $elasticIp = $alloc.PublicIp
    Write-Host "elastic ip: $elasticIp"
}

@{
    region     = $Region
    instanceId = $instanceId
    keyPath    = $keyPath
    sgId       = $sgId
    elasticIp  = $elasticIp
} | ConvertTo-Json | Set-Content $awsJson -Encoding utf8
Write-Host "== saved $awsJson =="

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Push this branch to GitHub (the instance clones your remote)."
Write-Host "  2. .\deploy\aws-start.ps1        # first boot builds the image (~5-10 min)"
Write-Host "  3. .\deploy\aws-env.ps1          # copy .env (GROQ_API_KEY) to the instance"
Write-Host "  4. .\deploy\aws-upload-data.ps1  # optional: reuse the local Qdrant index"
Write-Host "  5. open http://<ip>:8000"
