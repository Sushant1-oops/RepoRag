#!/bin/bash
# cloud-init user-data for the RepoRag EC2 instance (Ubuntu 24.04).
# Runs once on first boot: installs docker, clones the repo, and brings the
# compose stack up. The image build (frontend + CPU torch + model preheat)
# happens here, so the first boot takes ~5-10 minutes before :8000 answers.
set -euo pipefail

REPO_URL="https://github.com/Sushant1-oops/RepoRag.git"
APP_DIR="/home/ubuntu/RepoRag"

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get -y install docker.io docker-compose-v2 git
systemctl enable --now docker

# the compose build runs as ubuntu; give it docker rights without a re-login
usermod -aG docker ubuntu || true

if [ ! -d "${APP_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${APP_DIR}"
fi

cd "${APP_DIR}"
# .env is not in git (it holds GROQ_API_KEY); create a placeholder so the
# stack starts and the health endpoint works. Push the real key afterwards
# with deploy/aws-env.ps1, which restarts the container to pick it up.
[ -f .env ] || touch .env
mkdir -p data

sudo -u ubuntu docker compose -f deploy/docker-compose.yml up -d

echo "user-data done" > /var/log/reporag-user-data.done
