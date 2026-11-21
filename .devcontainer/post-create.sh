#!/usr/bin/env bash
# Installs dependencies and prepares the writable output folder.
# data/ is mounted read-only in this container and is never touched here.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r .devcontainer/requirements-dev.txt

mkdir -p runs
touch runs/.gitkeep
