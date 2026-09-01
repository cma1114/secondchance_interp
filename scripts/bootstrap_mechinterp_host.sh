#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv-mechinterp
.venv-mechinterp/bin/python -m pip install --upgrade pip
.venv-mechinterp/bin/python -m pip install -r requirements-mechinterp.txt

echo "Environment created. Activate it with: source .venv-mechinterp/bin/activate"
echo "No cloud instance is created and no paid API is called by this script."

