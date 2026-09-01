#!/usr/bin/env bash
set -euo pipefail
set -a
source .env
set +a
exec .venv/bin/python scripts/vast_fleet_guard.py "$@"
