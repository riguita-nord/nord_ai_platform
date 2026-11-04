#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
printf '\nInstalação concluída. Inicia com: .venv/bin/python -m nord_brain\n'
printf 'Para pesquisa opcional: NORD_INTERNET=on .venv/bin/python -m nord_brain\n'
