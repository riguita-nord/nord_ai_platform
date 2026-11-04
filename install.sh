#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
if [[ "$(uname -s)" == "Linux" ]]; then
  .venv/bin/python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
else
  .venv/bin/python -m pip install -r requirements.txt
fi
printf '\nNord Brain Transformer instalado.\nInicia: .venv/bin/python -m nord_brain\n'
printf 'Para placas/VM leves: NORD_BRAIN_PRESET=tiny .venv/bin/python -m nord_brain\n'
printf 'Pesquisa opcional: NORD_INTERNET=on .venv/bin/python -m nord_brain\n'
