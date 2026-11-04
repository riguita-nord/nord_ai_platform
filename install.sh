#!/usr/bin/env bash
# One-command local setup: bash install.sh
# Installs into the repository's virtual environment and launches Nord Brain.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'HELP'
Nord Brain — instalação e arranque num comando

  bash install.sh                   Instala o necessário e abre a Nord Brain
  bash install.sh --install-only    Instala sem iniciar a IA
  bash install.sh --data ./dados    Instala e abre com esta pasta de memória
  ./nord                           Inicia novamente, sem reinstalar

Variáveis opcionais: PYTHON, NORD_BRAIN_PRESET=tiny|small|medium,
                    NORD_INTERNET=on, NORD_AUTO_TRAIN=off.
Não necessita de sudo. Os dados persistentes ficam em ~/.nord-brain/ por defeito.
HELP
}

INSTALL_ONLY=0
case "${1:-}" in
  --help|-h) usage; exit 0 ;;
  --install-only) INSTALL_ONLY=1; shift ;;
  --) shift ;;
esac
if ((INSTALL_ONLY)) && (( $# )); then
  printf 'Erro: --install-only não aceita argumentos adicionais.\n' >&2
  exit 2
fi

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Erro: Python 3.10+ não encontrado. Instala python3 e python3-venv.\n' >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  printf 'Erro: é necessário Python 3.10 ou mais recente.\n' >&2
  exit 1
fi

printf '\n[Nord Brain] A preparar o ambiente local...\n'
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  if ! "$PYTHON" -m venv "$ROOT/.venv"; then
    printf 'Erro: não foi possível criar o ambiente virtual. Verifica python3-venv.\n' >&2
    exit 1
  fi
fi

VENV_PY="$ROOT/.venv/bin/python"
if ! "$VENV_PY" -c 'import torch; assert tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 2)' >/dev/null 2>&1; then
  printf '[Nord Brain] A instalar PyTorch (a primeira instalação pode descarregar vários MB)...\n'
  "$VENV_PY" -m pip install --upgrade pip
  if [[ "$(uname -s)" == Linux ]]; then
    "$VENV_PY" -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
  else
    "$VENV_PY" -m pip install -r "$ROOT/requirements.txt"
  fi
fi

if ! "$VENV_PY" -c 'import torch' >/dev/null 2>&1; then
  printf 'Erro: PyTorch não está disponível no ambiente virtual.\n' >&2
  exit 1
fi

printf '\n[Nord Brain] Dependências prontas.\n'
if ((INSTALL_ONLY)); then
  printf 'Para iniciar: %s/nord\n' "$ROOT"
  exit 0
fi
printf '[Nord Brain] A iniciar... (para sair escreve /sair)\n\n'
exec "$ROOT/nord" "$@"
