#!/usr/bin/env bash
# Nord Brain — one-command install and immediate terminal launch.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'HELP'
Nord Brain — instalação e arranque num comando
  bash install.sh                  Instala o necessário e abre a Nord Brain
  bash install.sh --install-only   Instala sem iniciar a IA
  bash install.sh --data ./dados   Instala e abre com esta pasta de memória
  ./nord                          Inicia novamente, sem reinstalar

Pode solicitar a palavra-passe sudo para instalar o python3.X-venv no Ubuntu/Debian.
Os pesos e a memória existentes são preservados.
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
  printf 'Erro: Python não encontrado. Instala Python 3.10+ e python3-venv.\n' >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  printf 'Erro: é necessário Python 3.10 ou mais recente.\n' >&2
  exit 1
fi
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"
venv_ready() {
  [[ -x "$VENV_PY" ]] &&
    "$VENV_PY" -c 'import sys; assert sys.prefix != sys.base_prefix' >/dev/null 2>&1 &&
    "$VENV_PY" -m pip --version >/dev/null 2>&1
}
install_venv_support() {
  if [[ "$(uname -s)" != Linux || ! -f /etc/os-release ]] || ! command -v apt-get >/dev/null 2>&1; then
    printf 'Erro: instala o pacote venv apropriado à distribuição e volta a executar bash install.sh.\n' >&2
    return 1
  fi
  # Only offer apt on Debian/Ubuntu derivatives.
  local distro=""
  distro="$(. /etc/os-release; printf '%s %s' "${ID:-}" "${ID_LIKE:-}")"
  if [[ ! " $distro " =~ (^|[[:space:]])(debian|ubuntu)([[:space:]]|$) ]]; then
    printf 'Erro: instalação automática de venv disponível apenas em Debian/Ubuntu.\n' >&2
    return 1
  fi
  local pkg
  pkg="$("$PYTHON" -c 'import sys; print("python%d.%d-venv" % sys.version_info[:2])')"
  local -a apt=(apt-get)
  if (( EUID != 0 )); then
    if ! command -v sudo >/dev/null 2>&1; then
      printf 'Erro: instala %s com privilégios de administrador (sudo indisponível).\n' "$pkg" >&2
      return 1
    fi
    apt=(sudo apt-get)
  fi
  printf '[Nord Brain] Falta o suporte venv/ensurepip. A instalar %s...\n' "$pkg"
  printf '[Nord Brain] O sistema poderá pedir a palavra-passe sudo.\n'
  if ! "${apt[@]}" install -y "$pkg"; then
    printf '[Nord Brain] A tentar o pacote genérico python3-venv...\n'
    "${apt[@]}" install -y python3-venv || {
      printf 'Erro: não foi possível instalar o venv automaticamente. Executa: sudo apt-get update && sudo apt-get install -y %s\n' "$pkg" >&2
      return 1
    }
  fi
}

printf '\n[Nord Brain] A preparar o ambiente local...\n'
if [[ -L "$VENV" ]]; then
  printf 'Erro: .venv é uma ligação simbólica; por segurança não a vamos substituir.\n' >&2
  exit 1
fi
if ! venv_ready; then
  # --clear repairs incomplete .venv directories left after ensurepip failed.
  if ! result="$("$PYTHON" -m venv --clear "$VENV" 2>&1)"; then
    printf '%s\n' "$result" >&2
    if [[ "${result,,}" == *ensurepip* || "${result,,}" == *python3-venv* || "${result,,}" =~ python[0-9]+\.[0-9]+-venv ]]; then
      install_venv_support || exit 1
      printf '[Nord Brain] A reconstruir o ambiente virtual incompleto...\n'
      "$PYTHON" -m venv --clear "$VENV" || {
        printf 'Erro: a criação do venv voltou a falhar. Verifica o Python e o espaço em disco.\n' >&2
        exit 1
      }
    else
      printf 'Erro: a criação do venv falhou por outro motivo. Verifica o erro acima.\n' >&2
      exit 1
    fi
  fi
fi
if ! venv_ready; then
  printf 'Erro: ambiente virtual criado sem pip funcional. Tenta sudo apt-get install python3-venv.\n' >&2
  exit 1
fi
if ! "$VENV_PY" -c 'import torch; assert tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 2)' >/dev/null 2>&1; then
  printf '[Nord Brain] A instalar PyTorch...\n'
  "$VENV_PY" -m pip install --upgrade pip
  if [[ "$(uname -s)" == Linux ]]; then
    "$VENV_PY" -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
  else
    "$VENV_PY" -m pip install -r "$ROOT/requirements.txt"
  fi
fi
if ! "$VENV_PY" -c 'import numpy as np; v=tuple(map(int,np.__version__.split(".")[:2])); assert (1,24)<=v<(3,0)' >/dev/null 2>&1; then
  printf '[Nord Brain] A instalar NumPy (dependência necessária para PyTorch)...\n'
  "$VENV_PY" -m pip install 'numpy>=1.24,<3'
fi
printf '\n[Nord Brain] Dependências prontas.\n'
if ((INSTALL_ONLY)); then
  printf 'Para iniciar: %s/nord\n' "$ROOT"
  exit 0
fi
printf '[Nord Brain] A iniciar...\n\n'
exec "$ROOT/nord" "$@"
