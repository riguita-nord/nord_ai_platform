#!/usr/bin/env bash
# Explicit opt-in: downloads pretrained GGUF (~397 MB), builds llama.cpp CPU,
# and starts a LOCAL-only inference service. Never runs during ./nord install.
set -euo pipefail
CACHE="${NORD_MODEL_CACHE:-${HOME}/.cache/nord-brain}"
MODEL="${CACHE}/models/Qwen3-0.6B-Q4_K_M.gguf"
URL="https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf"
SHA="ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a"
mkdir -p "${CACHE}/models"
if [[ ! -f "$MODEL" ]]; then
    command -v curl >/dev/null || { echo 'Instala curl e volta a executar.' >&2; exit 1; }
    echo '[Nord Brain] A descarregar o Qwen3-0.6B GGUF para a pasta local de modelos...'
    curl --fail --location --retry 2 --output "${MODEL}.part" "$URL"
    echo "${SHA}  ${MODEL}.part" | sha256sum -c - || { rm -f "${MODEL}.part"; exit 1; }
    mv "${MODEL}.part" "$MODEL"
fi
# Validate an already cached model as well; never launch damaged weights.
echo "${SHA}  ${MODEL}" | sha256sum -c - >/dev/null || { echo 'Modelo local com hash inesperado.' >&2; exit 1; }
SERVER="${NORD_LLAMA_SERVER:-}"
if [[ -z "$SERVER" ]]; then
    SERVER="$(command -v llama-server || true)"
fi
if [[ -z "$SERVER" && -x "${CACHE}/llama.cpp/build/bin/llama-server" ]]; then
    SERVER="${CACHE}/llama.cpp/build/bin/llama-server"
fi
if [[ -z "$SERVER" ]]; then
    for dep in git cmake c++; do
        command -v "$dep" >/dev/null || { echo "Precisas de $dep para compilar llama.cpp. Instala git cmake build-essential." >&2; exit 1; }
    done
    if [[ ! -d "${CACHE}/llama.cpp/.git" ]]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "${CACHE}/llama.cpp"
    fi
    cmake -S "${CACHE}/llama.cpp" -B "${CACHE}/llama.cpp/build" -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build "${CACHE}/llama.cpp/build" --target llama-server -j "${NORD_BUILD_JOBS:-1}"
    SERVER="${CACHE}/llama.cpp/build/bin/llama-server"
fi
[[ -x "$SERVER" ]] || { echo "Não encontrei llama-server executável: $SERVER" >&2; exit 1; }
echo '[Nord Brain] A iniciar modelo linguístico em http://127.0.0.1:8081 (sem GPU).'
exec "$SERVER" -m "$MODEL" --host 127.0.0.1 --port 8081 -c 1536 -ngl 0 -t "${NORD_THREADS:-2}" --parallel 1
