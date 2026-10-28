#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "Instalação interrompida na linha $LINENO" >&2' ERR
if [ "${EUID}" -ne 0 ]; then echo "Executa: sudo bash install.sh"; exit 1; fi
if [ ! -f /etc/os-release ]; then echo "Linux não reconhecido"; exit 1; fi
. /etc/os-release
if ! command -v apt-get >/dev/null 2>&1; then echo "Esta versão suporta Debian/Ubuntu (apt)"; exit 1; fi
if ! command -v systemctl >/dev/null 2>&1; then echo "É necessário systemd"; exit 1; fi
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[1/6] Dependências do sistema"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3 curl ca-certificates openssl
echo "[2/6] Conta e diretórios"
if ! id nordai >/dev/null 2>&1; then useradd --system --home /var/lib/nord-ai --shell /usr/sbin/nologin nordai; fi
install -d -m 750 -o nordai -g nordai /var/lib/nord-ai
install -d -m 755 /opt/nord-ai/server /opt/nord-ai/web
install -m 644 "$SOURCE/server/main.py" /opt/nord-ai/server/main.py
install -m 644 "$SOURCE/web/index.html" "$SOURCE/web/style.css" "$SOURCE/web/app.js" /opt/nord-ai/web/
chown -R root:root /opt/nord-ai
echo "[3/6] Credenciais"
if [ ! -f /etc/nord-ai.env ]; then
 PASSWORD="$(openssl rand -hex 18)"
 printf 'NORD_PASSWORD=%s\nNORD_HOST=127.0.0.1\nNORD_PORT=8787\nNORD_DATA=/var/lib/nord-ai\n' "$PASSWORD" > /etc/nord-ai.env
 chmod 600 /etc/nord-ai.env
 echo "PALAVRA-PASSE NORD AI: $PASSWORD"
 echo "Guarda-a num local seguro; também existe em /etc/nord-ai.env (root)."
else
 echo "A manter a palavra-passe existente em /etc/nord-ai.env."
fi
echo "[4/6] Motor Ollama"
if ! command -v ollama >/dev/null 2>&1; then
 curl -fsSL https://ollama.com/install.sh -o /tmp/nord-ollama-install.sh
 sh /tmp/nord-ollama-install.sh
 rm -f /tmp/nord-ollama-install.sh
fi
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/nord-limits.conf <<'EOF'
[Service]
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_CONTEXT_LENGTH=1024"
Environment="OLLAMA_KEEP_ALIVE=1m"
MemoryMax=1500M
EOF
echo "[5/6] Serviços"
cat > /etc/systemd/system/nord-ai.service <<'EOF'
[Unit]
Description=Nord AI lightweight private platform
After=network-online.target ollama.service
Wants=network-online.target
Requires=ollama.service
[Service]
Type=simple
User=nordai
Group=nordai
WorkingDirectory=/opt/nord-ai
EnvironmentFile=/etc/nord-ai.env
ExecStart=/usr/bin/python3 /opt/nord-ai/server/main.py
Restart=on-failure
RestartSec=3
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/nord-ai
MemoryMax=180M
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now ollama
for i in $(seq 1 30); do
 if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
 sleep 2
done
echo "[6/6] Modelo inicial: gemma3:270m"
if ! ollama list | grep -q '^gemma3:270m '; then
 ollama pull gemma3:270m
fi
systemctl enable --now nord-ai
systemctl restart nord-ai
echo
echo "Nord AI instalado."
echo "Estado: systemctl status nord-ai"
echo "Acesso remoto seguro através de túnel SSH:"
echo "  ssh -L 8787:127.0.0.1:8787 UTILIZADOR@IP_DA_VM"
echo "Depois abre http://127.0.0.1:8787 no teu browser."
