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
echo "[Chromium] A verificar instalação..."
if ! command -v chromium >/dev/null 2>&1 && ! command -v chromium-browser >/dev/null 2>&1; then
  if apt-cache show chromium 2>/dev/null | grep -q '^Package: chromium$'; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y chromium
  elif apt-cache show chromium-browser 2>/dev/null | grep -q '^Package: chromium-browser$'; then
    DEBIAN_FRONTEND=noninteractive apt-get install -y chromium-browser
  else
    echo "Aviso: Chromium indisponível nos repositórios; o leitor headless exigirá instalação manual." >&2
  fi
fi
# Chromium fornecido por Snap pode não arrancar sob o utilizador nordai
# quando o serviço systemd tem ProtectHome=true e um HOME não convencional.
CHROMIUM_BIN="$(command -v chromium || command -v chromium-browser || true)"
if [ -n "$CHROMIUM_BIN" ]; then
  case "$(readlink -f "$CHROMIUM_BIN" 2>/dev/null || true)" in
    /snap/*|/usr/bin/snap)
      echo "AVISO: Chromium usa Snap. Se aparecer código 46, verifica stderr no monitor e journalctl." >&2
      ;;
  esac
fi
echo "[2/6] Conta e diretórios"
if ! id nordai >/dev/null 2>&1; then useradd --system --home /var/lib/nord-ai --shell /usr/sbin/nologin nordai; fi
install -d -m 750 -o nordai -g nordai /var/lib/nord-ai
install -d -m 755 /opt/nord-ai/server /opt/nord-ai/web
install -m 644 "$SOURCE/server/main.py" "$SOURCE/server/learning.py" "$SOURCE/server/evolution.py" "$SOURCE/server/webtools.py" "$SOURCE/server/chromium_reader.py" "$SOURCE/server/knowledge_router.py" "$SOURCE/server/github_research.py" /opt/nord-ai/server/
install -m 644 "$SOURCE/web/index.html" "$SOURCE/web/style.css" "$SOURCE/web/app.js" /opt/nord-ai/web/
chown -R root:root /opt/nord-ai
echo "[3/6] Rede local e credenciais"
LAN_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)' | head -n 1 || true)"
if [ -z "$LAN_IP" ]; then LAN_IP="127.0.0.1"; echo "Sem IP de LAN privada; serviço local."; fi
if [ ! -f /etc/nord-ai.env ]; then
 PASSWORD="$(openssl rand -hex 18)"
 printf 'NORD_PASSWORD=%s\nNORD_HOST=%s\nNORD_PORT=8787\nNORD_DATA=/var/lib/nord-ai\nNORD_PREFERRED_MODEL=qwen3:0.6b\nNORD_CONTEXT=2048\nNORD_TOKENS_FAST=160\nNORD_TOKENS_ANALYSIS=512\nNORD_THREADS=1\nNORD_KEEP_ALIVE=30s\n' "$PASSWORD" "$LAN_IP" > /etc/nord-ai.env
 chmod 600 /etc/nord-ai.env
 echo "PALAVRA-PASSE NORD AI: $PASSWORD"
 echo "Guarda-a num local seguro; também existe em /etc/nord-ai.env (root)."
else
 echo "A manter a palavra-passe existente em /etc/nord-ai.env."
 if [ "$LAN_IP" != "127.0.0.1" ]; then
   sed -i "s/^NORD_HOST=127[.]0[.]0[.]1$/NORD_HOST=$LAN_IP/" /etc/nord-ai.env
 fi
fi
# Anexa defaults sem alterar definições escolhidas pelo administrador.
for entry in NORD_PREFERRED_MODEL=qwen3:0.6b NORD_CONTEXT=2048 NORD_TOKENS_FAST=160 NORD_TOKENS_ANALYSIS=512 NORD_THREADS=1 NORD_KEEP_ALIVE=30s; do
 key="${entry%%=*}"
 if ! grep -q "^${key}=" /etc/nord-ai.env; then printf '%s\n' "$entry" >> /etc/nord-ai.env; fi
done
# Atualiza o perfil anterior sem substituir personalizações do administrador.
if grep -qx 'NORD_CONTEXT=1024' /etc/nord-ai.env; then
 sed -i 's/^NORD_CONTEXT=1024$/NORD_CONTEXT=2048/' /etc/nord-ai.env
fi
if grep -qx 'NORD_TOKENS_ANALYSIS=256' /etc/nord-ai.env; then
 sed -i 's/^NORD_TOKENS_ANALYSIS=256$/NORD_TOKENS_ANALYSIS=512/' /etc/nord-ai.env
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
Environment="OLLAMA_CONTEXT_LENGTH=2048"
Environment="OLLAMA_KEEP_ALIVE=30s"
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
# O Chromium Snap exige XDG_RUNTIME_DIR=/run/user/<uid> gravável pelo utilizador.
# Criar apenas o diretório do serviço, sem alterar permissões de /run/user.
NORD_UID="$(id -u nordai)"
if [ -z "$NORD_UID" ]; then echo "Utilizador nordai inexistente" >&2; exit 1; fi
sed -i "/^ExecStart=\/usr\/bin\/python3/i Environment=XDG_RUNTIME_DIR=/run/user/$NORD_UID\nExecStartPre=+/usr/bin/install -d -m 0700 -o nordai -g nordai /run/user/$NORD_UID" /etc/systemd/system/nord-ai.service
echo "[Chromium] XDG_RUNTIME_DIR configurado para /run/user/$NORD_UID"
systemctl daemon-reload
systemctl enable --now ollama
systemctl restart ollama
for i in $(seq 1 30); do
 if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then break; fi
 sleep 2
done
echo "[6/6] Modelo preferido: qwen3:0.6b"
if ! ollama list | grep -q '^qwen3:0.6b '; then
 ollama pull qwen3:0.6b
fi
systemctl enable --now nord-ai
systemctl restart nord-ai
echo
echo "Nord AI instalado."
echo "Estado: systemctl status nord-ai"
ACTIVE_IP="$(sed -n 's/^NORD_HOST=//p' /etc/nord-ai.env | tail -n 1)"
echo "Interface: http://$ACTIVE_IP:8787"
echo "Autoriza TCP 8787 apenas da sub-rede LAN na firewall, se necessário."
echo "Ollama continua restrito a 127.0.0.1:11434."
echo "Não configures port forwarding no router."
