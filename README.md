# Nord AI Platform

Assistente local minimalista para Linux com 2 GB RAM e 1 vCPU.

## Instalar (Ubuntu/Debian com systemd)

```bash
git clone https://github.com/riguita-nord/nord_ai_platform.git
cd nord_ai_platform
sudo bash install.sh
```

O instalador instala Python, Ollama, descarrega `gemma3:270m`, cria o serviço `nord-ai` e gera credenciais em `/etc/nord-ai.env`. **Guarda a palavra-passe apresentada pelo instalador.**

## Acesso pela rede local

O instalador deteta um IP privado da VM (exemplo `192.168.1.50`) e liga o painel a esse endereço, na porta **8787**. Num telemóvel ou computador com acesso à mesma LAN, abre:

```text
http://192.168.1.50:8787
```

Substitui pelo IP mostrado pelo instalador. A API do Ollama continua apenas em `127.0.0.1:11434`.

Se a firewall UFW estiver ativa, autoriza somente a tua sub-rede (exemplo, adaptar à tua rede):

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8787 proto tcp
```

Se o instalador não encontrar IP privado, mantém `127.0.0.1`. Para consultar o endereço: `hostname -I`. Para atualizar uma instalação existente, executa `git pull && sudo bash install.sh` (a palavra-passe existente é mantida).

**Não abras a porta 8787 na Internet, em port forwarding, ou em regras de firewall de cloud que aceitem todo o mundo.** O login em HTTP na LAN não é encriptado; para redes com dispositivos não confiáveis ou acesso remoto, instala HTTPS com reverse proxy ou VPN (ex.: WireGuard/Tailscale). Se a VM for uma VPS sem ligação à tua LAN, um IP privado da VPS não significa que seja acessível pela rede de casa: precisas de VPN.

## Funções
- Login obrigatório, sessão em cookie HttpOnly SameSite=Strict
- Chat com histórico SQLite e respostas progressivas
- Seleção de modelos instalados
- Botão **+** para instalar modelos aprovados, visualização de progresso, remoção e ativação
- Limite de uma geração em simultâneo, contexto reduzido e CPU conservadora
- Serviço systemd

## Operação

```bash
sudo systemctl status nord-ai
sudo journalctl -u nord-ai -f
sudo systemctl restart nord-ai
ollama list
```

## Notas de capacidade

Com 2 GB e 1 vCPU, começa com `gemma3:270m`. Os modelos maiores poderão ficar lentos ou esgotar memória; o instalador limita o Ollama a 1 modelo carregado, 1 pedido em paralelo e contexto de 1024 tokens. A funcionalidade de instalação de modelos só aceita uma lista pequena de identificadores conhecidos e exige login. O download precisa de acesso à Internet. Não está incluído proxy HTTPS, gestão multiutilizador ou recuperação de senha.

## Atualizar

```bash
cd nord_ai_platform && git pull
sudo cp -r server web /opt/nord-ai/
sudo systemctl restart nord-ai
```

Licença: uso interno Nord-Lab.
