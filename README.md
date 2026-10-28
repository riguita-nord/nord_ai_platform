# Nord AI Platform

Assistente local minimalista para Linux com 2 GB RAM e 1 vCPU.

## Instalar (Ubuntu/Debian com systemd)

```bash
git clone https://github.com/riguita-nord/nord_ai_platform.git
cd nord_ai_platform
sudo bash install.sh
```

O instalador instala Python, Ollama, descarrega `gemma3:270m`, cria o serviço `nord-ai` e gera credenciais em `/etc/nord-ai.env`. **Guarda a palavra-passe apresentada pelo instalador.**

Por segurança a interface fica ligada a `127.0.0.1:8787`. Do teu computador, cria um túnel SSH:

```bash
ssh -L 8787:127.0.0.1:8787 UTILIZADOR@IP_DA_VM
```

Depois abre `http://127.0.0.1:8787`. Para acesso remoto permanente, configura um reverse proxy HTTPS autenticado; não exponhas a API do Ollama na Internet.

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
