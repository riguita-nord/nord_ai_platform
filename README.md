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

## Nord Learning Center (v1.1)

No menu lateral abre **Learning Center**. Escolhe uma das áreas **Programação, Inteligência artificial, Tecnologia, Ciência, Escrita criativa, FiveM / QBCore** e pesquisa um assunto. A pesquisa é **exclusivamente manual**, nunca agendada. A aprovação está configurada como **automática**: conteúdos retornados pela fonte permitida ficam disponíveis na memória logo após a pesquisa. Podes remover cada registo pelo painel.

- Fonte inicial estritamente autorizada: API HTTPS da Wikipédia em português (até cinco resultados por pesquisa). Isto NÃO é pesquisa livre por toda a Internet. FiveM/QBCore e outros temas técnicos podem ter cobertura limitada.
- Consulta com prazo de 8 segundos, limite de 100 KB de resposta, excertos de 2800 caracteres e máximo de 800 registos.
- URLs não são fornecidos pelo cliente ao fetch remoto; não são aceites redirecionamentos; não existe crawler ou execução de código externo.
- A biblioteca usa **SQLite FTS5**, com excertos aprovados automaticamente e referências recuperáveis em conversas futuras.
- Fontes são dados não fiáveis: o chat recebe aviso para nunca obedecer a instruções incluídas nas fontes. **A revisão automática não garante veracidade** nem aumenta a inteligência do modelo; avalia cuidadosamente o conteúdo antes de o usar.
- O motor usa no máximo um pedido de pesquisa externo simultâneo. Não há processos recorrentes nem treino dos pesos.

### Atualização da VM

```bash
cd nord_ai_platform
git pull
sudo bash install.sh
sudo systemctl status nord-ai
```

O instalador copia `server/learning.py`, atualiza o frontend e reinicia o serviço. Mantém a palavra-passe existente.

### Verificar a funcionalidade

Abre Learning Center, pesquisa **Python** na área Programação, confirma um resultado com ligação à fonte e depois faz uma pergunta sobre esse assunto no chat. Consulta os logs: `sudo journalctl -u nord-ai -n 80 --no-pager`.

**Segurança da rede:** HTTP na LAN não cifra palavras-passe. Não abras portas na Internet pública; considera HTTPS ou VPN numa rede com equipamentos não confiáveis.

## Nord-Qwen: modo automático e perfil de recursos

O Qwen3 0.6B é o modelo preferido. O chat escolhe entre geração rápida e modo de análise por critérios leves; isto é uma heurística e não uma avaliação de inteligência. A VM atual utiliza o perfil conservador **Low Resource** (1 vCPU/2 GB). Os parâmetros do modelo não são permanentes: edita `/etc/nord-ai.env` no futuro e reinicia o serviço.

```ini
NORD_PREFERRED_MODEL=qwen3:0.6b
NORD_CONTEXT=1024
NORD_TOKENS_FAST=160
NORD_TOKENS_ANALYSIS=256
NORD_THREADS=1
NORD_KEEP_ALIVE=30s
```

No futuro, quando migrares para uma máquina com GPU, poderás aumentar contexto/tokens e instalar outro modelo, sem mudar os pesos ou reescrever o chat. Os limites do **Ollama** são separados, no ficheiro `/etc/systemd/system/ollama.service.d/nord-limits.conf`: ajusta `MemoryMax`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL` e `OLLAMA_MAX_LOADED_MODELS` conforme o hardware; depois executa `sudo systemctl daemon-reload && sudo systemctl restart ollama nord-ai`. O instalador atual é orientado para a VM de 2 GB e repõe esse perfil ao ser executado novamente.

## Respostas interrompidas e estado de geração

O chat agora mostra indicadores reais (`A preparar resposta`, `A consultar memória`, `A analisar`, `A responder`, `A continuar a resposta`). **Não existe pesquisa automática no chat nesta versão**, pelo que a indicação diz explicitamente `Sem pesquisa na Internet`; o Learning Center manual é separado.

Se o Ollama comunicar `done_reason=length`, o servidor tenta continuar a geração até mais duas vezes, sem repetir o prefixo. Se continuar incompleta ou o fluxo terminar sem confirmação, a interface informa em vez de simular que terminou. A quantidade de tokens e o contexto continuam configuráveis em `/etc/nord-ai.env`. Com 2 GB e 1 vCPU as continuações podem ser lentas; não são uma garantia de completar qualquer resposta.

Para aplicar: `git pull && sudo bash install.sh`; atualiza a página com Ctrl+F5.

## Correção de streaming e pesquisa automática (29/10/2026)

- Corrigida a serialização NDJSON: cada evento do chat termina agora com uma mudança de linha real (antes eram os caracteres literais `\\n`, causando erro de JSON no browser).
- Antes da resposta, o Qwen3 0.6B pode decidir entre `WEB` e `LOCAL` numa consulta curta. Apenas quando decide `WEB` é executada uma pesquisa.
- A pesquisa nesta versão consulta **somente a API pública da Wikipédia em português**, via `learning.discover`, com os limites de origem, tamanho, tempo e número de resultados já definidos. Não é ainda um motor de pesquisa geral.
- O estado exibido no chat distingue preparação, decisão de pesquisa, pesquisa real, análise das fontes, resposta e continuação. Não apresenta pesquisa quando nenhuma aconteceu.
- Se a fonte falhar, o modelo continua a responder com as limitações indicadas. A decisão automática de pesquisar e a qualidade das fontes não são garantidas num modelo de 0.6B.
- Para atualizar: `cd nord_ai_platform && git pull && sudo bash install.sh`. Recarrega a página com Ctrl+F5.

## Pesquisa decidida pelo Qwen (modelo como investigador)

A sequência de pesquisa agora é: Qwen escolhe **responder ou pesquisar**, formula a sua própria consulta, recebe até cinco resultados de pesquisa e escolhe até três referências para orientar a resposta. Os indicadores mostram a consulta real e os títulos selecionados. A ferramenta externa limita downloads e tempo, e não visita nem executa URLs dos resultados.

O adaptador inicial usa **Bing RSS** para pesquisa geral, sem chave de API. Esta fonte pode mudar, bloquear pedidos ou devolver resultados incompletos. Nesta versão os resultados são **títulos e resumos**, não leitura integral das páginas. O Qwen pode escolher mal as fontes, especialmente com apenas 0.6B parâmetros. Não é correto afirmar que investigou o conteúdo integral ou que confirmou factos apenas pelos resumos. A pesquisa não altera os pesos do modelo.

Acessos continuam autenticados no painel. Para instalar o módulo `server/webtools.py`, atualiza com `git pull && sudo bash install.sh`. A VM continua com os limites anteriores.

## Investigação contextual e leitura de fontes (29/10/2026)

O Qwen é instruído a escolher consulta e fontes; pedidos explícitos de pesquisa forçam a pesquisa mesmo que a classificação do modelo falhe. Pedidos curtos de seguimento, como `faz isso`, herdam o tema recente. A pesquisa usa resultados do Bing RSS e tenta abrir até duas páginas HTTPS públicas selecionadas para retirar texto, sem executar JavaScript. São aplicados limites de tempo e tamanho, sem seguir redirecionamentos, com bloqueio de destinos privados/reservados. Para segurança reforçada contra DNS rebinding, aplica também restrições de egress ao nível da rede.

O indicador mostra `A pesquisar`, `A ler as páginas escolhidas` e `A analisar fontes` só quando essas ações acontecem. Se o motor de busca estiver indisponível, o chat indica a falha e o modelo deve admitir que não conseguiu verificar. Os motores de busca podem bloquear o endpoint RSS ou devolver cobertura incompleta; isto não é garantia de acesso a todos os sites nem de respostas sempre detalhadas. A VM de 2 GB continua limitada no tamanho do contexto e na velocidade.

Aplicar: `cd nord_ai_platform && git pull && sudo bash install.sh`, e depois `Ctrl+F5` no browser.

## Qwen3 0.6B — perfil de investigação (29/10/2026)

- Temperatura de geração quando pesquisa: **1.0** (escolha explícita do administrador; temperaturas elevadas podem produzir mais variação factual).
- Contexto por pedido: **2048 tokens**, geração até **512 tokens por etapa** de investigação.
- O Qwen escolhe a consulta e escolhe até duas fontes para ler. Se não houver páginas legíveis, tenta **uma segunda pesquisa** e apresenta estado específico.
- Um resumo do motor de busca não basta para marcar conteúdo como evidência; só páginas efetivamente lidas entram no contexto de investigação.
- Fontes externas são tratadas como dados não fiáveis, sem execução de scripts ou instruções das páginas. O processo não comprova automaticamente a veracidade das fontes; o modelo deve citar URLs e admitir incertezas.
- Os limites da VM permanecem: **1 vCPU**, **uma geração em simultâneo**, até **1500 MB** no serviço Ollama, respostas com tamanho limitado, sessões de modelo com keep-alive curto. Os valores podem ser reduzidos em `/etc/nord-ai.env` se houver pressão de memória.
- Nota: ao atualizar o instalador, `NORD_CONTEXT=1024` e `NORD_TOKENS_ANALYSIS=256` de perfis anteriores são migrados para os novos valores. Personalizações diferentes são mantidas.

Reinstala com `git pull && sudo bash install.sh` e faz `Ctrl+F5`. Verifica o consumo de memória com `free -h` e `systemctl status ollama nord-ai`. Se ocorrer OOM, reduz `NORD_CONTEXT` para 1024 e `OLLAMA_CONTEXT_LENGTH` para 1024 no drop-in systemd.
