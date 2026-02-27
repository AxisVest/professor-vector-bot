# Professor Vector Bot

Este é o repositório do Professor Vector, um bot de Telegram que atua como tutor de Matemática focado no ENEM, utilizando a API do Google Gemini para gerar respostas adaptativas e personalizadas.

## Funcionalidades

*   **Interação via Telegram:** Recebe mensagens e comandos através do Telegram.
*   **Geração de Respostas:** Utiliza a API do Google Gemini para gerar explicações e orientações matemáticas.
*   **Histórico de Conversa:** Mantém o contexto da conversa por usuário para respostas mais coerentes.
*   **Visão Computacional:** Suporta o envio de imagens (fotos de questões do ENEM) para interpretação pelo Gemini.
*   **Comandos Essenciais:** Inclui os comandos `/start`, `/reset` e `/ajuda` para iniciar, reiniciar a interação e obter informações.
*   **Formato Pedagógico:** Respostas curtas, fluidas, diretas e sem notação LaTeX, priorizando a autonomia do aluno.
*   **Health Check:** Endpoint `/healthz` para verificar o status do serviço.

## Configuração e Deploy no Render.com

Para configurar e fazer o deploy do Professor Vector no Render.com, siga os passos abaixo:

### 1. Variáveis de Ambiente

As seguintes variáveis de ambiente são **obrigatórias** para o funcionamento do bot. Elas devem ser configuradas no seu serviço Render:

*   `TELEGRAM_BOT_TOKEN`: O token do seu bot do Telegram. Você pode obtê-lo com o BotFather no Telegram.
*   `GEMINI_API_KEY`: Sua chave de API do Google Gemini. Obtenha-a no [Google AI Studio](https://aistudio.google.com/app/apikey).
*   `PUBLIC_URL`: A URL pública do seu serviço no Render. Por exemplo: `https://professor-vector-bot.onrender.com`.
*   `WEBHOOK_SECRET`: Uma string secreta de sua escolha para proteger o webhook do Telegram. **É crucial que esta string seja a mesma configurada no código do bot.**
*   `GEMINI_MODEL`: O modelo Gemini a ser utilizado. Recomenda-se `gemini-2.5-flash` ou `gemini-1.5-flash` como fallback. O bot tentará usar `gemini-2.5-flash` primeiro.
*   `LOG_LEVEL`: Nível de log para o bot (ex: `INFO`, `DEBUG`, `WARNING`).

### 2. Comando de Inicialização (Start Command)

No Render, o comando de inicialização para o seu serviço deve ser:

```bash
uvicorn bot:app --host 0.0.0.0 --port $PORT
```

Este comando inicia o servidor Uvicorn que hospeda a aplicação FastAPI (que por sua vez gerencia o webhook do Telegram).

### 3. Dependências

As dependências do projeto estão listadas no arquivo `requirements.txt`. O Render as instalará automaticamente ao detectar este arquivo na raiz do seu repositório.

### 4. Estrutura do Projeto

Certifique-se de que os arquivos `bot.py`, `requirements.txt` e `README.md` estejam na raiz do seu repositório GitHub.

### 5. Deploy

1.  **Crie um novo serviço web no Render:** Conecte seu repositório GitHub.
2.  **Configure as variáveis de ambiente:** Adicione todas as variáveis listadas na seção 1.
3.  **Defina o comando de inicialização:** Use o comando fornecido na seção 2.
4.  **Deploy:** O Render fará o deploy automaticamente. Após o deploy, a URL pública do seu serviço será o valor de `PUBLIC_URL`.

### 6. Configuração do Webhook no Telegram

O bot configurará o webhook automaticamente na inicialização, usando a `PUBLIC_URL` e `WEBHOOK_SECRET` que você forneceu. Certifique-se de que a `PUBLIC_URL` esteja correta e acessível publicamente.

---

**Desenvolvido por Manus AI**
