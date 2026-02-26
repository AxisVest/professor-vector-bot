import os
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from google import genai
from google.genai import types

load_dotenv()

# Telegram
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "vector")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # ex: https://professor-vector-bot.onrender.com

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector")

app = FastAPI()
tg_app = Application.builder().token(BOT_TOKEN).build()

# Cliente Gemini (SDK novo)
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def telegram_safe(text: str) -> str:
    # Telegram-friendly: sem LaTeX
    if not text:
        return "Não consegui gerar uma resposta agora."
    text = text.replace("$", "")
    if len(text) > 3500:
        text = text[:3500] + "…"
    return text


SYSTEM_STYLE = (
    "Você é o Professor Vector, tutor de Matemática para o ENEM.\n"
    "Responda SEM LaTeX. Escreva fórmulas em texto simples (ex: a + b = 10).\n"
    "Explique passo a passo e, a cada etapa, faça uma pergunta curta para checar entendimento.\n"
    "Se o aluno disser que não sabe por onde começar, ensine como iniciar.\n"
)


async def ask_gemini_text(user_text: str) -> str:
    if not gemini_client:
        return "Configuração do Gemini não encontrada. (GEMINI_API_KEY vazia)"
    prompt = f"{SYSTEM_STYLE}\nAluno: {user_text}\nProfessor Vector:"
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt],
    )
    return getattr(resp, "text", "") or "Não consegui gerar resposta."


async def ask_gemini_image(image_bytes: bytes, mime_type: str, user_text: str | None) -> str:
    if not gemini_client:
        return "Configuração do Gemini não encontrada. (GEMINI_API_KEY vazia)"

    # A doc oficial mostra envio de bytes via types.Part.from_bytes(...) no contents. :contentReference[oaicite:2]{index=2}
    parts = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
    ]

    text = (user_text or "").strip()
    if not text:
        text = "Resolva a questão do print/foto. Explique passo a passo e sem LaTeX."

    prompt = f"{SYSTEM_STYLE}\nTarefa: {text}\nProfessor Vector:"
    parts.append(prompt)

    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=parts,
    )
    return getattr(resp, "text", "") or "Não consegui interpretar a imagem."


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return

    try:
        # Texto
        if msg.text and not msg.photo:
            user_text = msg.text.strip()

            # Comandos simples (opcional)
            if user_text.lower() in ("/start", "start"):
                await msg.reply_text("Olá! Me diga seu nome e mande sua dúvida (pode ser foto/print).")
                return

            answer = await ask_gemini_text(user_text)
            await msg.reply_text(telegram_safe(answer))
            return

        # Foto/print
        if msg.photo:
            # pega a melhor resolução
            photo = msg.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            image_bytes = await file.download_as_bytearray()

            # Telegram manda geralmente JPEG
            caption = (msg.caption or "").strip()
            answer = await ask_gemini_image(bytes(image_bytes), "image/jpeg", caption)
            await msg.reply_text(telegram_safe(answer))
            return

        await msg.reply_text("Me manda texto ou foto/print da questão 🙂")

    except Exception:
        logger.exception("Falha geral no handler")
        await msg.reply_text("Deu um erro aqui do meu lado. Tenta de novo agora?")


tg_app.add_handler(MessageHandler(filters.ALL, on_message))


@app.on_event("startup")
async def startup():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN vazio")
        return

    await tg_app.initialize()
    await tg_app.start()

    # Webhook: não pode derrubar o deploy se PUBLIC_URL estiver errado
    if PUBLIC_URL and PUBLIC_URL.startswith("https://"):
        try:
            url = f"{PUBLIC_URL}/webhook/{WEBHOOK_SECRET}"
            await tg_app.bot.set_webhook(url=url)
            logger.info(f"Webhook setado: {url}")
        except Exception:
            logger.exception("Falha ao setar webhook (continuando sem derrubar o app).")
    else:
        logger.warning("PUBLIC_URL vazio/ inválido. Webhook não foi setado.")


@app.on_event("shutdown")
async def shutdown():
    await tg_app.stop()
    await tg_app.shutdown()


@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}


@app.get("/health")
async def health():
    return {"ok": True}
