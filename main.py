import os
import logging
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "vector")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # Render vai te dar depois

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector")

app = FastAPI()
tg_app = Application.builder().token(BOT_TOKEN).build()

def telegram_safe(text: str) -> str:
    # Telegram-friendly: sem LaTeX
    text = text.replace("$", "")
    if len(text) > 3500:
        text = text[:3500] + "…"
    return text

async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = update.message
        if not msg:
            return

        if msg.text:
            user_text = msg.text.strip()
            await msg.reply_text(telegram_safe(f"Recebi: {user_text}\n(Infra ok ✅)"))

        elif msg.photo:
            await msg.reply_text("Imagem recebida ✅\n(Infra ok ✅)\nVou ler a imagem no próximo passo.")

        else:
            await msg.reply_text("Me manda texto ou foto/print da questão 🙂")

    except Exception:
        logger.exception("Falha geral no handler")
        if update and update.message:
            await update.message.reply_text("Deu um erro aqui do meu lado. Tenta de novo agora?")

tg_app.add_handler(MessageHandler(filters.ALL, on_message))

@app.on_event("startup")
async def startup():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN vazio")
        return

    await tg_app.initialize()
    await tg_app.start()

    # Só tenta setar webhook se o PUBLIC_URL for válido
    if PUBLIC_URL and PUBLIC_URL.startswith("https://"):
        url = f"{PUBLIC_URL}/webhook/{WEBHOOK_SECRET}"
        try:
            await tg_app.bot.set_webhook(url=url)
            logger.info(f"Webhook setado: {url}")
        except Exception:
            logger.exception("Falha ao setar webhook (vou subir mesmo assim)")
    else:
        logger.warning("PUBLIC_URL inválido/ausente. Subindo sem webhook por enquanto.")


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
