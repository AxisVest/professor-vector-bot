import os
import re
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import (
    Application,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)

import google.generativeai as genai

load_dotenv()

# =========================
# ENV
# =========================
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "vector").strip()
PUBLIC_URL = os.getenv("PUBLIC_URL", "").strip()  # ex: https://professor-vector-bot.onrender.com
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()  # pode trocar depois

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("vector")

# =========================
# FASTAPI + TELEGRAM APP
# =========================
app = FastAPI()
tg_app = Application.builder().token(BOT_TOKEN).build()

# =========================
# STATE (Modo C)
# =========================
TRAVADO = "TRAVADO"
PRESSA = "PRESSA"
AUTONOMO = "AUTONOMO"

PRESSA_WORDS = [
    "resposta", "alternativa", "letra", "gabarito", "rápido", "rapido",
    "logo", "agora", "pressa", "corrige", "qual é"
]
TRAVADO_WORDS = [
    "não sei", "nao sei", "não entendi", "nao entendi", "travado",
    "socorro", "me ajuda", "não consigo", "nao consigo", "nada", "perdido"
]
AUTONOMO_HINTS = [
    "acho", "então", "entao", "porque", "pois", "logo", "daí", "dai", "=",
    "+", "-", "x", "*", "/", ">", "<"
]

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def telegram_safe(text: str) -> str:
    # Sem LaTeX e sem estourar limite
    text = (text or "").replace("$", "")
    if len(text) > 3500:
        text = text[:3500] + "…"
    return text

@dataclass
class UserState:
    mode: str = AUTONOMO
    score_travado: int = 0
    score_pressa: int = 0
    score_autonomo: int = 0
    history: List[Tuple[str, str]] = field(default_factory=list)  # (role, text)
    last_message_ids: Set[int] = field(default_factory=set)

USER: Dict[int, UserState] = {}

def get_user_state(user_id: int) -> UserState:
    if user_id not in USER:
        USER[user_id] = UserState()
    return USER[user_id]

def update_scores(st: UserState, text: str) -> None:
    t = normalize_text(text)

    # heurísticas simples e robustas
    if any(w in t for w in TRAVADO_WORDS) or len(t) <= 2:
        st.score_travado += 2

    if any(w in t for w in PRESSA_WORDS):
        st.score_pressa += 2

    if any(w in t for w in AUTONOMO_HINTS) and len(t) >= 8:
        st.score_autonomo += 2

def decide_mode(st: UserState) -> str:
    # prioridade: travado > pressa > autonomo (bem estável)
    if st.score_travado >= 2:
        return TRAVADO
    if st.score_pressa >= 2:
        return PRESSA
    return AUTONOMO

def clamp_history(st: UserState, max_items: int = 8) -> None:
    if len(st.history) > max_items:
        st.history = st.history[-max_items:]

def system_prompt_for_mode(mode: str) -> str:
    # MODO C (Inteligente) — texto enxuto e estilo certo
    base = """
Você é o Professor Vector, tutor de matemática ENEM do Prof. Wisner.
Tom: humano, direto, motivador, sem enrolação.
Sempre em português do Brasil.
Nunca use LaTeX. Fórmulas sempre em texto simples (ex: "a + b = 17").
Se precisar de símbolos, use Unicode comum (≤, ≥, √, etc) com parcimônia.
Evite textos enormes: 1 a 3 bullets por mensagem no máximo.
"""

    modo = f"""
Você deve adaptar automaticamente a resposta conforme o perfil do aluno, sem revelar o perfil.

PERFIL ATUAL: {mode}

Regras por perfil:

TRAVADO:
- Quebre em micro-passos.
- Faça 1 pergunta curta por vez.
- Confirme entendimento com "Fez sentido?" ou "Até aqui ok?"
- Evite parágrafos longos.

PRESSA:
- Dê primeiro a resposta final (resultado ou alternativa) em 1 linha.
- Depois explique em 3 passos bem curtos.
- Pergunte se quer detalhar passo a passo.

AUTONOMO:
- Conduza com perguntas estratégicas.
- Dê dicas e valide o raciocínio.
- Só entregue tudo completo se o aluno pedir.

Formato sugerido:
- Comece com "Vamos nessa."
- Se for PRESSA: "Resposta:" na primeira linha.
- Termine com 1 pergunta (exceto quando for extremamente direto).
"""
    return (base + "\n" + modo).strip()

# =========================
# GEMINI CALL (sync SDK -> async)
# =========================
def gemini_generate(system: str, messages: List[Tuple[str, str]]) -> str:
    """
    messages: lista de (role, text) onde role é 'user' ou 'assistant'
    """
    if not GEMINI_API_KEY:
        return "Estou sem conexão com o cérebro (API do Gemini). Me avise o administrador 🙂"

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system
    )

    # Converte histórico pro formato aceito
    contents = []
    for role, text in messages:
        if role == "assistant":
            contents.append({"role": "model", "parts": [text]})
        else:
            contents.append({"role": "user", "parts": [text]})

    resp = model.generate_content(contents)
    out = getattr(resp, "text", "") or ""
    out = out.strip()
    if not out:
        out = "Tive um branco aqui 😅. Me manda sua dúvida em 1 frase ou a alternativa que você quer confirmar."
    return out

async def gemini_generate_async(system: str, messages: List[Tuple[str, str]]) -> str:
    return await asyncio.to_thread(gemini_generate, system, messages)

# =========================
# COMMANDS
# =========================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    st = get_user_state(msg.from_user.id)
    st.score_travado = st.score_pressa = st.score_autonomo = 0
    st.mode = AUTONOMO
    st.history.clear()
    await msg.reply_text("Olá! Me diga seu nome e mande sua dúvida (pode ser texto).")

async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    st = get_user_state(msg.from_user.id)
    st.score_travado = st.score_pressa = st.score_autonomo = 0
    st.mode = AUTONOMO
    st.history.clear()
    await msg.reply_text("Beleza! Zerei a conversa. Me diga sua dúvida do ENEM 🙂")

async def cmd_ajuda(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg:
        return
    await msg.reply_text(
        "Me mande sua dúvida de matemática por texto.\n"
        "Dica: se estiver com pressa, diga 'quero só a alternativa'.\n"
        "Comandos: /start /reset /ajuda"
    )

# =========================
# MAIN HANDLER (texto)
# =========================
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        msg = update.message
        if not msg:
            return

        user_id = msg.from_user.id
        st = get_user_state(user_id)

        # dedupe simples (evita respostas repetidas)
        if msg.message_id in st.last_message_ids:
            return
        st.last_message_ids.add(msg.message_id)
        if len(st.last_message_ids) > 50:
            # mantém conjunto pequeno
            st.last_message_ids = set(list(st.last_message_ids)[-20:])

        # por enquanto: só texto
        if msg.photo:
            await msg.reply_text("Por enquanto, me mande o enunciado em TEXTO 🙂\n(Depois vamos ativar leitura de print.)")
            return

        if not msg.text:
            await msg.reply_text("Me manda sua dúvida em texto 🙂")
            return

        user_text = msg.text.strip()

        # Atualiza scores e decide modo
        update_scores(st, user_text)
        st.mode = decide_mode(st)

        # Histórico curto
        st.history.append(("user", user_text))
        clamp_history(st, max_items=8)

        system = system_prompt_for_mode(st.mode)
        answer = await gemini_generate_async(system, st.history)

        # salva resposta no histórico
        st.history.append(("assistant", answer))
        clamp_history(st, max_items=8)

        await msg.reply_text(telegram_safe(answer))

    except Exception:
        logger.exception("Falha no handler")
        if update and update.message:
            await update.message.reply_text("Deu um erro aqui do meu lado. Tenta de novo agora?")

# =========================
# ROUTES / WEBHOOK
# =========================
@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": int(time.time())}

@app.post("/webhook/{secret}")
async def webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {"ok": False}

    data = await request.json()
    update = Update.de_json(data, tg_app.bot)
    await tg_app.process_update(update)
    return {"ok": True}

@app.on_event("startup")
async def startup():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN vazio")
        return

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY vazio (o bot sobe, mas não responde com IA)")

    await tg_app.initialize()
    await tg_app.start()

    # Só seta webhook se PUBLIC_URL estiver correto (evita o erro do host)
    if PUBLIC_URL.startswith("https://"):
        url = f"{PUBLIC_URL}/webhook/{WEBHOOK_SECRET}"
        await tg_app.bot.set_webhook(url=url)
        logger.info(f"Webhook setado: {url}")
    else:
        logger.warning("PUBLIC_URL não definido (ou sem https). Webhook NÃO foi setado automaticamente.")

@app.on_event("shutdown")
async def shutdown():
    await tg_app.stop()
    await tg_app.shutdown()

# handlers
tg_app.add_handler(CommandHandler("start", cmd_start))
tg_app.add_handler(CommandHandler("reset", cmd_reset))
tg_app.add_handler(CommandHandler("ajuda", cmd_ajuda))
tg_app.add_handler(MessageHandler(filters.ALL, on_message))
