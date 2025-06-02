import os
import requests
import sqlite3
import time
import logging
from datetime import datetime
from fastapi import FastAPI, Request, Query
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== ENV ====
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://oprojects.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "changeme_secret_key_123456")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]  # Edit sesuai kebutuhan

ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000"]

# ==== FASTAPI APP ====
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
)

# ==== GOOGLE OAUTH ====
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get("/auth/google")
async def login_via_google(request: Request):
    redirect_uri = request.url_for('auth_google_callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
        print("TOKEN:", token)
        user = None
        if token and "id_token" in token:
            user = await oauth.google.parse_id_token(request, token)
        else:
            resp = await oauth.google.get('userinfo', token=token)
            user = resp.json()
        print("USER:", user)
        email = user.get("email", "") if user else ""
        if not email:
            logger.error("Google OAuth: Email not found in user profile.")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_no_email")
        response = RedirectResponse(url=f"{FRONTEND_URL}/?email={email}")
        return response
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        response = RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_failed")
        return response

# ==== DATABASE (sqlite) ====
def init_db():
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    credits INTEGER,
                    login_streak INTEGER,
                    last_login TEXT,
                    last_guest_timestamp INTEGER,
                    last_reward_date TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TEXT
                )''')
    conn.commit()
    conn.close()
init_db()

def check_credits(user_id, need=1):
    if not user_id:
        return False
    if user_id in ADMIN_USERS:
        return True
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    if not result or result[0] < need:
        conn.close()
        return False
    c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
    conn.commit()
    conn.close()
    return True

def get_credits(user_id):
    if not user_id:
        return "0"
    if user_id in ADMIN_USERS:
        return "∞"
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return str(result[0]) if result else "0"

def add_or_init_user(user_id, user_name="User"):
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (user_id, user_name, 25, 0, datetime.now().strftime("%Y-%m-%d"), int(time.time()), ''))
    conn.commit()
    conn.close()

def save_chat_history(user_id, question, answer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
        (user_id, question, answer, now)
    )
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=20):
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute(
        "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    # Kembalikan dari terbaru ke terlama (DESC)
    return [
        {"question": row[0], "answer": row[1], "created_at": row[2]}
        for row in rows
    ][::-1]

# ==== API CHAT - OpenRouter ====
class ChatRequest(BaseModel):
    user_email: str
    message: str
    model_select: str = "x-ai/grok-3-mini-beta"  # default

@app.post("/api/chat")
async def ai_chat(req: ChatRequest):
    user_id = req.user_email
    if not user_id:
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
    add_or_init_user(user_id, user_id)
    if not check_credits(user_id, 1):
        return JSONResponse({"error": "NOT_ENOUGH_CREDITS"}, status_code=402)
    # ==== OpenRouter API call ====
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kugy.ai",
        "X-Title": "KugyAI"
    }
    model_map = {
        "OpenRouter (Grok 3 Mini Beta)": "x-ai/grok-3-mini-beta",
        "OpenRouter (Gemini 2.0 Flash)": "google/gemini-flash-1.5"
    }
    model_id = model_map.get(req.model_select, "x-ai/grok-3-mini-beta")
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Act as an assistant."},
            {"role": "user", "content": req.message}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return JSONResponse({"error": f"OpenRouter error: {response.text}"}, status_code=500)
        reply = response.json()["choices"][0]["message"]["content"]
        credits = get_credits(user_id)
        save_chat_history(user_id, req.message, reply)
        return {"reply": reply, "credits": credits}
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return JSONResponse({"error": "AI Service Unavailable"}, status_code=503)

# ==== GET CREDIT API ====
@app.get("/api/credits")
async def api_credits(user_email: str):
    credits = get_credits(user_email)
    return {"credits": credits}

# ==== GET HISTORY CHAT API ====
@app.get("/api/history")
async def api_history(user_email: str = Query(...), limit: int = Query(20, le=100)):
    if not user_email:
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
    history = get_chat_history(user_email, limit)
    return {"history": history}

# ==== HEALTH CHECK ====
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

# ==== LOCAL RUN ====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)