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
from PIL import Image
from io import BytesIO
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment Variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://front-end-bpup.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "changeme_secret_key_123456")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]

ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app"
]

app = FastAPI()

# CORS Middleware dengan konfigurasi yang diperbarui
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=3600,
)

# Google OAuth setup
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'state': lambda: os.urandom(16).hex()
    }
)

@app.get("/auth/google")
async def login_via_google(request: Request):
    redirect_uri = request.url_for('auth_google_callback')
    state = os.urandom(16).hex()
    request.session['oauth_state'] = state
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    try:
        expected_state = request.session.get('oauth_state')
        if not expected_state:
            logger.error("No state in session")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=no_state")
        
        token = await oauth.google.authorize_access_token(request)
        actual_state = request.query_params.get('state')
        if actual_state != expected_state:
            logger.error(f"mismatching_state: Expected {expected_state}, got {actual_state}")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=csrf_warning")
        
        del request.session['oauth_state']

        user = None
        email = ""
        if "userinfo" in token and token["userinfo"]:
            user = token["userinfo"]
        elif token and "id_token" in token:
            user = await oauth.google.parse_id_token(request, token)
        else:
            resp = await oauth.google.get('userinfo', token=token)
            user = resp.json()

        if user:
            email = user.get("email", "")
        if not email:
            logger.error("Google OAuth: Email not found in user profile.")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_no_email")
        return RedirectResponse(url=f"{FRONTEND_URL}/menu?email={email}")
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_failed")

# Database functions
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
    is_email = "@" in user_id
    default_credits = 75 if is_email else 25
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (user_id, user_name, default_credits, 0, datetime.now().strftime("%Y-%m-%d"), int(time.time()), ''))
    conn.commit()
    conn.close()

def save_chat_history(user_id, question, answer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:S")
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
    return [
        {"question": row[0], "answer": row[1], "created_at": row[2]}
        for row in rows
    ][::-1]

# API Models
class ChatRequest(BaseModel):
    user_email: str
    message: str
    model_select: str = "x-ai/grok-3-mini-beta"

class ImageRequest(BaseModel):
    user_email: str
    prompt: str

# API Endpoints
@app.post("/api/chat")
async def ai_chat(req: ChatRequest):
    user_id = req.user_email
    if not user_id:
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
    add_or_init_user(user_id, user_id)
    if not check_credits(user_id, 1):
        return JSONResponse({"error": "NOT_ENOUGH_CREDITS"}, status_code=402)
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

@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    """Generate image using Stability AI API"""
    cors_headers = {
        "Access-Control-Allow-Origin": "https://front-end-bpup.vercel.app",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Credentials": "true"
    }

    if not req.user_email:
        return JSONResponse(
            {"error": "UNAUTHORIZED"}, 
            status_code=401,
            headers=cors_headers
        )

    if not check_credits(req.user_email, 10):
        return JSONResponse(
            {"error": "NOT_ENOUGH_CREDITS", "message": "Need 10 credits"}, 
            status_code=402,
            headers=cors_headers
        )

    if not STABILITY_API_KEY:
        return JSONResponse(
            {"error": "API_KEY_MISSING", "message": "Stability AI API key not set"}, 
            status_code=503,
            headers=cors_headers
        )

    stability_headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text_prompts": [{"text": req.prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30
    }

    try:
        response = requests.post(
            STABILITY_API_URL,
            headers=stability_headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            return JSONResponse(
                {"error": f"Stability AI error: {response.text}"}, 
                status_code=500,
                headers=cors_headers
            )

        resp_data = response.json()
        if "artifacts" in resp_data and resp_data["artifacts"]:
            base64_img = resp_data["artifacts"][0]["base64"]
            credits = get_credits(req.user_email)
            return JSONResponse({
                "image": base64_img,
                "credits": credits,
                "message": "Kugy.ai: Ini gambarnya buat ayang, cute banget kan? 🐱"
            }, headers=cors_headers)

        return JSONResponse(
            {"error": "Failed to get image from Stability AI"}, 
            status_code=500,
            headers=cors_headers
        )

    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return JSONResponse(
            {"error": f"Error generating image: {str(e)}"}, 
            status_code=500,
            headers=cors_headers
        )

@app.options("/api/generate-image")
async def generate_image_options():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "https://front-end-bpup.vercel.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@app.post("/api/guest-login")
async def guest_login(request: Request):
    data = await request.json()
    user_email = data.get("email")
    if not user_email:
        return JSONResponse({"error": "Email wajib diisi"}, status_code=400)

    add_or_init_user(user_email, "Guest")
    credits = get_credits(user_email)
    dummy_token = f"guest-token-{user_email.split('@')[0]}"
    return JSONResponse({
        "token": dummy_token,
        "credits": credits,
        "message": "Kugy.ai: Mode tamu aktif! 😺"
    })

@app.get("/api/credits")
async def api_credits(user_email: str):
    credits = get_credits(user_email)
    return {"credits": credits}

@app.get("/api/history")
async def api_history(user_email: str = Query(...), limit: int = Query(20, le=100)):
    if not user_email:
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401)
    history = get_chat_history(user_email, limit)
    return {"history": history}

@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
