import base64
import logging
import os
import time
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import requests
import sqlite3
from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# Configure logging
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
VIRTUSIM_API_KEY = os.getenv("VIRTUSIM_API_KEY")
VIRTUSIM_API_URL = "https://virtusim.com/api/v2"

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app",
]

# Initialize FastAPI
app = FastAPI()

# Configure middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=3600)


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update(
        {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )
    return response


# Configure OAuth
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile", "state": lambda: os.urandom(16).hex()},
)


# Database Operations
def init_db():
    """Initialize SQLite database with required tables."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_name TEXT,
                credits INTEGER,
                login_streak INTEGER,
                last_login TEXT,
                last_guest_timestamp INTEGER,
                last_reward_date TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                answer TEXT,
                created_at TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS virtual_numbers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                phone_number TEXT,
                provider TEXT,
                purchase_date TEXT,
                status TEXT,
                price INTEGER,
                service_id TEXT,
                country TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS number_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                phone_number TEXT,
                provider TEXT,
                price INTEGER,
                purchase_date TEXT,
                status TEXT
            )"""
        )
        conn.commit()


init_db()


def check_credits(user_id: str, need: int = 1) -> bool:
    """Check if user has enough credits and deduct if available."""
    if not user_id or user_id in ADMIN_USERS:
        return True

    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        if not result or result[0] < need:
            return False
        c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
        conn.commit()
        return True


def get_credits(user_id: str) -> str:
    """Get user's credit balance."""
    if not user_id:
        return "0"
    if user_id in ADMIN_USERS:
        return "∞"
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        return str(result[0]) if result else "0"


def add_or_init_user(user_id: str, user_name: str = "User"):
    """Add or initialize user in database."""
    default_credits = 75 if "@" in user_id else 25
    if user_name == "Guest":
        default_credits = 25

    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            """INSERT OR IGNORE INTO users 
            (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                user_name,
                default_credits,
                0,
                datetime.now().strftime("%Y-%m-%d"),
                int(time.time()),
                "",
            ),
        )
        conn.commit()


def save_chat_history(user_id: str, question: str, answer: str):
    """Save chat history to database."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()


def get_chat_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Retrieve chat history for a user."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        return [
            {"question": row[0], "answer": row[1], "created_at": row[2]} for row in c.fetchall()
        ][::-1]


# VirtuSim Service
class VirtuSimService:
    def __init__(self):
        self.api_key = VIRTUSIM_API_KEY
        self.base_url = VIRTUSIM_API_URL

    async def get_available_services(self, country: str = "Indonesia", service: str = "Whatsapp") -> Dict[str, Any]:
        """Get real-time service availability and pricing."""
        async with httpx.AsyncClient() as client:
            try:
                params = {
                    "api_key": self.api_key,
                    "action": "services",
                    "country": country,
                    "service": service,
                }
                response = await client.get(f"{self.base_url}/json.php", params=params)
                response.raise_for_status()
                services_data = response.json()

                services = [
                    {
                        "service_id": service.get("service"),
                        "name": service.get("name", "Unknown Service"),
                        "description": f"Verifikasi {service.get('name', 'Service')} dengan nomor virtual",
                        "price": service.get("price", "Contact Admin"),
                        "price_formatted": f"Rp {float(service.get('price', 0)):,.2f}",
                        "available_numbers": service.get("available", 0),
                        "country": country,
                        "status": "Available" if service.get("available", 0) > 0 else "Out of Stock",
                        "duration": "48 jam",
                    }
                    for service in services_data.get("data", [])
                ]

                return {
                    "status": "success",
                    "data": services,
                    "contact": {"whatsapp": "wa.me/+628xxxxxxxx", "discord": "discord.gg/xxxxx"},
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            except Exception as e:
                logger.error(f"Error fetching VirtuSim services: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def get_supported_countries(self) -> List[str]:
        """Return list of supported countries."""
        return ["Indonesia", "Russia", "Vietnam", "Kazakhstan", "Ukraine"]


virtusim_service = VirtuSimService()


# Pydantic Models
class ChatRequest(BaseModel):
    user_email: str
    message: str
    model_select: str = "x-ai/grok-3-mini-beta"


class ImageRequest(BaseModel):
    user_email: str
    prompt: str


class VirtuSimPurchase(BaseModel):
    country: str
    service_id: str
    user_email: str


# API Endpoints
@app.get("/auth/google")
async def login_via_google(request: Request):
    """Initiate Google OAuth login."""
    redirect_uri = request.url_for("auth_google_callback")
    state = os.urandom(16).hex()
    request.session["oauth_state"] = state
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)


@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        expected_state = request.session.get("oauth_state")
        if not expected_state:
            logger.error("No state in session")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=no_state")

        token = await oauth.google.authorize_access_token(request)
        actual_state = request.query_params.get("state")
        if actual_state != expected_state:
            logger.error(f"Mismatching state: Expected {expected_state}, got {actual_state}")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=csrf_warning")

        del request.session["oauth_state"]

        user = token.get("userinfo") or await oauth.google.parse_id_token(request, token)
        if not user:
            resp = await oauth.google.get("userinfo", token=token)
            user = resp.json()

        email = user.get("email", "")
        if not email:
            logger.error("Google OAuth: Email not found in user profile")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_no_email")

        return RedirectResponse(url=f"{FRONTEND_URL}/menu?email={email}")
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_failed")


@app.post("/api/chat")
async def ai_chat(req: ChatRequest):
    """Handle AI chat requests."""
    if not req.user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")

    add_or_init_user(req.user_email, req.user_email)
    if not check_credits(req.user_email, 1):
        raise HTTPException(status_code=402, detail="NOT_ENOUGH_CREDITS")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kugy.ai",
        "X-Title": "KugyAI",
    }
    model_map = {
        "OpenRouter (Grok 3 Mini Beta)": "x-ai/grok-3-mini-beta",
        "OpenRouter (Gemini 2.0 Flash)": "google/gemini-flash-1.5",
    }
    payload = {
        "model": model_map.get(req.model_select, "x-ai/grok-3-mini-beta"),
        "messages": [
            {"role": "system", "content": "Act as an assistant."},
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        save_chat_history(req.user_email, req.message, reply)
        return {"reply": reply, "credits": get_credits(req.user_email)}
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        raise HTTPException(status_code=503, detail="AI Service Unavailable")


@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    """Generate image using Stability AI."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }

    if not req.user_email:
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401, headers=cors_headers)
    if not check_credits(req.user_email, 10):
        return JSONResponse(
            {"error": "NOT_ENOUGH_CREDITS", "message": "Need 10 credits"},
            status_code=402,
            headers=cors_headers,
        )
    if not STABILITY_API_KEY:
        return JSONResponse(
            {"error": "API_KEY_MISSING", "message": "Stability AI API key not set"},
            status_code=503,
            headers=cors_headers,
        )

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [{"text": req.prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }

    try:
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "artifacts" in data and data["artifacts"]:
            return JSONResponse(
                {
                    "image": data["artifacts"][0]["base64"],
                    "credits": get_credits(req.user_email),
                    "message": "Kugy.ai: Ini gambarnya buat ayang, cute banget kan? 🐱",
                },
                headers=cors_headers,
            )
        return JSONResponse(
            {"error": "Failed to get image from Stability AI"}, status_code=500, headers=cors_headers
        )
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return JSONResponse(
            {"error": f"Error generating image: {str(e)}"}, status_code=500, headers=cors_headers
        )


@app.get("/api/virtusim/services")
async def get_virtusim_services(country: str = Query("Indonesia"), service: str = Query("Whatsapp")):
    """Get VirtuSim services."""
    try:
        logger.info(f"Fetching VirtuSim services for country: {country}, service: {service}")
        result = await virtusim_service.get_available_services(country, service)
        return JSONResponse(
            {
                "status": "success",
                "data": result.get("data", []),
                "contact": result.get("contact"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": "lillysummer9794",
            }
        )
    except Exception as e:
        logger.error(f"Error in get_virtusim_services: {e}")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": "lillysummer9794",
            },
            status_code=500,
        )


@app.get("/api/virtusim/countries")
async def get_virtusim_countries():
    """Get supported VirtuSim countries."""
    try:
        countries = await virtusim_service.get_supported_countries()
        return JSONResponse(
            {
                "status": "success",
                "data": countries,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": "lillysummer9794",
            }
        )
    except Exception as e:
        logger.error(f"Error getting countries: {e}")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": "lillysummer9794",
            },
            status_code=500,
        )


@app.post("/api/virtusim/purchase")
async def purchase_virtusim_service(purchase: VirtuSimPurchase):
    """Handle VirtuSim service purchase."""
    try:
        service_info = await virtusim_service.get_available_services(purchase.country, purchase.service_id)
        if not service_info.get("data"):
            return JSONResponse(
                {
                    "status": "error",
                    "message": "Service not available for selected country",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": purchase.user_email,
                },
                status_code=400,
            )

        service_details = service_info["data"][0]
        return JSONResponse(
            {
                "status": "success",
                "message": "Silakan hubungi admin untuk melanjutkan pembelian",
                "service_details": {
                    "name": service_details["name"],
                    "price": service_details["price_formatted"],
                    "duration": service_details["duration"],
                    "country": purchase.country,
                    "available": service_details["available_numbers"],
                    "status": service_details["status"],
                },
                "contact": {"whatsapp": "wa.me/+628xxxxxxxx", "discord": "discord.gg/xxxxx"},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": purchase.user_email,
            }
        )
    except Exception as e:
        logger.error(f"Error in purchase request: {e}")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": purchase.user_email,
            },
            status_code=500,
        )


@app.get("/api/credits")
async def api_credits(user_email: str):
    """Get user credits."""
    return {"credits": get_credits(user_email)}


@app.get("/api/history")
async def api_history(user_email: str = Query(...), limit: int = Query(20, le=100)):
    """Get chat history."""
    if not user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    return {"history": get_chat_history(user_email, limit)}


@app.post("/api/guest-login")
async def guest_login(request: Request):
    """Handle guest login."""
    data = await request.json()
    user_email = data.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="Email wajib diisi")

    add_or_init_user(user_email, "Guest")
    dummy_token = f"guest-token-{user_email.split('@')[0]}"
    return JSONResponse(
        {
            "token": dummy_token,
            "credits": get_credits(user_email),
            "message": "Kugy.ai: Mode tamu aktif! 😺",
        }
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "message": "API is running",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": "lillysummer9794",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
