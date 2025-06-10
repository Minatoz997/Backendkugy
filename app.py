import base64
import logging
import os
import time
import uvicorn
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
from loguru import logger

# Configure logging with Loguru
logger.remove() # Remove default handler to avoid duplicate logs
logger.add(
    "file.log",
    rotation="10 MB",
    compression="zip",
    level="INFO",
    format="{time} {level} {message}",
    enqueue=True # Use multiprocessing-safe queue
)
logger.add(
    os.sys.stderr,
    level="INFO",
    format="{time} {level} {message}"
)

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
VIRTUSIM_API_URL = "https://api.1rentpro.online" # Pastikan ini URL yang benar

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
# ALLOWED_ORIGINS should be restricted in production
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    # Add other specific frontend origins here if necessary
]

# Initialize FastAPI
app = FastAPI()

# Configure middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: Change this to ALLOWED_ORIGINS in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=3600)


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
            logger.warning(f"User {user_id} does not have enough credits. Has {result[0] if result else 0}, needs {need}.")
            return False
        c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
        conn.commit()
        logger.info(f"User {user_id} credits reduced by {need}. Remaining: {result[0] - need}.")
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
    default_credits = 75 if "@" in user_id else 25 # Increased default for general users
    # For guest, credits are given on each "login" or as needed.
    # The guest login endpoint handles specific guest credit logic.

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_unix_timestamp = int(time.time())

    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        existing_user = c.fetchone()

        if existing_user:
            # Update last_login or last_guest_timestamp for existing users
            if user_name == "Guest":
                c.execute(
                    "UPDATE users SET last_guest_timestamp = ? WHERE user_id = ?",
                    (current_unix_timestamp, user_id)
                )
                logger.info(f"Guest user {user_id} last_guest_timestamp updated.")
            else:
                c.execute(
                    "UPDATE users SET last_login = ? WHERE user_id = ?, user_name = ?",
                    (current_time, user_id, user_name)
                )
                logger.info(f"Existing user {user_id} last_login updated.")
        else:
            # Insert new user
            c.execute(
                """INSERT INTO users
                (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    user_name,
                    default_credits,
                    0, # login_streak
                    current_time if user_name != "Guest" else "", # last_login for registered users
                    current_unix_timestamp if user_name == "Guest" else 0, # last_guest_timestamp for guests
                    "", # last_reward_date
                ),
            )
            logger.info(f"New user {user_id} ({user_name}) added with {default_credits} credits.")
        conn.commit()

def save_chat_history(user_id: str, question: str, answer: str):
    """Save chat history to database."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, current_time),
        )
        conn.commit()
        logger.info(f"Chat history saved for user {user_id}.")

def get_chat_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Retrieve chat history for a user."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        history = [
            {"question": row[0], "answer": row[1], "created_at": row[2]} for row in c.fetchall()
        ][::-1]
        logger.info(f"Retrieved {len(history)} chat entries for user {user_id}.")
        return history

# Configure OAuth
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile", "state": lambda: os.urandom(16).hex()},
)

# VirtuSim Service
class VirtuSimService:
    def __init__(self):
        self.api_key = VIRTUSIM_API_KEY
        self.base_url = VIRTUSIM_API_URL
        self.service_user = "KugyAI_Backend" # User identifier for VirtuSim service responses

    async def get_available_services(self, country: str = "indonesia", service: str = "wa") -> Dict[str, Any]:
        """Get real-time service availability and pricing."""
        current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if not self.api_key:
            logger.error("VIRTUSIM_API_KEY is not set.")
            return {
                "status": "error",
                "message": "VirtuSim API key is missing from environment variables.",
                "timestamp": current_datetime_str,
                "user": self.service_user
            }

        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}/json.php"
                params = {
                    "api_key": self.api_key,
                    "action": "services",
                    "country": country.lower() if country else "",
                    "service": service.lower() if service else ""
                }

                logger.info(f"VIRTUSIM API Request - URL: {url}, Params: {params}")
                
                response = await client.get(
                    url,
                    params=params,
                    timeout=30.0
                )
                
                logger.info(f"VIRTUSIM API Response - Status Code: {response.status_code}, Raw: {response.text[:500]}...") # Log partial raw response

                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.info(f"VIRTUSIM API Response - Parsed Data: {data}")
                        
                        services = []
                        raw_services_list = []

                        # Handle different possible API response structures
                        if isinstance(data, dict):
                            if "data" in data and isinstance(data["data"], list):
                                raw_services_list = data["data"]
                                logger.debug("VirtuSim response is dict with 'data' key.")
                            elif "error" in data:
                                logger.error(f"VirtuSim API returned an error: {data.get('error', 'Unknown error')}")
                                return {
                                    "status": "error",
                                    "message": f"VirtuSim API Error: {data.get('error', 'Unknown error')}",
                                    "timestamp": current_datetime_str,
                                    "user": self.service_user
                                }
                            else:
                                # Fallback if it's a dict but not in expected 'data' key
                                logger.warning(f"VirtuSim response is a dict but no 'data' key found, attempting to parse as list: {data}")
                                raw_services_list = [] # Explicitly set to empty if not in expected format
                        elif isinstance(data, list):
                            raw_services_list = data
                            logger.debug("VirtuSim response is a direct list.")
                        else:
                            logger.warning(f"VirtuSim response is neither a dict nor a list: {type(data)}")
                            raw_services_list = []

                        if not raw_services_list:
                            logger.info("No services found in VirtuSim response or response format was unexpected.")
                            
                        for service_item_from_api in raw_services_list:
                            try:
                                # Ensure 'stock' is an integer before checking
                                stock_value = int(service_item_from_api.get("stock", 0))

                                # Only include services with stock > 0 for display
                                if stock_value > 0:
                                    price_value = float(service_item_from_api.get("price", 0))
                                    service_item = {
                                        "service_id": str(service_item_from_api.get("id", "")),
                                        "name": service_item_from_api.get("name", "Unknown Service"),
                                        "description": f"Verifikasi {service_item_from_api.get('name', 'Unknown')} dengan nomor virtual",
                                        "price": price_value,
                                        "price_formatted": f"Rp {price_value:,.2f}",
                                        "available_numbers": stock_value,
                                        "country": service_item_from_api.get("country", country).capitalize(),
                                        "status": "available",
                                        "duration": service_item_from_api.get("validity", "48 jam"),
                                        "is_promo": bool(service_item_from_api.get("is_promo", False)),
                                        "category": service_item_from_api.get("category", "OTP")
                                    }
                                    services.append(service_item)
                                    logger.debug(f"Added service: {service_item['name']} (Stock: {stock_value})")
                                else:
                                    logger.debug(f"Skipping service with 0 stock: {service_item_from_api.get('name', 'Unknown Service')}")

                            except ValueError as ve:
                                logger.error(f"Error converting price/stock for service {service_item_from_api.get('name', 'Unknown')}: {ve}")
                                continue # Skip this malformed service

                        logger.info(f"Successfully processed {len(services)} available services.")
                        return {
                            "status": "success",
                            "data": services,
                            "contact": {
                                "whatsapp": "wa.me/+628xxxxxxxx", # TODO: Dynamize this
                                "discord": "discord.gg/xxxxx"      # TODO: Dynamize this
                            },
                            "timestamp": current_datetime_str,
                            "user": self.service_user
                        }
                    except Exception as e:
                        logger.error(f"Error parsing VirtuSim API response JSON: {e}", exc_info=True)
                        return {
                            "status": "error",
                            "message": f"Failed to parse VirtuSim API response: {str(e)}",
                            "timestamp": current_datetime_str,
                            "user": self.service_user
                        }

                else:
                    logger.error(f"VirtuSim API returned non-200 status code: {response.status_code} - {response.text}")
                    return {
                        "status": "error",
                        "message": f"VirtuSim API Error: {response.status_code} - {response.text}",
                        "timestamp": current_datetime_str,
                        "user": self.service_user
                    }

            except httpx.RequestError as e:
                logger.error(f"VirtuSim Request Error: {e}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"Network/Request Error connecting to VirtuSim API: {str(e)}",
                    "timestamp": current_datetime_str,
                    "user": self.service_user
                }
            except Exception as e:
                logger.error(f"An unexpected error occurred during VirtuSim API call: {e}", exc_info=True)
                return {
                    "status": "error",
                    "message": f"An unexpected error occurred: {str(e)}",
                    "timestamp": current_datetime_str,
                    "user": self.service_user
                }

    async def get_supported_countries(self) -> List[str]:
        """Return list of supported countries."""
        # TODO: If VirtuSim API has an endpoint for countries, fetch dynamically.
        # Otherwise, this hardcoded list is fine.
        current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Providing hardcoded list of supported countries for VirtuSim.")
        return [
            "Indonesia", "Russia", "Vietnam", "Kazakhstan", "Ukraine",
            "Philippines", "Thailand", "Malaysia", "India", "China", "United States"
        ] # Added more common countries for demo

# Initialize VirtuSim service
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
    logger.info(f"Initiating Google OAuth login. Redirect URI: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """Handle Google OAuth callback."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        expected_state = request.session.get("oauth_state")
        if not expected_state:
            logger.error("Google OAuth callback: No state found in session.")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=no_oauth_state")

        token = await oauth.google.authorize_access_token(request)
        actual_state = request.query_params.get("state")
        if actual_state != expected_state:
            logger.error(f"Google OAuth callback: Mismatching state. Expected: {expected_state}, Got: {actual_state}")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=csrf_warning")

        del request.session["oauth_state"]

        user = token.get("userinfo") or await oauth.google.parse_id_token(request, token)
        if not user:
            resp = await oauth.google.get("userinfo", token=token)
            user = resp.json()

        email = user.get("email", "")
        if not email:
            logger.error("Google OAuth: Email not found in user profile after successful authorization.")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_no_email")

        add_or_init_user(email, user.get("name", "User"))
        logger.info(f"Google OAuth successful for user: {email}")
        return RedirectResponse(url=f"{FRONTEND_URL}/menu?email={email}")
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}", exc_info=True)
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_failed")

@app.post("/api/chat")
async def ai_chat(req: ChatRequest):
    """Handle AI chat requests."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not req.user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED: User email is required.")

    add_or_init_user(req.user_email, req.user_email)
    if not check_credits(req.user_email, 1):
        logger.warning(f"AI Chat: User {req.user_email} not enough credits for chat.")
        raise HTTPException(status_code=402, detail="NOT_ENOUGH_CREDITS: You do not have enough credits.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kugy.ai", # Replace with your actual domain
        "X-Title": "KugyAI",
    }
    model_map = {
        "OpenRouter (Grok 3 Mini Beta)": "x-ai/grok-3-mini-beta",
        "OpenRouter (Gemini 2.0 Flash)": "google/gemini-flash-1.5",
        "OpenRouter (Meta Llama 3 8B)": "meta-llama/llama-3-8b-instruct",
        "OpenRouter (Google Gemini 1.5 Pro)": "google/gemini-1.5-pro-flash",
    }
    selected_model = model_map.get(req.model_select, "x-ai/grok-3-mini-beta")
    
    payload = {
        "model": selected_model,
        "messages": [
            {"role": "system", "content": "You are a helpful and friendly AI assistant named Kugy.ai. Answer truthfully and concisely."},
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.7,
    }

    try:
        logger.info(f"Sending chat request to OpenRouter for user {req.user_email} with model {selected_model}.")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        reply = response.json()["choices"][0]["message"]["content"]
        save_chat_history(req.user_email, req.message, reply)
        logger.info(f"AI Chat successful for user {req.user_email}. Reply length: {len(reply)}.")
        return {"reply": reply, "credits": get_credits(req.user_email)}
    except requests.exceptions.Timeout:
        logger.error(f"OpenRouter request timed out for user {req.user_email}.")
        raise HTTPException(status_code=504, detail="AI Service Timeout: The AI service took too long to respond.")
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API error for user {req.user_email}: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"AI Service Unavailable: {e}")
    except KeyError:
        logger.error(f"OpenRouter API response malformed for user {req.user_email}: {response.json()}")
        raise HTTPException(status_code=500, detail="AI Service Error: Malformed response from AI provider.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in AI chat for user {req.user_email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    """Generate image using Stability AI."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cors_headers = {
        "Access-Control-Allow-Origin": "*", # WARN: Change to specific origins in prod
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }

    if not req.user_email:
        logger.warning("Image generation request: Unauthorized (no user email).")
        return JSONResponse({"error": "UNAUTHORIZED", "message": "User email is required."}, status_code=401, headers=cors_headers)
    if not check_credits(req.user_email, 10):
        logger.warning(f"Image generation: User {req.user_email} not enough credits (need 10).")
        return JSONResponse(
            {"error": "NOT_ENOUGH_CREDITS", "message": "Anda membutuhkan 10 kredit untuk menghasilkan gambar."},
            status_code=402,
            headers=cors_headers,
        )
    if not STABILITY_API_KEY:
        logger.error("Stability AI API key is not set.")
        return JSONResponse(
            {"error": "API_KEY_MISSING", "message": "Kunci API Stability AI belum diatur di server."},
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
        logger.info(f"Sending image generation request to Stability AI for user {req.user_email}. Prompt: '{req.prompt[:50]}...'")
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "artifacts" in data and data["artifacts"]:
            logger.info(f"Image generated successfully for user {req.user_email}.")
            return JSONResponse(
                {
                    "image": data["artifacts"][0]["base64"],
                    "credits": get_credits(req.user_email),
                    "message": "Kugy.ai: Ini gambarnya buat ayang, cute banget kan? 🐱",
                },
                headers=cors_headers,
            )
        logger.warning(f"Stability AI response did not contain artifacts for user {req.user_email}. Response: {data}")
        return JSONResponse(
            {"error": "FAILED_TO_GENERATE_IMAGE", "message": "Gagal mendapatkan gambar dari Stability AI. Respon tidak valid."}, status_code=500, headers=cors_headers
        )
    except requests.exceptions.Timeout:
        logger.error(f"Stability AI request timed out for user {req.user_email}.")
        return JSONResponse({"error": "IMAGE_GEN_TIMEOUT", "message": "Layanan pembuatan gambar terlalu lama merespons."}, status_code=504, headers=cors_headers)
    except requests.exceptions.RequestException as e:
        logger.error(f"Stability AI API error for user {req.user_email}: {e}", exc_info=True)
        return JSONResponse(
            {"error": "IMAGE_GEN_ERROR", "message": f"Terjadi kesalahan saat menghasilkan gambar: {str(e)}"}, status_code=500, headers=cors_headers
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred in image generation for user {req.user_email}: {e}", exc_info=True)
        return JSONResponse(
            {"error": "UNEXPECTED_ERROR", "message": f"Terjadi kesalahan tak terduga: {str(e)}"}, status_code=500, headers=cors_headers
        )

@app.get("/api/virtusim/services")
async def get_virtusim_services(country: str = Query("Indonesia"), service: str = Query("Whatsapp")):
    """Get VirtuSim services."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        logger.info(f"API Call: /api/virtusim/services (Country: {country}, Service: {service})")
        
        result = await virtusim_service.get_available_services(country, service)
        
        # Ensure status and message are properly propagated from service
        if result.get("status") == "error":
            logger.error(f"Error fetching VirtuSim services: {result.get('message', 'Unknown error')}")
            return JSONResponse(
                {
                    "status": "error",
                    "message": result.get("message", "Failed to retrieve services."),
                    "timestamp": current_datetime_str,
                    "user": "API_Caller" # Or actual user_email if available
                },
                status_code=500,
            )
        
        logger.info(f"Successfully retrieved {len(result.get('data', []))} VirtuSim services.")
        return JSONResponse(result) # Return the full result including status, data, contact, timestamp, user
    except Exception as e:
        logger.error(f"Exception in /api/virtusim/services endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            {
                "status": "error",
                "message": f"Terjadi kesalahan saat mengambil layanan VirtuSim: {str(e)}",
                "timestamp": current_datetime_str,
                "user": "API_Caller" # Or actual user_email if available
            },
            status_code=500,
        )

@app.get("/api/virtusim/countries")
async def get_virtusim_countries():
    """Get supported VirtuSim countries."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        countries = await virtusim_service.get_supported_countries()
        logger.info(f"Successfully retrieved {len(countries)} supported VirtuSim countries.")
        return JSONResponse(
            {
                "status": "success",
                "data": countries,
                "timestamp": current_datetime_str,
                "user": virtusim_service.service_user # Use service user defined in class
            }
        )
    except Exception as e:
        logger.error(f"Error getting VirtuSim countries: {e}", exc_info=True)
        return JSONResponse(
            {
                "status": "error",
                "message": f"Terjadi kesalahan saat mengambil daftar negara: {str(e)}",
                "timestamp": current_datetime_str,
                "user": virtusim_service.service_user
            },
            status_code=500,
        )

@app.post("/api/virtusim/purchase")
async def purchase_virtusim_service(purchase: VirtuSimPurchase):
    """Handle VirtuSim service purchase."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Purchase request received from {purchase.user_email} for service_id {purchase.service_id} in {purchase.country}")

    try:
        # Step 1: Get service details to determine price and availability
        service_info_result = await virtusim_service.get_available_services(purchase.country, purchase.service_id)
        
        if service_info_result.get("status") == "error" or not service_info_result.get("data"):
            logger.warning(f"Purchase failed for {purchase.user_email}: Service '{purchase.service_id}' in '{purchase.country}' not found or API error.")
            return JSONResponse(
                {
                    "status": "error",
                    "message": "Layanan yang dipilih tidak tersedia atau terjadi kesalahan saat mengambil data layanan.",
                    "timestamp": current_datetime_str,
                    "user": purchase.user_email,
                },
                status_code=400,
            )

        # Find the specific service_id from the list
        service_details = None
        for s in service_info_result["data"]:
            if s["service_id"] == purchase.service_id:
                service_details = s
                break
        
        if not service_details:
            logger.warning(f"Purchase failed for {purchase.user_email}: Specific service_id {purchase.service_id} not found in available services list.")
            return JSONResponse(
                {
                    "status": "error",
                    "message": "Service ID tidak ditemukan atau tidak tersedia untuk pembelian.",
                    "timestamp": current_datetime_str,
                    "user": purchase.user_email,
                },
                status_code=400,
            )
        
        service_price = int(service_details["price"]) # Assuming price is in integer (Rupiah without decimals for credits)
        
        # Step 2: Check and deduct user credits
        if not check_credits(purchase.user_email, service_price):
            logger.warning(f"Purchase failed for {purchase.user_email}: Not enough credits to purchase {service_details['name']} (needed {service_price}).")
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"Kredit Anda tidak cukup untuk pembelian ini. Anda membutuhkan {service_price} kredit.",
                    "timestamp": current_datetime_str,
                    "user": purchase.user_email,
                },
                status_code=402,
            )

        # --- IMPORTANT: Placeholder for actual VirtuSim number purchase ---
        # At this point, credits are deducted. Now you would call VirtuSim API
        # to actually acquire the number. This part is NOT implemented here
        # as it depends on the exact "buy number" endpoint of VirtuSim API.
        # Example:
        # virtu_number_response = await client.get(f"{self.base_url}/json.php", params={"action": "get_number", "service_id": service_details["service_id"], ...})
        # If successful, extract phone_number and other details.
        # --- END Placeholder ---

        # For now, we'll simulate a successful "purchase" message.
        # In a real scenario, you'd get the actual number and status from VirtuSim.
        
        # Step 3: Save purchase details to database (even if just for "pending contact")
        with sqlite3.connect("credits.db") as conn:
            c = conn.cursor()
            c.execute(
                """INSERT INTO number_purchases
                (user_id, phone_number, provider, price, purchase_date, status)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    purchase.user_email,
                    "SIMULATED_NUMBER", # Replace with actual number from VirtuSim API
                    service_details["name"],
                    service_price,
                    current_datetime_str,
                    "PENDING_ADMIN_CONTACT" # Change to "ACTIVE" or "WAITING_SMS" after real purchase
                ),
            )
            # Also save to virtual_numbers if this is a long-term number
            c.execute(
                """INSERT INTO virtual_numbers
                (user_id, phone_number, provider, purchase_date, status, price, service_id, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    purchase.user_email,
                    "SIMULATED_NUMBER", # Replace with actual number from VirtuSim API
                    service_details["name"],
                    current_datetime_str,
                    "PENDING",
                    service_price,
                    service_details["service_id"],
                    service_details["country"]
                )
            )
            conn.commit()
        logger.info(f"Purchase simulated/logged for {purchase.user_email}. Credits deducted: {service_price}.")

        return JSONResponse(
            {
                "status": "success",
                "message": "Pembelian berhasil! Silakan hubungi admin untuk melanjutkan dan mendapatkan nomor Anda.", # Updated message
                "service_details": {
                    "name": service_details["name"],
                    "price": service_details["price_formatted"],
                    "duration": service_details["duration"],
                    "country": purchase.country,
                    "available": service_details["available_numbers"],
                    "status": service_details["status"],
                },
                "contact": {"whatsapp": "wa.me/+628xxxxxxxx", "discord": "discord.gg/xxxxx"}, # TODO: Dynamize this
                "current_credits": get_credits(purchase.user_email),
                "timestamp": current_datetime_str,
                "user": purchase.user_email,
            }
        )
    except Exception as e:
        logger.error(f"Error in VirtuSim purchase request for {purchase.user_email}: {e}", exc_info=True)
        return JSONResponse(
            {
                "status": "error",
                "message": f"Terjadi kesalahan tak terduga saat memproses pembelian: {str(e)}",
                "timestamp": current_datetime_str,
                "user": purchase.user_email,
            },
            status_code=500,
        )

@app.get("/api/credits")
async def api_credits(user_email: str = Query(...)):
    """Get user credits."""
    if not user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED: User email is required.")
    
    add_or_init_user(user_email, user_email) # Ensure user exists for credit lookup
    credits = get_credits(user_email)
    logger.info(f"Credits requested for {user_email}: {credits}.")
    return {"credits": credits}

@app.get("/api/history")
async def api_history(user_email: str = Query(...), limit: int = Query(20, le=100)):
    """Get chat history."""
    if not user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED: User email is required.")
    
    history = get_chat_history(user_email, limit)
    logger.info(f"History requested for {user_email}. Returning {len(history)} entries.")
    return {"history": history}

@app.post("/api/guest-login")
async def guest_login(request: Request):
    """Handle guest login."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = await request.json()
    user_email_prefix = data.get("email_prefix") # Expecting "guest_user_123" instead of full email
    
    if not user_email_prefix:
        logger.warning("Guest login attempt: Missing email prefix.")
        raise HTTPException(status_code=400, detail="Email prefix wajib diisi untuk login tamu.")

    guest_user_id = f"guest_{user_email_prefix}" # Standardize guest ID format

    # Add/init user, update last_guest_timestamp
    add_or_init_user(guest_user_id, "Guest")
    
    # Give guest users some default credits (e.g., 25) if they don't have enough
    # This ensures new guests get credits and existing ones get refreshed or simply checked
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (guest_user_id,))
        current_credits_result = c.fetchone()
        current_credits = current_credits_result[0] if current_credits_result else 0
        
        # Decide when to give more credits. Example: If below 10, add up to 25.
        # Or, if it's a new guest session after some time, give full 25.
        # For simplicity, let's just ensure they have a base amount.
        if current_credits < 25:
             # Just add 25 for guests if they are low. Could be more complex logic for daily rewards etc.
            c.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (25 - current_credits, guest_user_id))
            conn.commit()
            logger.info(f"Guest user {guest_user_id} credits topped up to 25.")
            
    dummy_token = f"guest-token-{guest_user_id}"
    credits = get_credits(guest_user_id) # Get updated credits
    logger.info(f"Guest login successful for {guest_user_id}. Credits: {credits}.")

    return JSONResponse(
        {
            "token": dummy_token,
            "user_id": guest_user_id, # Return the generated guest user ID
            "credits": credits,
            "message": "Kugy.ai: Mode tamu aktif! Kredit awal 25 telah ditambahkan. 😺",
            "timestamp": current_datetime_str,
        }
    )

@app.get("/")
async def root():
    """Root endpoint."""
    current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "status": "ok",
        "message": "KugyAI API is running smoothly!",
        "timestamp": current_datetime_str,
        "location": "Purwokerto, Central Java, Indonesia",
        "service_health": {
            "database": "connected",
            "openrouter_api": "status_ok", # Placeholder, would need actual check
            "stability_ai_api": "status_ok", # Placeholder
            "virtusim_api": "status_ok" # Placeholder
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Remove reload=True in production for performance
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True, log_level="info")

