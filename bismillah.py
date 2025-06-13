
import base64
import logging
import os
import sys
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

# Environment Variables with better defaults
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://front-end-bpup.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "changeme_secret_key_123456")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
VIRTUSIM_API_KEY = os.getenv("VIRTUSIM_API_KEY")

# Log environment status
logger.info(f"Using SESSION_SECRET_KEY: {SESSION_SECRET_KEY[:5]}..." if SESSION_SECRET_KEY else "SESSION_SECRET_KEY not set")
logger.info(f"VIRTUSIM_API_KEY: {VIRTUSIM_API_KEY[:5]}..." if VIRTUSIM_API_KEY else "VIRTUSIM_API_KEY is not set")
logger.info(f"FRONTEND_URL: {FRONTEND_URL}")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app",
    "https://front-end-beta-liard.vercel.app",
    "*"  # Allow all origins for deployment
]

# Dynamic callback URL based on environment
BACKEND_URL = os.getenv("BACKEND_URL", "https://backend-cb98.onrender.com")
CALLBACK_URL = f"{BACKEND_URL}/auth/google/callback"

DB_PATH = os.getenv("DB_PATH", "credits.db")
GUEST_INITIAL_CREDITS = 25  # Fixed guest credits to 25

def ensure_db_and_log():
    """Ensure database and log file exist with proper initialization"""
    try:
        # Configure logging
        logger.remove()
        logger.add(
            "app.log",
            rotation="10 MB",
            compression="zip",
            level="INFO",
            format="{time} {level} {message}",
            enqueue=True
        )
        logger.add(
            sys.stderr,
            level="INFO",
            format="{time} {level} {message}"
        )
        
        # Initialize database with admin users
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Create users table
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    credits INTEGER,
                    login_streak INTEGER,
                    last_login TEXT,
                    last_guest_timestamp INTEGER,
                    last_reward_date TEXT
                )
            ''')
            
            # Create chat_history table
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TEXT
                )
            ''')
            
            # Insert admin users if they don't exist
            admin_users = [
                ("admin@kugy.ai", "Admin", 999999, 0, "2025-06-11 18:12:33", 0, ""),
                ("testadmin", "Test Admin", 999999, 0, "2025-06-11 18:12:33", 0, "")
            ]
            
            c.executemany('''
                INSERT OR IGNORE INTO users 
                (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', admin_users)
            
            conn.commit()
            
        logger.info("Database and log file initialized successfully")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

# Initialize database and logging
ensure_db_and_log()

# Initialize FastAPI with Kugy AI branding
app = FastAPI(title="Kugy AI API Backend", version="1.0.0")

# Configure middleware - SESSION MIDDLEWARE HARUS DITAMBAHKAN DULU SEBELUM CORS
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=86400,  # 24 hours
    same_site="none",  # Changed to "none" for cross-origin requests
    https_only=True,  # Set to True for production HTTPS
)

# Configure CORS middleware setelah SessionMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def check_credits(user_id: str, need: int = 1) -> bool:
    """Check if user has enough credits and deduct if available."""
    if not user_id or user_id in ADMIN_USERS:
        return True

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
            result = c.fetchone()
            if not result or result[0] < need:
                return False
            c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error checking credits: {e}")
        return False

def get_credits(user_id: str) -> str:
    """Get user's credit balance."""
    if not user_id:
        return "0"
    if user_id in ADMIN_USERS:
        return "∞"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
            result = c.fetchone()
            return str(result[0]) if result else "0"
    except Exception as e:
        logger.error(f"Error getting credits: {e}")
        return "0"

def add_or_init_user(user_id: str, user_name: str = "User"):
    """Add or initialize user in database."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if "@" in user_id:
        default_credits = 75  # Email users get 75 credits
    else:
        default_credits = GUEST_INITIAL_CREDITS  # Guest users get 25 credits

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            exists = c.fetchone()
            
            if not exists:
                c.execute(
                    """INSERT INTO users 
                    (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        user_name,
                        default_credits,
                        0,
                        current_time,
                        int(time.time()),
                        "",
                    ),
                )
                conn.commit()
                logger.info(f"New user initialized: {user_id} with {default_credits} credits")
            else:
                logger.info(f"User already exists: {user_id}")
                
    except Exception as e:
        logger.error(f"Error initializing user: {e}")
        raise

def save_chat_history(user_id: str, question: str, answer: str):
    """Save chat history to database."""
    try:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, current_time),
            )
            conn.commit()
            logger.info(f"Chat history saved for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        raise

def get_chat_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Retrieve chat history for a user."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            )
            history = [
                {"question": row[0], "answer": row[1], "created_at": row[2]} for row in c.fetchall()
            ][::-1]
            logger.info(f"Retrieved {len(history)} chat entries for user {user_id}")
            return history
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

# Configure OAuth only if credentials are available
oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        authorize_url="https://accounts.google.com/o/oauth2/auth",
        authorize_params=None,
        token_url="https://accounts.google.com/o/oauth2/token",
        token_params=None,
        api_base_url="https://www.googleapis.com/oauth2/v1/",
        client_kwargs={"scope": "openid email profile"}
    )
    logger.info("Google OAuth configured successfully")
else:
    logger.warning("Google OAuth not configured - missing client credentials")

# VirtuSim Service (kept for compatibility)
class VirtuSimService:
    def __init__(self):
        self.api_key = VIRTUSIM_API_KEY
        self.base_url = "https://virtusim.com/api/v2/json.php"

    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to VirtuSim API with error handling."""
        try:
            if not self.api_key:
                logger.error("VIRTUSIM_API_KEY is not set")
                return {"status": False, "data": {"msg": "API key missing"}}

            params["api_key"] = self.api_key
            async with httpx.AsyncClient() as client:
                logger.info(f"VirtuSim Request - Params: {params}")
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                logger.info(f"VirtuSim Response - Status: {response.status_code}")
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"VirtuSim HTTP Error: {e}")
            return {"status": False, "data": {"msg": str(e)}}
        except Exception as e:
            logger.error(f"VirtuSim Error: {e}")
            return {"status": False, "data": {"msg": str(e)}}

    # Account Endpoints
    async def check_balance(self) -> Dict[str, Any]:
        """Check account balance."""
        return await self._make_request({"action": "balance"})

    async def get_balance_logs(self) -> Dict[str, Any]:
        """Get balance mutation history."""
        return await self._make_request({"action": "balance_logs"})

    async def get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity."""
        return await self._make_request({"action": "recent_activity"})

    # Service Endpoints
    async def get_available_services(self, country: str = "indonesia") -> Dict[str, Any]:
        """Get available services."""
        return await self._make_request({
            "action": "services",
            "country": country,
        })

    async def get_countries(self) -> Dict[str, Any]:
        """Get list of available countries."""
        return await self._make_request({"action": "list_country"})

    async def get_operators(self, country: str) -> Dict[str, Any]:
        """Get list of operators for a country."""
        return await self._make_request({
            "action": "list_operator",
            "country": country
        })

    # Transaction Endpoints
    async def get_active_orders(self) -> Dict[str, Any]:
        """Get active transactions."""
        return await self._make_request({"action": "active_order"})

    async def create_order(self, service: str, operator: str = "any") -> Dict[str, Any]:
        """Create new order."""
        return await self._make_request({
            "action": "order",
            "service": service,
            "operator": operator
        })

    async def reactive_order(self, order_id: str) -> Dict[str, Any]:
        """Reactivate existing order."""
        return await self._make_request({
            "action": "reactive_order",
            "id": order_id
        })

    async def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """Check order status."""
        return await self._make_request({
            "action": "check_order",
            "id": order_id
        })

# Multi-Agent System
class MultiAgentSystem:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = OPENROUTER_API_URL
        
        # Definisi 3 agent dengan peran berbeda
        self.agents = {
            "analyzer": {
                "model": "anthropic/claude-3.5-sonnet",
                "role": "Analisis dan Pemahaman",
                "system_prompt": """Anda adalah Agent Analyzer yang bertugas menganalisis task dan memecahnya menjadi komponen-komponen detail. 
                Tugas Anda:
                1. Menganalisis pertanyaan/task yang diberikan
                2. Mengidentifikasi informasi kunci yang dibutuhkan
                3. Memecah task menjadi sub-task yang lebih kecil
                4. Memberikan insight dan perspektif analitis
                Jawab dengan format terstruktur dan detail."""
            },
            "researcher": {
                "model": "x-ai/grok-2-mini",
                "role": "Riset dan Informasi",
                "system_prompt": """Anda adalah Agent Researcher yang bertugas mencari informasi mendalam dan memberikan data pendukung.
                Tugas Anda:
                1. Menggali informasi detail tentang topik yang dianalisis
                2. Memberikan contoh, referensi, dan data pendukung
                3. Mencari solusi alternatif dan best practices
                4. Menyediakan context dan background information
                Fokus pada akurasi dan kelengkapan informasi."""
            },
            "synthesizer": {
                "model": "meta-llama/llama-3.1-405b-instruct",
                "role": "Sintesis dan Solusi",
                "system_prompt": """Anda adalah Agent Synthesizer yang bertugas menggabungkan hasil analisis dan riset menjadi solusi final.
                Tugas Anda:
                1. Menggabungkan hasil dari Agent Analyzer dan Researcher
                2. Membuat solusi komprehensif dan actionable
                3. Memberikan rekomendasi praktis dan implementable
                4. Menyusun jawaban final yang jelas dan mudah dipahami
                Berikan solusi yang konkret dan dapat diterapkan."""
            }
        }

    async def _call_agent(self, agent_name: str, messages: List[Dict[str, str]]) -> str:
        """Memanggil individual agent dengan model yang ditentukan."""
        try:
            if not self.api_key:
                return f"Error: OpenRouter API key tidak tersedia untuk {agent_name}"

            agent_config = self.agents[agent_name]
            
            # Tambahkan system prompt ke messages
            full_messages = [
                {"role": "system", "content": agent_config["system_prompt"]}
            ] + messages

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": BACKEND_URL,
                "X-Title": "Kugy AI Multi-Agent System"  # Changed to Kugy AI
            }

            payload = {
                "model": agent_config["model"],
                "messages": full_messages,
                "max_tokens": 2000,
                "temperature": 0.7
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return f"Error: Tidak ada response dari {agent_name}"

        except Exception as e:
            logger.error(f"Error calling {agent_name}: {e}")
            return f"Error {agent_name}: {str(e)}"

    async def process_multi_agent_task(self, task: str) -> Dict[str, Any]:
        """Memproses task menggunakan 3 agent secara berurutan."""
        try:
            results = {}
            
            # Step 1: Agent Analyzer menganalisis task
            logger.info("Memulai analisis dengan Agent Analyzer...")
            analyzer_messages = [
                {"role": "user", "content": f"Analisis task berikut: {task}"}
            ]
            results["analyzer"] = await self._call_agent("analyzer", analyzer_messages)
            
            # Step 2: Agent Researcher melakukan riset berdasarkan analisis
            logger.info("Memulai riset dengan Agent Researcher...")
            researcher_messages = [
                {"role": "user", "content": f"Berdasarkan analisis berikut, lakukan riset mendalam:\n\nAnalisis: {results['analyzer']}\n\nTask asli: {task}"}
            ]
            results["researcher"] = await self._call_agent("researcher", researcher_messages)
            
            # Step 3: Agent Synthesizer menggabungkan hasil menjadi solusi final
            logger.info("Mensintesis hasil dengan Agent Synthesizer...")
            synthesizer_messages = [
                {"role": "user", "content": f"Gabungkan hasil analisis dan riset berikut menjadi solusi komprehensif:\n\nTask: {task}\n\nAnalisis: {results['analyzer']}\n\nRiset: {results['researcher']}\n\nBerikan solusi final yang praktis dan actionable."}
            ]
            results["synthesizer"] = await self._call_agent("synthesizer", synthesizer_messages)
            
            # Kompilasi hasil final
            final_result = {
                "task": task,
                "multi_agent_results": {
                    "analysis": {
                        "agent": "Analyzer (Claude-3.5-Sonnet)",
                        "role": "Analisis dan Pemahaman",
                        "result": results["analyzer"]
                    },
                    "research": {
                        "agent": "Researcher (Grok-2-Mini)",
                        "role": "Riset dan Informasi", 
                        "result": results["researcher"]
                    },
                    "synthesis": {
                        "agent": "Synthesizer (Llama-3.1-405B)",
                        "role": "Sintesis dan Solusi",
                        "result": results["synthesizer"]
                    }
                },
                "final_answer": results["synthesizer"],
                "processing_time": "Diproses oleh 3 AI agent secara berurutan",
                "models_used": ["Claude-3.5-Sonnet", "Grok-2-Mini", "Llama-3.1-405B"]
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error dalam multi-agent processing: {e}")
            return {
                "error": True,
                "message": f"Error dalam multi-agent system: {str(e)}",
                "task": task
            }

# Initialize services
virtusim_service = VirtuSimService()
multi_agent = MultiAgentSystem()

# Pydantic Models
class ChatRequest(BaseModel):
    query: str
    user_id: str

class ImageRequest(BaseModel):
    prompt: str
    user_id: str

class VirtuSimOrderRequest(BaseModel):
    service: str
    operator: str = "any"

class VirtuSimCheckRequest(BaseModel):
    order_id: str

class VirtuSimReactiveRequest(BaseModel):
    order_id: str

class MultiAgentRequest(BaseModel):
    task: str
    user_id: str = "guest"
    use_multi_agent: bool = True

# Helper Functions
def resize_image(image_data: bytes, max_size: int = 1024) -> bytes:
    """Resize image to max_size while maintaining aspect ratio."""
    try:
        image = Image.open(BytesIO(image_data))
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image_data

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint dengan informasi API."""
    return {
        "message": "Kugy AI API Backend",  # Changed to Kugy AI
        "version": "1.0.0",
        "status": "active",
        "features": ["Google OAuth", "Single AI Chat", "Multi-Agent System", "Image Generation", "VirtuSim Integration"],
        "multi_agent_models": ["Claude-3.5-Sonnet", "Grok-2-Mini", "Llama-3.1-405B"],
        "timestamp": datetime.now().isoformat(),
        "backend_url": BACKEND_URL,
        "frontend_url": FRONTEND_URL
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "virtusim_api_configured": bool(VIRTUSIM_API_KEY),
            "openrouter_api_configured": bool(OPENROUTER_API_KEY),
            "google_oauth_configured": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
            "stability_api_configured": bool(STABILITY_API_KEY)
        }
    }

# Auth Endpoints
@app.get("/auth/google")
async def google_auth(request: Request):
    """Initiate Google OAuth."""
    try:
        if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
            
        redirect_uri = CALLBACK_URL
        logger.info(f"Initiating Google OAuth with redirect_uri: {redirect_uri}")
        return await oauth.google.authorize_redirect(request, redirect_uri)
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
            return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=oauth_not_configured")
            
        # Get token from Google
        token = await oauth.google.authorize_access_token(request)
        logger.info("Successfully received Google token")
        
        # Get user info
        user_info = token.get("userinfo")
        if not user_info:
            logger.error("No user info in token")
            return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=no_user_info")
        
        user_email = user_info.get("email")
        user_name = user_info.get("name", "User")
        
        logger.info(f"User authenticated: {user_email}")
        
        # Store user session
        request.session["user"] = {
            "email": user_email,
            "name": user_name,
            "authenticated": True,
        }
        
        logger.info(f"User session stored: {user_email}")
        
        # Initialize user in database
        add_or_init_user(user_email, user_name)
        
        # Redirect to frontend with proper success parameter
        return RedirectResponse(url=f"{FRONTEND_URL}/?auth=success&user={user_email}")
        
    except Exception as e:
        logger.error(f"Google callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=callback_failed")

@app.post("/auth/logout")
async def logout(request: Request):
    """Logout user."""
    request.session.clear()
    return {"message": "Logged out successfully"}

@app.get("/auth/user")
async def get_user(request: Request):
    """Get current user information."""
    user = request.session.get("user")
    if not user:
        return {"user": None, "authenticated": False}
    
    user_id = user.get("email", "guest")
    credits = get_credits(user_id)
    
    return {
        "user": user,
        "authenticated": True,
        "credits": credits
    }

@app.post("/auth/guest")
async def guest_login(request: Request):
    """Create guest session."""
    guest_id = f"guest_{int(time.time())}"
    request.session["user"] = {
        "email": guest_id,
        "name": "Guest User",
        "authenticated": False,
    }
    
    # Initialize guest user
    add_or_init_user(guest_id, "Guest User")
    
    return {
        "user": request.session["user"],
        "authenticated": False,
        "credits": get_credits(guest_id)
    }

# API endpoints dengan prefix /api untuk kompatibilitas frontend
@app.post("/api/guest-login")
async def api_guest_login(request: Request):
    """Create guest session (API endpoint)."""
    try:
        guest_id = f"guest_{int(time.time())}"
        guest_name = "Guest User"
        
        # Initialize guest user in database
        add_or_init_user(guest_id, guest_name)
        
        # Store in session
        request.session["user"] = {
            "email": guest_id,
            "name": guest_name,
            "authenticated": False,
        }
        
        logger.info(f"Guest user created: {guest_id}")
        
        return {
            "success": True,
            "user": {
                "user_id": guest_id,
                "user_name": guest_name,
                "email": guest_id,
                "authenticated": False
            },
            "credits": get_credits(guest_id),
            "message": "Guest login successful"
        }
    except Exception as e:
        logger.error(f"Guest login error: {e}")
        raise HTTPException(status_code=500, detail=f"Guest login failed: {str(e)}")

@app.get("/api/auth/user")
async def api_get_user(request: Request):
    """Get current user info (API endpoint)."""
    try:
        user = request.session.get("user")
        if not user:
            return {
                "success": False,
                "user": None,
                "authenticated": False,
                "credits": "0"
            }
        
        user_id = user.get("email", "guest")
        credits = get_credits(user_id)
        
        return {
            "success": True,
            "user": {
                "user_id": user_id,
                "user_name": user.get("name", "User"),
                "email": user_id,
                "authenticated": user.get("authenticated", False)
            },
            "credits": credits
        }
    except Exception as e:
        logger.error(f"Get user error: {e}")
        return {
            "success": False,
            "user": None,
            "authenticated": False,
            "credits": "0"
        }

@app.post("/api/chat")
async def api_chat_completion(chat_request: ChatRequest, request: Request):
    """Generate chat completion using OpenRouter API (API endpoint)."""
    try:
        # Get user from session or use provided user_id
        session_user = request.session.get("user")
        user_id = session_user.get("email") if session_user else chat_request.user_id
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Check credits
        if not check_credits(user_id, 1):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OpenRouter API not configured")
        
        # Make request to OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": chat_request.query}],
            "max_tokens": 1000,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
        
        answer = result["choices"][0]["message"]["content"]
        
        # Save chat history
        save_chat_history(user_id, chat_request.query, answer)
        
        return {
            "success": True,
            "response": answer,
            "credits_remaining": get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter API error: {e}")
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multi-agent")
async def api_multi_agent_task(request: MultiAgentRequest, req: Request):
    """Process task using multi-agent system (API endpoint)."""
    try:
        # Get user from session or use provided user_id
        session_user = req.session.get("user")
        user_id = session_user.get("email") if session_user else request.user_id
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Check credits (multi-agent costs 5 credits)
        if not check_credits(user_id, 5):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        # Process with multi-agent system
        result = await multi_agent.process_multi_agent_task(request.task)
        
        # Save chat history
        save_chat_history(user_id, request.task, result.get("final_answer", "Error"))
        
        return {
            "success": True,
            **result,
            "credits_remaining": get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multi-agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/credits")
async def api_get_user_credits(request: Request):
    """Get user's credit balance (API endpoint)."""
    session_user = request.session.get("user")
    if not session_user:
        return {
            "success": False,
            "credits": "0",
            "message": "User not authenticated"
        }
    
    user_id = session_user.get("email")
    credits = get_credits(user_id)
    
    return {
        "success": True,
        "credits": credits,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat()
    }

# Chat Endpoints
@app.post("/chat")
async def chat_completion(chat_request: ChatRequest, request: Request):
    """Generate chat completion using OpenRouter API."""
    try:
        # Get user from session or use provided user_id
        session_user = request.session.get("user")
        user_id = session_user.get("email") if session_user else chat_request.user_id
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Check credits
        if not check_credits(user_id, 1):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OpenRouter API not configured")
        
        # Make request to OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": chat_request.query}],
            "max_tokens": 1000,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
        
        answer = result["choices"][0]["message"]["content"]
        
        # Save chat history
        save_chat_history(user_id, chat_request.query, answer)
        
        return {
            "response": answer,
            "credits_remaining": get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter API error: {e}")
        raise HTTPException(status_code=500, detail="Chat service temporarily unavailable")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_user_chat_history(request: Request, limit: int = Query(20, ge=1, le=100)):
    """Get user's chat history."""
    session_user = request.session.get("user")
    if not session_user:
        raise HTTPException(status_code=401, detail="User not authenticated")
    
    user_id = session_user.get("email")
    history = get_chat_history(user_id, limit)
    
    return {
        "history": history,
        "total": len(history),
        "user_id": user_id
    }

# Image Generation Endpoint
@app.post("/image/generate")
async def generate_image(image_request: ImageRequest, request: Request):
    """Generate image using Stability AI."""
    try:
        # Get user from session or use provided user_id
        session_user = request.session.get("user")
        user_id = session_user.get("email") if session_user else image_request.user_id
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Check credits (image generation costs 3 credits)
        if not check_credits(user_id, 3):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        if not STABILITY_API_KEY:
            raise HTTPException(status_code=500, detail="Stability AI API not configured")
        
        # Make request to Stability AI
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "text_prompts": [{"text": image_request.prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "steps": 30,
            "samples": 1,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(STABILITY_API_URL, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
        
        # Get the base64 image
        image_b64 = result["artifacts"][0]["base64"]
        image_data = base64.b64decode(image_b64)
        
        # Resize image if needed
        resized_image = resize_image(image_data)
        final_b64 = base64.b64encode(resized_image).decode()
        
        return {
            "image": final_b64,
            "prompt": image_request.prompt,
            "credits_remaining": get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"Stability AI error: {e}")
        raise HTTPException(status_code=500, detail="Image generation service temporarily unavailable")
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# VirtuSim Endpoints (kept for compatibility)
@app.get("/virtusim/balance")
async def get_balance(request: Request):
    """Get VirtuSim account balance."""
    try:
        result = await virtusim_service.check_balance()
        return result
    except Exception as e:
        logger.error(f"Balance check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/balance/logs")
async def get_balance_logs(request: Request):
    """Get VirtuSim balance mutation history."""
    try:
        result = await virtusim_service.get_balance_logs()
        return result
    except Exception as e:
        logger.error(f"Balance logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/activity")
async def get_recent_activity(request: Request):
    """Get VirtuSim recent activity."""
    try:
        result = await virtusim_service.get_recent_activity()
        return result
    except Exception as e:
        logger.error(f"Recent activity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/services")
async def get_services(request: Request, country: str = Query("indonesia")):
    """Get available VirtuSim services."""
    try:
        result = await virtusim_service.get_available_services(country)
        return result
    except Exception as e:
        logger.error(f"Services error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/countries")
async def get_countries(request: Request):
    """Get available countries."""
    try:
        result = await virtusim_service.get_countries()
        return result
    except Exception as e:
        logger.error(f"Countries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/operators")
async def get_operators(request: Request, country: str = Query(...)):
    """Get operators for a country."""
    try:
        result = await virtusim_service.get_operators(country)
        return result
    except Exception as e:
        logger.error(f"Operators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/orders/active")
async def get_active_orders(request: Request):
    """Get active VirtuSim orders."""
    try:
        result = await virtusim_service.get_active_orders()
        return result
    except Exception as e:
        logger.error(f"Active orders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/orders/create")
async def create_order(order_request: VirtuSimOrderRequest, request: Request):
    """Create new VirtuSim order."""
    try:
        result = await virtusim_service.create_order(
            order_request.service, 
            order_request.operator
        )
        return result
    except Exception as e:
        logger.error(f"Create order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/orders/reactive")
async def reactive_order(reactive_request: VirtuSimReactiveRequest, request: Request):
    """Reactivate VirtuSim order."""
    try:
        result = await virtusim_service.reactive_order(reactive_request.order_id)
        return result
    except Exception as e:
        logger.error(f"Reactive order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/orders/check")
async def check_order(check_request: VirtuSimCheckRequest, request: Request):
    """Check VirtuSim order status."""
    try:
        result = await virtusim_service.check_order_status(check_request.order_id)
        return result
    except Exception as e:
        logger.error(f"Check order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Multi-Agent Endpoint
@app.post("/multi-agent")
async def multi_agent_task(request: MultiAgentRequest, req: Request):
    """Process task using multi-agent system with 3 different AI models."""
    try:
        # Get user from session or use provided user_id
        session_user = req.session.get("user")
        user_id = session_user.get("email") if session_user else request.user_id
        
        if not user_id:
            raise HTTPException(status_code=401, detail="User not authenticated")
        
        # Check credits (multi-agent costs 5 credits)
        if not check_credits(user_id, 5):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        # Process with multi-agent system
        result = await multi_agent.process_multi_agent_task(request.task)
        
        # Save chat history
        save_chat_history(user_id, request.task, result.get("final_answer", "Error"))
        
        return {
            **result,
            "credits_remaining": get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multi-agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/multi-agent/status")
async def get_multi_agent_status():
    """Get status of multi-agent system."""
    return {
        "status": "active",
        "agents": {
            "analyzer": {
                "model": "anthropic/claude-3.5-sonnet",
                "role": "Analisis dan Pemahaman"
            },
            "researcher": {
                "model": "x-ai/grok-2-mini", 
                "role": "Riset dan Informasi"
            },
            "synthesizer": {
                "model": "meta-llama/llama-3.1-405b-instruct",
                "role": "Sintesis dan Solusi"
            }
        },
        "cost": "5 credits per task",
        "processing": "Sequential (Analyzer → Researcher → Synthesizer)"
    }

# Credits Endpoint
@app.get("/credits")
async def get_user_credits(request: Request):
    """Get user's credit balance."""
    session_user = request.session.get("user")
    if not session_user:
        raise HTTPException(status_code=401, detail="User not authenticated")
    
    user_id = session_user.get("email")
    credits = get_credits(user_id)
    
    return {
        "credits": credits,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(
        "app:app",  # Fixed: changed from "app:app" to match actual filename
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True
    )


