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

# Environment Variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://front-end-bpup.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "changeme_secret_key_123456")
logger.info(f"Using SESSION_SECRET_KEY: {SESSION_SECRET_KEY[:5]}...") # Log first 5 chars for security
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
VIRTUSIM_API_KEY = os.getenv("VIRTUSIM_API_KEY")
logger.info(f"VIRTUSIM_API_KEY: {VIRTUSIM_API_KEY[:5]}..." if VIRTUSIM_API_KEY else "VIRTUSIM_API_KEY is not set")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app",
    "*"  # Allow all origins for deployment
]
CALLBACK_URL = "https://backend-cb98.onrender.com/auth/google/callback"
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

# Initialize FastAPI
app = FastAPI(title="VirtuSim API Backend", version="1.0.0")

# Configure middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=3600,
    same_site="none"  # Required for cross-site cookies in some scenarios
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

# Configure OAuth
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

# VirtuSim Service
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

    async def set_order_status(self, order_id: str, status: int) -> Dict[str, Any]:
        """
        Change order status.
        Status codes:
        1 = Ready
        2 = Cancel
        3 = Resend
        4 = Completed
        """
        return await self._make_request({
            "action": "set_status",
            "id": order_id,
            "status": status
        })

    async def get_order_history(self) -> Dict[str, Any]:
        """Get order history."""
        return await self._make_request({"action": "history"})

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
                "model": "x-ai/grok-3-mini-beta",
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
                "HTTP-Referer": "https://virtusim-backend.onrender.com",
                "X-Title": "VirtuSim Multi-Agent System"
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
                        "agent": "Researcher (Grok-3-Mini)",
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
                "models_used": ["Claude-3.5-Sonnet", "Grok-3-Mini", "Llama-3.1-405B"]
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
virtusim = VirtuSimService()
multi_agent = MultiAgentSystem()

# Pydantic Models
class ChatRequest(BaseModel):
    question: str
    user_id: str = "guest"

class MultiAgentRequest(BaseModel):
    task: str
    user_id: str = "guest"
    use_multi_agent: bool = True

class ImageRequest(BaseModel):
    prompt: str
    user_id: str = "guest"

class OrderRequest(BaseModel):
    service: str
    operator: str = "any"

class OrderStatusRequest(BaseModel):
    order_id: str
    status: int

# API Routes

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "VirtuSim API Backend is running!", 
        "version": "1.0.0",
        "features": [
            "Single AI Chat (/chat)",
            "Multi-Agent System (/multi-agent)", 
            "Image Generation (/generate-image)",
            "VirtuSim Integration (/virtusim/*)",
            "User Management (/auth/*)"
        ],
        "multi_agent_models": ["Claude-3.5-Sonnet", "Grok-3-Mini", "Llama-3.1-405B"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Authentication Routes
@app.get("/auth/google")
async def google_auth(request: Request):
    """Initiate Google OAuth."""
    if not oauth.google:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    redirect_uri = CALLBACK_URL
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def google_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        if not oauth.google:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
            
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
        
        if user_info:
            user_id = user_info.get("email")
            user_name = user_info.get("name", "User")
            
            # Initialize user in database
            add_or_init_user(user_id, user_name)
            
            # Store user info in session
            request.session["user"] = {
                "user_id": user_id,
                "user_name": user_name,
                "credits": get_credits(user_id)
            }
            
            logger.info(f"User authenticated: {user_id}")
            return RedirectResponse(url=f"{FRONTEND_URL}?auth=success")
        else:
            logger.error("Failed to get user info from Google")
            return RedirectResponse(url=f"{FRONTEND_URL}?auth=error")
            
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}?auth=error")

@app.get("/auth/user")
async def get_user(request: Request):
    """Get current user info."""
    user = request.session.get("user")
    if user:
        # Update credits in real-time
        user["credits"] = get_credits(user["user_id"])
        return user
    return {"user_id": None, "user_name": None, "credits": "0"}

@app.post("/auth/logout")
async def logout(request: Request):
    """Logout user."""
    request.session.clear()
    return {"message": "Logged out successfully"}

@app.post("/auth/guest")
async def guest_login(request: Request):
    """Initialize guest user."""
    guest_id = f"guest_{int(time.time())}"
    add_or_init_user(guest_id, "Guest")
    
    request.session["user"] = {
        "user_id": guest_id,
        "user_name": "Guest",
        "credits": get_credits(guest_id)
    }
    
    logger.info(f"Guest user created: {guest_id}")
    return {"user_id": guest_id, "user_name": "Guest", "credits": get_credits(guest_id)}

# VirtuSim API Routes

# Account Routes
@app.get("/virtusim/balance")
async def get_balance():
    """Get VirtuSim account balance."""
    try:
        result = await virtusim.check_balance()
        return result
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/balance-logs")
async def get_balance_logs():
    """Get balance mutation history."""
    try:
        result = await virtusim.get_balance_logs()
        return result
    except Exception as e:
        logger.error(f"Error getting balance logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/recent-activity")
async def get_recent_activity():
    """Get recent activity."""
    try:
        result = await virtusim.get_recent_activity()
        return result
    except Exception as e:
        logger.error(f"Error getting recent activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Service Routes
@app.get("/virtusim/services")
async def get_services(country: str = "indonesia"):
    """Get available services."""
    try:
        result = await virtusim.get_available_services(country)
        return result
    except Exception as e:
        logger.error(f"Error getting services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/countries")
async def get_countries():
    """Get list of available countries."""
    try:
        result = await virtusim.get_countries()
        return result
    except Exception as e:
        logger.error(f"Error getting countries: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/operators")
async def get_operators(country: str):
    """Get list of operators for a country."""
    try:
        result = await virtusim.get_operators(country)
        return result
    except Exception as e:
        logger.error(f"Error getting operators: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Transaction Routes
@app.get("/virtusim/active-orders")
async def get_active_orders():
    """Get active orders."""
    try:
        result = await virtusim.get_active_orders()
        return result
    except Exception as e:
        logger.error(f"Error getting active orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/create-order")
async def create_order(order: OrderRequest):
    """Create new order."""
    try:
        result = await virtusim.create_order(order.service, order.operator)
        return result
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/reactive-order")
async def reactive_order(order_id: str):
    """Reactivate existing order."""
    try:
        result = await virtusim.reactive_order(order_id)
        return result
    except Exception as e:
        logger.error(f"Error reactivating order: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/check-order")
async def check_order_status(order_id: str):
    """Check order status."""
    try:
        result = await virtusim.check_order_status(order_id)
        return result
    except Exception as e:
        logger.error(f"Error checking order status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/set-order-status")
async def set_order_status(request: OrderStatusRequest):
    """Set order status."""
    try:
        result = await virtusim.set_order_status(request.order_id, request.status)
        return result
    except Exception as e:
        logger.error(f"Error setting order status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/order-history")
async def get_order_history():
    """Get order history."""
    try:
        result = await virtusim.get_order_history()
        return result
    except Exception as e:
        logger.error(f"Error getting order history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat and AI Routes
@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Chat with AI using OpenRouter."""
    try:
        user_id = request.user_id
        
        # Check credits
        if not check_credits(user_id, 1):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://virtusim-backend.onrender.com",
            "X-Title": "VirtuSim Chat"
        }
        
        payload = {
            "model": "microsoft/wizardlm-2-8x22b",
            "messages": [
                {"role": "user", "content": request.question}
            ]
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OPENROUTER_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            
            # Save chat history
            save_chat_history(user_id, request.question, answer)
            
            return {
                "answer": answer,
                "credits_remaining": get_credits(user_id)
            }
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-agent")
async def multi_agent_task(request: MultiAgentRequest):
    """Process task using multi-agent system with 3 different AI models."""
    try:
        user_id = request.user_id
        
        # Check credits (multi-agent costs more - 3 API calls)
        if not check_credits(user_id, 3):
            raise HTTPException(status_code=402, detail="Insufficient credits (3 required for multi-agent)")
        
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OpenRouter API key not configured")
        
        # Process task using multi-agent system
        result = await multi_agent.process_multi_agent_task(request.task)
        
        # Save chat history with final result
        save_chat_history(user_id, request.task, result.get("final_answer", "Multi-agent processing completed"))
        
        # Add credits info to result
        result["credits_remaining"] = get_credits(user_id)
        result["credits_used"] = 3
        
        return result
        
    except Exception as e:
        logger.error(f"Error in multi-agent processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    """Generate image using Stability AI."""
    try:
        user_id = request.user_id
        
        # Check credits (image generation costs more)
        if not check_credits(user_id, 5):
            raise HTTPException(status_code=402, detail="Insufficient credits (5 required)")
        
        if not STABILITY_API_KEY:
            raise HTTPException(status_code=500, detail="Stability API key not configured")
        
        headers = {
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "text_prompts": [{"text": request.prompt}],
            "cfg_scale": 7,
            "height": 1024,
            "width": 1024,
            "samples": 1,
            "steps": 30
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(STABILITY_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            image_base64 = result["artifacts"][0]["base64"]
            
            return {
                "image": f"data:image/png;base64,{image_base64}",
                "credits_remaining": get_credits(user_id)
            }
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/multi-agent/status")
async def get_multi_agent_status():
    """Get status of multi-agent system."""
    return {
        "system": "Multi-Agent AI System",
        "agents": [
            {
                "name": "Analyzer",
                "model": "anthropic/claude-3.5-sonnet",
                "role": "Analisis dan Pemahaman",
                "description": "Menganalisis task dan memecahnya menjadi komponen detail"
            },
            {
                "name": "Researcher", 
                "model": "x-ai/grok-3-mini-beta",
                "role": "Riset dan Informasi",
                "description": "Mencari informasi mendalam dan memberikan data pendukung"
            },
            {
                "name": "Synthesizer",
                "model": "meta-llama/llama-3.1-405b-instruct", 
                "role": "Sintesis dan Solusi",
                "description": "Menggabungkan hasil menjadi solusi komprehensif"
            }
        ],
        "workflow": "Sequential Processing: Analyzer → Researcher → Synthesizer",
        "credits_required": 3,
        "api_provider": "OpenRouter",
        "status": "Ready" if OPENROUTER_API_KEY else "API Key Required"
    }

@app.get("/chat-history")
async def get_user_chat_history(user_id: str, limit: int = 20):
    """Get user's chat history."""
    try:
        history = get_chat_history(user_id, limit)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Admin Routes
@app.get("/admin/users")
async def get_all_users(request: Request):
    """Get all users (admin only)."""
    user = request.session.get("user")
    if not user or user.get("user_id") not in ADMIN_USERS:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT user_id, user_name, credits, login_streak, last_login FROM users")
            users = [
                {
                    "user_id": row[0],
                    "user_name": row[1],
                    "credits": row[2],
                    "login_streak": row[3],
                    "last_login": row[4]
                }
                for row in c.fetchall()
            ]
            return {"users": users}
    except Exception as e:
        logger.error(f"Error getting users: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/add-credits")
async def add_credits_to_user(request: Request, user_id: str, credits: int):
    """Add credits to user (admin only)."""
    admin_user = request.session.get("user")
    if not admin_user or admin_user.get("user_id") not in ADMIN_USERS:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (credits, user_id))
            conn.commit()
            
            if c.rowcount == 0:
                raise HTTPException(status_code=404, detail="User not found")
            
            return {"message": f"Added {credits} credits to {user_id}"}
    except Exception as e:
        logger.error(f"Error adding credits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# ASGI application for deployment
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))