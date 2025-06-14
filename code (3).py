# --- SECTION: IMPORTS & ENVIRONMENT SETUP ---
import base64
import os
import sys
import time
import uuid
import json
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
import uvicorn
import aiosqlite
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_fixed

# --- SECTION: ENVIRONMENT VARIABLES & VALIDATION ---
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///credits.db")
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

if not SESSION_SECRET_KEY:
    print("FATAL: SESSION_SECRET_KEY environment variable is not set. Application cannot start.", file=sys.stderr)
    sys.exit(1)

# --- SECTION: LOGGING SETUP ---
def setup_logging():
    logger.remove()
    logger.add("app.log", rotation="10 MB", compression="zip", level="INFO", format="{time} {level} {message}", enqueue=True)
    logger.add(sys.stderr, level="INFO", format="{time} {level} {message}")

setup_logging()
logger.info(f"FRONTEND_URL: {FRONTEND_URL}")
logger.info(f"DATABASE_URL: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")

# --- SECTION: CONSTANTS & CONFIGURATION ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [FRONTEND_URL, "http://localhost:3000", "https://front-end-bpup.vercel.app", "https://front-end-beta-liard.vercel.app"]
CREDIT_COSTS = {"chat": 1, "image": 2, "agent_system": 5}
GUEST_INITIAL_CREDITS = 25
limiter = Limiter(key_func=get_remote_address)

# --- SECTION: FASTAPI APP INITIALIZATION ---
app = FastAPI(title="Kugy AI Backend (Multi-Agent Vision)", version="4.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=86400, same_site="none", https_only=True)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- SECTION: DATABASE SETUP AND FUNCTIONS ---
async def ensure_db_schema():
    try:
        if DATABASE_URL.startswith("sqlite"):
            async with aiosqlite.connect(DATABASE_URL.replace("sqlite:///", "")) as db:
                await db.execute('''CREATE TABLE IF NOT EXISTS users ( user_id TEXT PRIMARY KEY, user_name TEXT, credits INTEGER, login_streak INTEGER, last_login TEXT, last_guest_timestamp BIGINT, last_reward_date TEXT )''')
                await db.execute('''CREATE TABLE IF NOT EXISTS chat_history ( id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, question TEXT, answer TEXT, created_at TEXT )''')
                await db.executemany('INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?, ?, ?)', [("admin@kugy.ai", "Admin", 999999, 0, datetime.now().isoformat(), 0, ""), ("testadmin", "Test Admin", 999999, 0, datetime.now().isoformat(), 0, "")])
                await db.commit()
        else: # PostgreSQL
            pool = await asyncpg.create_pool(dsn=DATABASE_URL)
            async with pool.acquire() as conn:
                await conn.execute('''CREATE TABLE IF NOT EXISTS users ( user_id TEXT PRIMARY KEY, user_name TEXT, credits INTEGER, login_streak INTEGER, last_login TEXT, last_guest_timestamp BIGINT, last_reward_date TEXT )''')
                await conn.execute('''CREATE TABLE IF NOT EXISTS chat_history ( id SERIAL PRIMARY KEY, user_id TEXT, question TEXT, answer TEXT, created_at TEXT )''')
                await conn.execute('INSERT INTO users VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT DO NOTHING', "admin@kugy.ai", "Admin", 999999, 0, datetime.now().isoformat(), 0, "")
                await conn.execute('INSERT INTO users VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT DO NOTHING', "testadmin", "Test Admin", 999999, 0, datetime.now().isoformat(), 0, "")
            await pool.close()
        logger.info("Database schema checked/initialized successfully")
    except Exception as e:
        logger.error(f"FATAL: Database schema initialization error: {e}"); sys.exit(1)

async def get_db(request: Request) -> Any:
    if DATABASE_URL.startswith("sqlite"):
        try:
            db = await aiosqlite.connect(DATABASE_URL.replace("sqlite:///", "")); yield db
        finally:
            if 'db' in locals() and db: await db.close()
    else:
        async with request.app.state.pool.acquire() as connection: yield connection

async def check_credits(db: Any, user_id: str, need: int) -> bool:
    if not user_id or user_id in ADMIN_USERS: return True
    try:
        if isinstance(db, aiosqlite.Connection):
            async with db.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,)) as c: result = await c.fetchone()
            if not result or result[0] < need: return False
            await db.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id)); await db.commit()
        else:
            credits = await db.fetchval("SELECT credits FROM users WHERE user_id = $1", user_id)
            if credits is None or credits < need: return False
            await db.execute("UPDATE users SET credits = credits - $1 WHERE user_id = $2", need, user_id)
        return True
    except Exception as e:
        logger.error(f"Error checking/deducting credits for {user_id}: {e}"); return False

async def get_credits(db: Any, user_id: str) -> str:
    if not user_id or user_id in ADMIN_USERS: return "∞"
    try:
        if isinstance(db, aiosqlite.Connection):
            async with db.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,)) as c: result = await c.fetchone()
            return str(result[0]) if result else "0"
        else:
            credits = await db.fetchval("SELECT credits FROM users WHERE user_id = $1", user_id)
            return str(credits) if credits is not None else "0"
    except Exception as e:
        logger.error(f"Error getting credits for {user_id}: {e}"); return "0"

async def add_or_init_user(db: Any, user_id: str, user_name: str):
    current_time_iso = datetime.now().isoformat()
    default_credits = 75 if "@" in user_id else GUEST_INITIAL_CREDITS
    try:
        if isinstance(db, aiosqlite.Connection):
            async with db.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)) as c: exists = await c.fetchone()
            if not exists: await db.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)", (user_id, user_name, default_credits, 0, current_time_iso, int(time.time()), "")); await db.commit()
        else:
            exists = await db.fetchval("SELECT 1 FROM users WHERE user_id = $1", user_id)
            if not exists: await db.execute("INSERT INTO users VALUES ($1, $2, $3, $4, $5, $6, $7)", user_id, user_name, default_credits, 0, current_time_iso, int(time.time()), "")
    except Exception as e: logger.error(f"Error initializing user {user_id}: {e}")

async def save_chat_history(db: Any, user_id: str, question: str, answer: str):
    ts = datetime.now().isoformat()
    try:
        if isinstance(db, aiosqlite.Connection):
            await db.execute("INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)", (user_id, question, answer, ts)); await db.commit()
        else: await db.execute("INSERT INTO chat_history (user_id, question, answer, created_at) VALUES ($1, $2, $3, $4)", user_id, question, answer, ts)
    except Exception as e: logger.error(f"Error saving chat history for {user_id}: {e}")

# --- SECTION: AUTHENTICATION & USER MANAGEMENT ---
async def get_current_user(request: Request) -> Dict[str, Any]:
    user = request.session.get("user")
    if not user or not user.get("email"): raise HTTPException(status_code=401, detail="User not authenticated")
    return user

oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(name="google", client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET, server_metadata_url="https://accounts.google.com/.well-known/openid-configuration", client_kwargs={"scope": "openid email profile"})

# --- SECTION: PYDANTIC MODELS ---
class ChatRequest(BaseModel): query: str
class ImageRequest(BaseModel): prompt: str
class AgentRequest(BaseModel):
    task: str
    image_data: Optional[str] = None # Base64 data URL

# --- SECTION: EXTERNAL SERVICES CLASSES ---
class MultiAgentVisionSystem:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = OPENROUTER_API_URL
        self.agents = {
            "analyzer": {"model": "anthropic/claude-3-haiku-20240307", "system_prompt": "You are an expert Analyzer Agent. Analyze the user's text and any image. Describe the image in detail. Break down the task into sub-tasks. Respond ONLY in JSON: {\"image_analysis\": \"...\", \"sub_tasks\": [...], \"main_intent\": \"...\"}"},
            "researcher": {"model": "google/gemini-flash-1.5", "system_prompt": "You are a Researcher Agent. Based on the Analyzer's output, gather concise information for each sub-task. Respond ONLY in JSON: {\"research_findings\": {\"sub_task_1\": \"...\", \"sub_task_2\": \"...\"}}"},
            "synthesizer": {"model": "anthropic/claude-3-haiku-20240307", "system_prompt": "You are a Synthesizer Agent. Combine the analysis and research into a final, comprehensive, user-facing response. Format it using Markdown."}
        }

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _call_agent(self, agent_name: str, messages: List[Dict[str, Any]]) -> str:
        agent_config = self.agents[agent_name]
        full_messages = [{"role": "system", "content": agent_config["system_prompt"]}] + messages
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"model": agent_config["model"], "messages": full_messages, "max_tokens": 2048}
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def process_task_with_vision(self, task: str, image_data: Optional[str]) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting agent task. Image provided: {bool(image_data)}")

        # 1. Analyzer Agent
        analyzer_content = [{"type": "text", "text": f"User task: {task}"}]
        if image_data:
            analyzer_content.insert(0, {"type": "image_url", "image_url": {"url": image_data}})
        
        analyzer_messages = [{"role": "user", "content": analyzer_content}]
        try:
            analyzer_raw_result = await self._call_agent("analyzer", analyzer_messages)
            analyzer_result = json.loads(analyzer_raw_result)
        except Exception as e:
            logger.error(f"Analyzer agent failed: {e}. Raw output: {analyzer_raw_result if 'analyzer_raw_result' in locals() else 'N/A'}")
            return {"success": False, "error": "The Analyzer agent failed to process the request."}

        # 2. Researcher Agent
        researcher_task = f"Analysis:\n{json.dumps(analyzer_result, indent=2)}\n\nPlease conduct your research based on the sub-tasks."
        try:
            researcher_raw_result = await self._call_agent("researcher", [{"role": "user", "content": researcher_task}])
            researcher_result = json.loads(researcher_raw_result)
        except Exception as e:
            logger.error(f"Researcher agent failed: {e}. Raw output: {researcher_raw_result if 'researcher_raw_result' in locals() else 'N/A'}")
            return {"success": False, "error": "The Researcher agent failed to gather information."}

        # 3. Synthesizer Agent
        synthesizer_task = f"Original Task: '{task}'\n\nAnalysis & Image Description:\n{json.dumps(analyzer_result, indent=2)}\n\nResearch Findings:\n{json.dumps(researcher_result, indent=2)}\n\nProvide a complete and final answer."
        try:
            final_answer = await self._call_agent("synthesizer", [{"role": "user", "content": synthesizer_task}])
        except Exception as e:
            logger.error(f"Synthesizer agent failed: {e}"); return {"success": False, "error": "The Synthesizer agent failed to create a final answer."}
            
        return {"success": True, "solution": final_answer, "processing_time": f"{time.time() - start_time:.2f}s", "stages": {"analysis": analyzer_result, "research": researcher_result}}

agent_system = MultiAgentVisionSystem()

# --- SECTION: HELPER FUNCTIONS ---
def resize_image(image_data: bytes, max_size: int = 1024) -> bytes:
    try:
        image = Image.open(BytesIO(image_data)); image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = BytesIO(); image.save(output, format="JPEG", quality=85); return output.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image: {e}"); return image_data

# --- SECTION: API ENDPOINTS ---
@app.get("/", tags=["General"])
async def root(): return {"message": "Kugy AI Backend (Multi-Agent Vision)", "version": "4.0.0"}

@app.get("/health", tags=["General"])
async def health_check(): return {"status": "healthy"}

@app.get("/auth/google", tags=["Authentication"])
async def google_auth(request: Request):
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET): raise HTTPException(500, "Google OAuth not configured")
    return await oauth.google.authorize_redirect(request, f"{BACKEND_URL}/auth/google/callback")

@app.get("/auth/google/callback", tags=["Authentication"])
async def google_callback(request: Request, db: Any = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request); user_info = token.get("userinfo")
        user_email = user_info.get("email"); user_name = user_info.get("name", "User")
        request.session["user"] = {"email": user_email, "name": user_name, "authenticated": True}
        await add_or_init_user(db, user_email, user_name); return RedirectResponse(url=f"{FRONTEND_URL}/?auth=success")
    except Exception as e:
        logger.error(f"Google callback error: {e}"); return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=callback_failed")

@app.post("/auth/logout", tags=["Authentication"])
async def logout(request: Request): request.session.clear(); return {"success": True, "message": "Logged out"}

@app.get("/auth/user", tags=["Authentication"])
async def get_user(user: dict = Depends(get_current_user), db: Any = Depends(get_db)):
    return {"success": True, "user": user, "authenticated": True, "credits": await get_credits(db, user["email"])}

@app.post("/auth/guest", tags=["Authentication"])
async def guest_login(request: Request, db: Any = Depends(get_db)):
    guest_id = f"guest_{int(time.time())}"; request.session["user"] = {"email": guest_id, "name": "Guest User", "authenticated": False}
    await add_or_init_user(db, guest_id, "Guest User")
    return {"success": True, "user": request.session["user"], "authenticated": True, "credits": await get_credits(db, guest_id)}

@app.post("/api/chat", tags=["Chat"])
@limiter.limit("20/minute")
async def api_chat(req: ChatRequest, user: dict = Depends(get_current_user), db: Any = Depends(get_db)):
    user_id = user["email"]
    if not await check_credits(db, user_id, CREDIT_COSTS["chat"]): raise HTTPException(402, "Insufficient credits")
    if not OPENROUTER_API_KEY: raise HTTPException(500, "OpenRouter API not configured")
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}; data = {"model": "google/gemini-flash-1.5", "messages": [{"role": "user", "content": req.query}], "max_tokens": 1500}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(OPENROUTER_API_URL, json=data, headers=headers); response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    await save_chat_history(db, user_id, req.query, answer)
    return {"success": True, "response": answer, "credits_remaining": await get_credits(db, user_id)}

@app.post("/api/agent-system", tags=["AI Agent System"])
@limiter.limit("5/minute")
async def api_agent_system(req: AgentRequest, user: dict = Depends(get_current_user), db: Any = Depends(get_db)):
    user_id = user["email"]
    if not await check_credits(db, user_id, CREDIT_COSTS["agent_system"]): raise HTTPException(402, "Insufficient credits")
    
    result = await agent_system.process_task_with_vision(req.task, req.image_data)
    
    if result.get("success"):
        question = f"Task: {req.task}" + (" (with image)" if req.image_data else "")
        await save_chat_history(db, user_id, question, result.get('solution', 'Error in solution.'))
    
    return {**result, "credits_remaining": await get_credits(db, user_id)}

@app.post("/image/generate", tags=["Image Generation"])
@limiter.limit("10/minute")
async def generate_image(req: ImageRequest, user: dict = Depends(get_current_user), db: Any = Depends(get_db)):
    user_id = user["email"]
    if not await check_credits(db, user_id, CREDIT_COSTS["image"]): raise HTTPException(402, "Insufficient credits")
    if not STABILITY_API_KEY: raise HTTPException(500, "Stability AI API not configured")
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"}
    data = {"text_prompts": [{"text": req.prompt}], "cfg_scale": 7, "height": 512, "width": 512, "steps": 25, "samples": 1}
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(STABILITY_API_URL, json=data, headers=headers); response.raise_for_status()
    image_b64 = response.json()["artifacts"][0]["base64"]; resized_image = resize_image(base64.b64decode(image_b64))
    return {"success": True, "image": base64.b64encode(resized_image).decode(), "credits_remaining": await get_credits(db, user_id)}

# --- SECTION: STARTUP & SHUTDOWN EVENTS ---
@app.on_event("startup")
async def startup_event():
    await ensure_db_schema()
    if not DATABASE_URL.startswith("sqlite"):
        try: app.state.pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=2, max_size=10); logger.info("PostgreSQL connection pool created.")
        except Exception as e: logger.error(f"FATAL: Failed to create PostgreSQL connection pool: {e}"); sys.exit(1)
    else: logger.info("Using aiosqlite for database connections.")

@app.on_event("shutdown")
async def shutdown_event():
    if hasattr(app.state, 'pool') and app.state.pool: await app.state.pool.close(); logger.info("PostgreSQL connection pool closed.")

# --- SECTION: MAIN EXECUTION ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    reload_flag = os.getenv("ENV", "production").lower() == "development"
    uvicorn.run("__main__:app", host="0.0.0.0", port=port, reload=reload_flag)