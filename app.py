import base64
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import asyncpg
import httpx
import uvicorn
import json
from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from loguru import logger
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_fixed

# Environment Variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://front-end-bpup.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", str(uuid.uuid4()))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
VIRTUSIM_API_KEY = os.getenv("VIRTUSIM_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///credits.db")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# Log environment status
logger.info(f"SESSION_SECRET_KEY: {SESSION_SECRET_KEY[:5]}...")
logger.info(f"VIRTUSIM_API_KEY: {VIRTUSIM_API_KEY[:5]}..." if VIRTUSIM_API_KEY else "VIRTUSIM_API_KEY not set")
logger.info(f"FRONTEND_URL: {FRONTEND_URL}")
logger.info(f"DATABASE_URL: {DATABASE_URL}")

# API URLs
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
VIRTUSIM_API_URL = "https://virtusim.com/api/v2/json.php"

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app",
    "https://front-end-beta-liard.vercel.app",
]
CREDIT_COSTS = {
    "chat": 1,
    "image": 3,
    "multi-agent": 5,
}
GUEST_INITIAL_CREDITS = 25
VIRTUSIM_MINIMUM_BALANCE = 1000

# Rate Limiter
limiter = Limiter(key_func=get_remote_address)

def setup_logging():
    """Configure logging with rotation and compression."""
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

async def ensure_db_and_log():
    """Initialize database and log file."""
    try:
        setup_logging()
        if DATABASE_URL.startswith("sqlite"):
            import sqlite3
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
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
                c.execute('''
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        question TEXT,
                        answer TEXT,
                        created_at TEXT
                    )
                ''')
                c.execute('''
                    CREATE TABLE IF NOT EXISTS virtusim_orders (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        order_id TEXT,
                        service TEXT,
                        operator TEXT,
                        number TEXT,
                        status TEXT,
                        created_at TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')
                c.executemany('''
                    INSERT OR IGNORE INTO users 
                    (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', [
                    ("admin@kugy.ai", "Admin", 999999, 0, "2025-06-11 18:12:33", 0, ""),
                    ("testadmin", "Test Admin", 999999, 0, "2025-06-11 18:12:33", 0, "")
                ])
                c.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id)')
                c.execute('CREATE INDEX IF NOT EXISTS idx_virtusim_orders_user_id ON virtusim_orders(user_id)')
                conn.commit()
        else:
            pool = await asyncpg.create_pool(dsn=DATABASE_URL)
            async with pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        user_name TEXT,
                        credits INTEGER,
                        login_streak INTEGER,
                        last_login TEXT,
                        last_guest_timestamp BIGINT,
                        last_reward_date TEXT
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        question TEXT,
                        answer TEXT,
                        created_at TEXT
                    )
                ''')
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS virtusim_orders (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        order_id TEXT,
                        service TEXT,
                        operator TEXT,
                        number TEXT,
                        status TEXT,
                        created_at TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
                    )
                ''')
                await conn.execute('''
                    INSERT INTO users 
                    (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT DO NOTHING
                ''', "admin@kugy.ai", "Admin", 999999, 0, "2025-06-11 18:12:33", 0, "")
                await conn.execute('''
                    INSERT INTO users 
                    (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT DO NOTHING
                ''', "testadmin", "Test Admin", 999999, 0, "2025-06-11 18:12:33", 0, "")
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id)')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_virtusim_orders_user_id ON virtusim_orders(user_id)')
            await pool.close()
        logger.info("Database and log file initialized successfully")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

# Initialize FastAPI
app = FastAPI(
    title="Kugy AI API Backend",
    description="Backend API for Kugy AI with chat, image generation, multi-agent, and VirtuSim integration.",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET_KEY,
    max_age=86400,
    same_site="none",
    https_only=True,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Database Pool
async def get_db_pool():
    if DATABASE_URL.startswith("sqlite"):
        return None
    return await asyncpg.create_pool(dsn=DATABASE_URL)

# Dependency for user authentication
async def get_current_user(request: Request):
    user = request.session.get("user")
    if not user or not user.get("email"):
        raise HTTPException(status_code=401, detail="User not authenticated")
    return user

# Database Functions
async def check_credits(user_id: str, need: int = 1) -> bool:
    """Check and deduct user credits."""
    if not user_id or user_id in ADMIN_USERS:
        return True
    try:
        if DATABASE_URL.startswith("sqlite"):
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
                c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
                result = c.fetchone()
                if not result or result[0] < need:
                    return False
                c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
                conn.commit()
                return True
        else:
            async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                async with pool.acquire() as conn:
                    credits = await conn.fetchval("SELECT credits FROM users WHERE user_id = $1", user_id)
                    if credits is None or credits < need:
                        return False
                    await conn.execute("UPDATE users SET credits = credits - $1 WHERE user_id = $2", need, user_id)
                    return True
    except Exception as e:
        logger.error(f"Error checking credits for {user_id}: {e}")
        return False

async def get_credits(user_id: str) -> str:
    """Get user's credit balance."""
    if not user_id:
        return "0"
    if user_id in ADMIN_USERS:
        return "∞"
    try:
        if DATABASE_URL.startswith("sqlite"):
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
                c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
                result = c.fetchone()
                return str(result[0]) if result else "0"
        else:
            async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                async with pool.acquire() as conn:
                    credits = await conn.fetchval("SELECT credits FROM users WHERE user_id = $1", user_id)
                    return str(credits) if credits is not None else "0"
    except Exception as e:
        logger.error(f"Error getting credits for {user_id}: {e}")
        return "0"

async def add_or_init_user(user_id: str, user_name: str = "User"):
    """Add or initialize user."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_credits = 75 if "@" in user_id else GUEST_INITIAL_CREDITS
    try:
        if DATABASE_URL.startswith("sqlite"):
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
                c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
                if not c.fetchone():
                    c.execute(
                        """INSERT INTO users 
                        (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) 
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (user_id, user_name, default_credits, 0, current_time, int(time.time()), "")
                    )
                    conn.commit()
                    logger.info(f"New user initialized: {user_id}")
        else:
            async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                async with pool.acquire() as conn:
                    exists = await conn.fetchval("SELECT user_id FROM users WHERE user_id = $1", user_id)
                    if not exists:
                        await conn.execute(
                            """INSERT INTO users 
                            (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) 
                            VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                            user_id, user_name, default_credits, 0, current_time, int(time.time()), ""
                        )
                        logger.info(f"New user initialized: {user_id}")
    except Exception as e:
        logger.error(f"Error initializing user {user_id}: {e}")
        raise

async def save_chat_history(user_id: str, question: str, answer: str):
    """Save chat history."""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if DATABASE_URL.startswith("sqlite"):
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, question, answer, current_time)
                )
                conn.commit()
        else:
            async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES ($1, $2, $3, $4)",
                        user_id, question, answer, current_time
                    )
        logger.info(f"Chat history saved for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving chat history for {user_id}: {e}")
        raise

async def get_chat_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Retrieve chat history."""
    try:
        if DATABASE_URL.startswith("sqlite"):
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit)
                )
                history = [{"question": row[0], "answer": row[1], "created_at": row[2]} for row in c.fetchall()][::-1]
        else:
            async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT question, answer, created_at FROM chat_history WHERE user_id = $1 ORDER BY id DESC LIMIT $2",
                        user_id, limit
                    )
                    history = [{"question": row["question"], "answer": row["answer"], "created_at": row["created_at"]} for row in rows][::-1]
        logger.info(f"Retrieved {len(history)} chat entries for user {user_id}")
        return history
    except Exception as e:
        logger.error(f"Error getting chat history for {user_id}: {e}")
        return []

# OAuth Configuration
oauth = OAuth()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        authorize_url="https://accounts.google.com/o/oauth2/auth",
        token_url="https://accounts.google.com/o/oauth2/token",
        client_kwargs={"scope": "openid email profile"}
    )
    logger.info("Google OAuth configured successfully")
else:
    logger.warning("Google OAuth not configured")

# Multi-Agent System - Optimized for 3 AI Models Collaboration
class MultiAgentSystem:
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = OPENROUTER_API_URL
        self.max_iterations = 3
        self.memory = {}
        self.collaboration_history = {}
        
        # Optimized agent configuration with specialized models
        self.agents = {
            "analyzer": {
                "models": {
                    "coding": "deepseek/deepseek-coder",
                    "analysis": "anthropic/claude-3.5-sonnet", 
                    "writing": "meta-llama/llama-3.1-405b-instruct",
                    "general": "anthropic/claude-3.5-sonnet"
                },
                "role": "Strategic Analysis & Task Decomposition",
                "system_prompt": """Anda adalah KugyAgent Analyzer - AI strategis yang menganalisis dan memecah masalah kompleks.

SPESIALISASI:
- Analisis mendalam terhadap task dan konteks
- Decomposisi masalah menjadi sub-komponen yang actionable
- Identifikasi dependencies dan prioritas
- Strategic planning untuk kolaborasi multi-agent

TUGAS UTAMA:
1. Menganalisis kompleksitas dan scope task
2. Mengidentifikasi jenis task (coding/analysis/writing/general)
3. Memecah task menjadi sub-tasks yang spesifik dan terukur
4. Menentukan strategi kolaborasi dengan agent lain
5. Memberikan feedback konstruktif untuk optimasi

FORMAT RESPONSE JSON:
{
    "task_type": "coding|analysis|writing|general",
    "complexity_level": "low|medium|high|expert",
    "sub_tasks": [
        {
            "id": "subtask_1",
            "description": "Detail sub-task",
            "priority": "high|medium|low",
            "estimated_effort": "1-5 scale",
            "dependencies": ["subtask_id"]
        }
    ],
    "analysis": "Analisis mendalam dan reasoning",
    "collaboration_strategy": "Strategi kerja sama dengan agent lain",
    "success_criteria": ["kriteria_1", "kriteria_2"],
    "feedback": "Feedback untuk agent lain (opsional)"
}"""
            },
            "researcher": {
                "models": {
                    "coding": "deepseek/deepseek-coder",
                    "analysis": "meta-llama/llama-3.1-70b-instruct",
                    "writing": "anthropic/claude-3.5-sonnet",
                    "general": "meta-llama/llama-3.1-70b-instruct"
                },
                "role": "Deep Research & Knowledge Synthesis",
                "system_prompt": """Anda adalah KugyAgent Researcher - AI peneliti yang menggali informasi mendalam dan menyediakan knowledge base.

SPESIALISASI:
- Research mendalam untuk setiap sub-task
- Pengumpulan referensi, contoh, dan best practices
- Validasi informasi dan fact-checking
- Knowledge synthesis dari multiple sources

TUGAS UTAMA:
1. Melakukan research mendalam berdasarkan analisis Analyzer
2. Mengumpulkan data, referensi, dan contoh yang relevan
3. Menyediakan context dan background information
4. Memberikan alternative approaches dan solutions
5. Validasi dan quality assurance untuk informasi

FORMAT RESPONSE JSON:
{
    "research_summary": "Ringkasan hasil research",
    "detailed_research": {
        "subtask_id": {
            "data": "Data dan informasi relevan",
            "references": ["ref_1", "ref_2"],
            "examples": ["example_1", "example_2"],
            "best_practices": ["practice_1", "practice_2"]
        }
    },
    "alternative_approaches": ["approach_1", "approach_2"],
    "knowledge_gaps": ["gap_1", "gap_2"],
    "quality_score": "1-10 scale",
    "recommendations": "Rekomendasi untuk Synthesizer",
    "feedback": "Feedback untuk Analyzer atau Synthesizer"
}"""
            },
            "synthesizer": {
                "models": {
                    "coding": "meta-llama/llama-3.1-405b-instruct",
                    "analysis": "anthropic/claude-3.5-sonnet",
                    "writing": "meta-llama/llama-3.1-405b-instruct",
                    "general": "meta-llama/llama-3.1-405b-instruct"
                },
                "role": "Solution Integration & Final Output",
                "system_prompt": """Anda adalah KugyAgent Synthesizer - AI integrator yang menggabungkan analisis dan research menjadi solusi final yang optimal.

SPESIALISASI:
- Integrasi hasil dari Analyzer dan Researcher
- Sintesis informasi menjadi solusi praktis dan actionable
- Quality assurance dan consistency checking
- Final output optimization

TUGAS UTAMA:
1. Mengintegrasikan analisis strategis dan hasil research
2. Menyusun solusi final yang komprehensif dan praktis
3. Memastikan konsistensi dan kualitas output
4. Memberikan penjelasan yang clear dan actionable
5. Melakukan final review dan optimization

FORMAT RESPONSE JSON:
{
    "solution": "Solusi final yang komprehensif",
    "format": "text|code|table|mixed",
    "implementation_steps": [
        {
            "step": 1,
            "action": "Aksi yang harus dilakukan",
            "details": "Detail implementasi",
            "expected_outcome": "Hasil yang diharapkan"
        }
    ],
    "explanation": "Penjelasan mendalam tentang solusi",
    "quality_metrics": {
        "completeness": "1-10 scale",
        "accuracy": "1-10 scale", 
        "practicality": "1-10 scale"
    },
    "validation_checklist": ["check_1", "check_2"],
    "next_steps": ["step_1", "step_2"],
    "feedback": "Feedback untuk iterasi berikutnya"
}"""
            }
        }

    def _detect_task_type(self, task: str) -> str:
        """Enhanced task type detection with better accuracy."""
        task_lower = task.lower()
        
        # Coding keywords - expanded list
        coding_keywords = [
            "code", "program", "python", "javascript", "debug", "function", "class", 
            "algorithm", "api", "database", "sql", "html", "css", "react", "node",
            "git", "github", "deployment", "server", "backend", "frontend", "bug",
            "error", "exception", "syntax", "compile", "build", "test", "unit test"
        ]
        
        # Analysis keywords - expanded list  
        analysis_keywords = [
            "analyze", "analysis", "break down", "evaluate", "assess", "compare",
            "research", "study", "investigate", "examine", "review", "audit",
            "metrics", "data", "statistics", "report", "insights", "trends"
        ]
        
        # Writing keywords - expanded list
        writing_keywords = [
            "write", "create", "draft", "compose", "article", "blog", "content",
            "documentation", "manual", "guide", "tutorial", "story", "essay",
            "proposal", "presentation", "summary", "description", "explanation"
        ]
        
        # Count keyword matches
        coding_score = sum(1 for keyword in coding_keywords if keyword in task_lower)
        analysis_score = sum(1 for keyword in analysis_keywords if keyword in task_lower)
        writing_score = sum(1 for keyword in writing_keywords if keyword in task_lower)
        
        # Determine task type based on highest score
        if coding_score > analysis_score and coding_score > writing_score:
            return "coding"
        elif analysis_score > writing_score:
            return "analysis"
        elif writing_score > 0:
            return "writing"
        else:
            return "general"
    
    def _calculate_collaboration_score(self, task_id: str) -> float:
        """Calculate collaboration effectiveness score."""
        if task_id not in self.memory or not self.memory[task_id]["iterations"]:
            return 0.0
            
        iterations = self.memory[task_id]["iterations"]
        total_score = 0.0
        
        for iteration in iterations:
            # Check if all agents provided valid responses
            agent_responses = 0
            for agent in ["analyzer", "researcher", "synthesizer"]:
                if agent in iteration and "error" not in iteration[agent]:
                    agent_responses += 1
            
            # Calculate iteration score (0-1)
            iteration_score = agent_responses / 3.0
            total_score += iteration_score
            
        return total_score / len(iterations) if iterations else 0.0
    
    def _build_iteration_context(self, task_id: str, iteration_index: int) -> str:
        """Build context from previous iterations for better collaboration."""
        if task_id not in self.memory or iteration_index >= len(self.memory[task_id]["iterations"]):
            return ""
        
        prev_iteration = self.memory[task_id]["iterations"][iteration_index]
        context_parts = []
        
        # Add previous analysis insights
        if "analyzer" in prev_iteration:
            analyzer_data = prev_iteration["analyzer"]
            if "analysis" in analyzer_data:
                context_parts.append(f"Previous Analysis: {analyzer_data['analysis']}")
        
        # Add previous research findings
        if "researcher" in prev_iteration:
            researcher_data = prev_iteration["researcher"]
            if "research_summary" in researcher_data:
                context_parts.append(f"Previous Research: {researcher_data['research_summary']}")
        
        # Add previous solution attempts
        if "synthesizer" in prev_iteration:
            synthesizer_data = prev_iteration["synthesizer"]
            if "solution" in synthesizer_data:
                context_parts.append(f"Previous Solution: {synthesizer_data['solution'][:200]}...")
        
        return "\n\n".join(context_parts)
    
    def _evaluate_agent_response(self, response: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of an agent's response."""
        if "error" in response:
            return {"completeness": 0.0, "relevance": 0.0, "quality": 0.0}
        
        # Basic quality metrics
        completeness = 1.0 if response else 0.0
        relevance = 0.8  # Default relevance score
        quality = 0.7    # Default quality score
        
        # Adjust based on response content
        if isinstance(response, dict):
            # Check for key fields based on agent type
            key_fields = ["analysis", "research_summary", "solution"]
            present_fields = sum(1 for field in key_fields if field in response and response[field])
            completeness = min(1.0, present_fields / len(key_fields) + 0.3)
            
            # Check for feedback (indicates engagement)
            if response.get("feedback"):
                quality += 0.2
            
            # Check for structured data
            if any(isinstance(response.get(field), (list, dict)) for field in response.keys()):
                relevance += 0.1
        
        return {
            "completeness": min(1.0, completeness),
            "relevance": min(1.0, relevance),
            "quality": min(1.0, quality)
        }
    
    def _calculate_iteration_quality(self, iteration_data: Dict[str, Any]) -> float:
        """Calculate overall quality score for an iteration."""
        if "agents_performance" not in iteration_data:
            return 5.0  # Default score
        
        performance_data = iteration_data["agents_performance"]
        total_score = 0.0
        agent_count = 0
        
        for agent_name, metrics in performance_data.items():
            if isinstance(metrics, dict):
                agent_score = (
                    metrics.get("completeness", 0.0) * 0.4 +
                    metrics.get("relevance", 0.0) * 0.3 +
                    metrics.get("quality", 0.0) * 0.3
                )
                total_score += agent_score
                agent_count += 1
        
        # Convert to 1-10 scale
        average_score = (total_score / agent_count) if agent_count > 0 else 0.5
        return average_score * 10.0

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _call_agent(self, agent_name: str, task_type: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Panggil agent."""
        try:
            if not self.api_key:
                raise ValueError(f"OpenRouter API key tidak tersedia untuk {agent_name}")
            agent_config = self.agents[agent_name]
            model = agent_config["models"].get(task_type, "meta-llama/llama-3.1-405b-instruct")
            full_messages = [{"role": "system", "content": agent_config["system_prompt"]}] + messages
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("BACKEND_URL", "https://backend-cb98.onrender.com"),
                "X-Title": "Kugy AI Multi-Agent System"
            }
            payload = {
                "model": model,
                "messages": full_messages,
                "max_tokens": 2000,
                "temperature": 0.7
            }
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {agent_name}: {content}")
                    return {"error": f"Invalid response format from {agent_name}", "raw": content}
        except Exception as e:
            logger.error(f"Error calling {agent_name}: {e}")
            return {"error": str(e)}

    async def process_multi_agent_task(self, task: str, use_multi_agent: bool = True, user_id: str = "guest") -> Dict[str, Any]:
        """Enhanced multi-agent task processing with optimized collaboration."""
        start_time = time.time()
        task_id = str(uuid.uuid4())
        task_type = self._detect_task_type(task)
        
        # Initialize enhanced memory structure
        self.memory[task_id] = {
            "task": task,
            "task_type": task_type,
            "iterations": [],
            "collaboration_metrics": {
                "total_feedback_exchanges": 0,
                "quality_improvements": [],
                "convergence_score": 0.0
            },
            "user_id": user_id,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"🚀 KugyAgent processing task {task_id} (type: {task_type}) with multi-agent: {use_multi_agent}")

        try:
            # Single agent mode (fallback)
            if not use_multi_agent:
                result = await self._call_agent("synthesizer", task_type, [{"role": "user", "content": task}])
                self.memory[task_id]["iterations"].append({"synthesizer": result})
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "task": task,
                    "task_type": task_type,
                    "response": result.get("solution", result.get("error", "No response")),
                    "format": result.get("format", "text"),
                    "explanation": result.get("explanation", ""),
                    "model_used": self.agents["synthesizer"]["models"].get(task_type, "meta-llama/llama-3.1-405b-instruct"),
                    "processing_time": f"{time.time() - start_time:.2f} seconds",
                    "iterations": 1,
                    "collaboration_score": 0.0
                }

            # Multi-agent collaborative processing
            convergence_threshold = 0.8
            quality_threshold = 7.0
            
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"🔄 Iteration {iteration}/{self.max_iterations} for task {task_id}")
                iteration_data = {
                    "iteration_number": iteration,
                    "start_time": time.time(),
                    "agents_performance": {}
                }

                # Phase 1: Strategic Analysis
                analyzer_input = [{"role": "user", "content": f"TASK: {task}\n\nPlease provide strategic analysis and task decomposition."}]
                
                # Add context from previous iterations
                if iteration > 1 and self.memory[task_id]["iterations"]:
                    prev_context = self._build_iteration_context(task_id, iteration - 1)
                    analyzer_input.append({"role": "user", "content": f"PREVIOUS CONTEXT:\n{prev_context}"})
                
                analyzer_result = await self._call_agent("analyzer", task_type, analyzer_input)
                iteration_data["analyzer"] = analyzer_result
                iteration_data["agents_performance"]["analyzer"] = self._evaluate_agent_response(analyzer_result)

                # Phase 2: Deep Research
                research_context = f"""
TASK: {task}
TASK_TYPE: {task_type}
STRATEGIC_ANALYSIS: {json.dumps(analyzer_result, indent=2)}

Please conduct deep research and provide comprehensive knowledge synthesis.
"""
                researcher_input = [{"role": "user", "content": research_context}]
                
                if analyzer_result.get("feedback"):
                    researcher_input.append({"role": "user", "content": f"ANALYZER_FEEDBACK: {analyzer_result['feedback']}"})
                
                researcher_result = await self._call_agent("researcher", task_type, researcher_input)
                iteration_data["researcher"] = researcher_result
                iteration_data["agents_performance"]["researcher"] = self._evaluate_agent_response(researcher_result)

                # Phase 3: Solution Synthesis
                synthesis_context = f"""
ORIGINAL_TASK: {task}
TASK_TYPE: {task_type}

STRATEGIC_ANALYSIS:
{json.dumps(analyzer_result, indent=2)}

RESEARCH_FINDINGS:
{json.dumps(researcher_result, indent=2)}

Please integrate all information and provide the optimal final solution.
"""
                synthesizer_input = [{"role": "user", "content": synthesis_context}]
                
                # Add feedback from previous agents
                feedback_items = []
                if analyzer_result.get("feedback"):
                    feedback_items.append(f"Analyzer: {analyzer_result['feedback']}")
                if researcher_result.get("feedback"):
                    feedback_items.append(f"Researcher: {researcher_result['feedback']}")
                
                if feedback_items:
                    synthesizer_input.append({"role": "user", "content": f"AGENT_FEEDBACK:\n" + "\n".join(feedback_items)})
                
                synthesizer_result = await self._call_agent("synthesizer", task_type, synthesizer_input)
                iteration_data["synthesizer"] = synthesizer_result
                iteration_data["agents_performance"]["synthesizer"] = self._evaluate_agent_response(synthesizer_result)

                # Calculate iteration metrics
                iteration_data["end_time"] = time.time()
                iteration_data["duration"] = iteration_data["end_time"] - iteration_data["start_time"]
                iteration_data["quality_score"] = self._calculate_iteration_quality(iteration_data)
                
                self.memory[task_id]["iterations"].append(iteration_data)
                
                # Update collaboration metrics
                feedback_count = sum(1 for agent in [analyzer_result, researcher_result, synthesizer_result] if agent.get("feedback"))
                self.memory[task_id]["collaboration_metrics"]["total_feedback_exchanges"] += feedback_count
                self.memory[task_id]["collaboration_metrics"]["quality_improvements"].append(iteration_data["quality_score"])

                # Check convergence criteria
                convergence_score = self._calculate_collaboration_score(task_id)
                current_quality = iteration_data["quality_score"]
                
                logger.info(f"📊 Iteration {iteration} - Quality: {current_quality:.2f}, Convergence: {convergence_score:.2f}")
                
                # Early termination conditions
                should_continue = (
                    iteration < self.max_iterations and
                    (convergence_score < convergence_threshold or current_quality < quality_threshold) and
                    (analyzer_result.get("feedback") or researcher_result.get("feedback") or synthesizer_result.get("feedback"))
                )
                
                if not should_continue:
                    logger.info(f"✅ Convergence achieved at iteration {iteration}")
                    break

            # Prepare final response
            final_iteration = self.memory[task_id]["iterations"][-1]
            final_result = final_iteration["synthesizer"]
            collaboration_score = self._calculate_collaboration_score(task_id)
            
            # Store collaboration history for future reference
            self.collaboration_history[task_id] = {
                "task_type": task_type,
                "iterations": len(self.memory[task_id]["iterations"]),
                "collaboration_score": collaboration_score,
                "processing_time": time.time() - start_time,
                "quality_progression": self.memory[task_id]["collaboration_metrics"]["quality_improvements"]
            }

            return {
                "success": True,
                "task_id": task_id,
                "task": task,
                "task_type": task_type,
                "solution": final_result.get("solution", "No solution provided"),
                "format": final_result.get("format", "text"),
                "explanation": final_result.get("explanation", ""),
                "implementation_steps": final_result.get("implementation_steps", []),
                "quality_metrics": final_result.get("quality_metrics", {}),
                "validation_checklist": final_result.get("validation_checklist", []),
                "next_steps": final_result.get("next_steps", []),
                "iterations": len(self.memory[task_id]["iterations"]),
                "collaboration_score": collaboration_score,
                "multi_agent_results": [
                    {
                        "iteration": i + 1,
                        "duration": iter_data.get("duration", 0),
                        "quality_score": iter_data.get("quality_score", 0),
                        "analyzer": iter_data["analyzer"],
                        "researcher": iter_data["researcher"],
                        "synthesizer": iter_data["synthesizer"],
                        "agents_performance": iter_data.get("agents_performance", {})
                    }
                    for i, iter_data in enumerate(self.memory[task_id]["iterations"])
                ],
                "models_used": {
                    "analyzer": self.agents["analyzer"]["models"].get(task_type, "anthropic/claude-3.5-sonnet"),
                    "researcher": self.agents["researcher"]["models"].get(task_type, "meta-llama/llama-3.1-70b-instruct"),
                    "synthesizer": self.agents["synthesizer"]["models"].get(task_type, "meta-llama/llama-3.1-405b-instruct")
                },
                "collaboration_metrics": self.memory[task_id]["collaboration_metrics"],
                "processing_time": f"{time.time() - start_time:.2f} seconds"
            }
        except Exception as e:
            logger.error(f"❌ Error processing task {task_id}: {e}")
            return {
                "success": False,
                "task_id": task_id,
                "task": task,
                "task_type": task_type,
                "error": str(e),
                "iterations": len(self.memory.get(task_id, {}).get("iterations", [])),
                "processing_time": f"{time.time() - start_time:.2f} seconds"
            }

# VirtuSim Service
class VirtuSimService:
    def __init__(self):
        self.api_key = VIRTUSIM_API_KEY
        self.base_url = VIRTUSIM_API_URL

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make VirtuSim API request."""
        try:
            if not self.api_key:
                logger.error("VIRTUSIM_API_KEY not set")
                return {"status": False, "data": {"msg": "API key missing"}}
            params["api_key"] = self.api_key
            params_log = {k: v[:5] + "..." if k == "api_key" else v for k, v in params.items()}
            logger.info(f"VirtuSim Request - Params: {params_log}")
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict) or "status" not in result:
                    raise ValueError("Invalid API response format")
                return result
        except Exception as e:
            logger.error(f"VirtuSim Error: {e}")
            return {"status": False, "data": {"msg": str(e)}}

    async def check_balance(self) -> Dict[str, Any]:
        """Check VirtuSim balance."""
        result = await self._make_request({"action": "balance"})
        if result.get("status") and result.get("data", {}).get("balance", 0) < VIRTUSIM_MINIMUM_BALANCE:
            logger.warning("VirtuSim balance low!")
            if WEBHOOK_URL:
                async with httpx.AsyncClient() as client:
                    await client.post(WEBHOOK_URL, json={"message": f"VirtuSim balance low: {result.get('data', {}).get('balance', 0)}"})
        return result

    async def get_balance_logs(self) -> Dict[str, Any]:
        """Get balance mutation history."""
        return await self._make_request({"action": "balance_logs"})

    async def get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity."""
        return await self._make_request({"action": "recent_activity"})

    async def get_available_services(self, country: str = "indonesia") -> Dict[str, Any]:
        """Get available services."""
        return await self._make_request({"action": "services", "country": country})

    async def get_countries(self) -> Dict[str, Any]:
        """Get available countries."""
        return await self._make_request({"action": "list_country"})

    async def get_operators(self, country: str) -> Dict[str, Any]:
        """Get operators for a country."""
        return await self._make_request({"action": "list_operator", "country": country})

    async def get_active_orders(self) -> Dict[str, Any]:
        """Get active transactions."""
        return await self._make_request({"action": "active_order"})

    async def create_order(self, user_id: str, service: str, operator: str = "any") -> Dict[str, Any]:
        """Create new order."""
        services = await self.get_available_services()
        if not services.get("status") or service not in [s.get("id") for s in services.get("data", [])]:
            return {"status": False, "data": {"msg": f"Invalid service: {service}"}}
        balance = await self.check_balance()
        if not balance.get("status") or balance.get("data", {}).get("balance", 0) < VIRTUSIM_MINIMUM_BALANCE:
            return {"status": False, "data": {"msg": "Insufficient VirtuSim balance"}}
        result = await self._make_request({"action": "order", "service": service, "operator": operator})
        if result.get("status") and "order_id" in result.get("data", {}):
            try:
                if DATABASE_URL.startswith("sqlite"):
                    with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                        c = conn.cursor()
                        c.execute(
                            """INSERT INTO virtusim_orders 
                            (user_id, order_id, service, operator, number, status, created_at) 
                            VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                user_id,
                                result["data"].get("order_id"),
                                service,
                                operator,
                                result["data"].get("number", ""),
                                "pending",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                        )
                        conn.commit()
                else:
                    async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                        async with pool.acquire() as conn:
                            await conn.execute(
                                """INSERT INTO virtusim_orders 
                                (user_id, order_id, service, operator, number, status, created_at) 
                                VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                                user_id,
                                result["data"].get("order_id"),
                                service,
                                operator,
                                result["data"].get("number", ""),
                                "pending",
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            )
                logger.info(f"VirtuSim order saved for user {user_id}")
            except Exception as e:
                logger.error(f"Error saving VirtuSim order for {user_id}: {e}")
        return result

    async def reactive_order(self, order_id: str) -> Dict[str, Any]:
        """Reactivate order."""
        return await self._make_request({"action": "reactive_order", "id": order_id})

    async def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """Check order status."""
        return await self._make_request({"action": "check_order", "id": order_id})

# Initialize Services
virtusim_service = VirtuSimService()
multi_agent = MultiAgentSystem()

# Pydantic Models
class ChatRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None

class VirtuSimOrderRequest(BaseModel):
    service: str
    operator: str = "any"

class VirtuSimCheckRequest(BaseModel):
    order_id: str

class VirtuSimReactiveRequest(BaseModel):
    order_id: str

class MultiAgentRequest(BaseModel):
    task: str
    user_id: Optional[str] = "guest"
    use_multi_agent: bool = True
    max_iterations: int = 2

# Helper Functions
def resize_image(image_data: bytes, max_size: int = 1024) -> bytes:
    """Resize image."""
    try:
        image = Image.open(BytesIO(image_data))
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=85)
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return image_data

# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Kugy AI API Backend",
        "version": "1.0.0",
        "status": "active",
        "features": ["Google OAuth", "Chat", "Multi-Agent", "Image Generation", "VirtuSim"],
        "timestamp": datetime.now().isoformat(),
        "backend_url": os.getenv("BACKEND_URL", "https://backend-cb98.onrender.com"),
        "frontend_url": FRONTEND_URL
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check."""
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
@app.get("/auth/google", tags=["Authentication"])
async def google_auth(request: Request):
    """Initiate Google OAuth."""
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        raise HTTPException(status_code=500, detail="Google OAuth not configured")
    redirect_uri = f"{os.getenv('BACKEND_URL', 'https://backend-cb98.onrender.com')}/auth/google/callback"
    logger.info(f"Initiating Google OAuth with redirect_uri: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback", tags=["Authentication"])
async def google_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
            return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=oauth_not_configured")
        token = await oauth.google.authorize_access_token(request)
        user_info = token.get("userinfo")
        if not user_info:
            logger.error("No user info in token")
            return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=no_user_info")
        user_email = user_info.get("email")
        user_name = user_info.get("name", "User")
        request.session["user"] = {"email": user_email, "name": user_name, "authenticated": True}
        await add_or_init_user(user_email, user_name)
        logger.info(f"User authenticated: {user_email}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?auth=success&user={user_email}")
    except Exception as e:
        logger.error(f"Google callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?auth=error&msg=callback_failed")

@app.post("/auth/logout", tags=["Authentication"])
async def logout(request: Request):
    """Logout user."""
    request.session.clear()
    return {"success": True, "message": "Logged out successfully"}

@app.get("/auth/user", tags=["Authentication"])
async def get_user(user: dict = Depends(get_current_user)):
    """Get user info."""
    user_id = user.get("email")
    credits = await get_credits(user_id)
    return {
        "success": True,
        "user": user,
        "authenticated": True,
        "credits": credits
    }

@app.post("/auth/guest", tags=["Authentication"])
async def guest_login(request: Request):
    """Create guest session."""
    guest_id = f"guest_{int(time.time())}"
    request.session["user"] = {"email": guest_id, "name": "Guest User", "authenticated": False}
    await add_or_init_user(guest_id, "Guest User")
    return {
        "success": True,
        "user": request.session["user"],
        "authenticated": False,
        "credits": await get_credits(guest_id)
    }

# API Endpoints
@app.post("/api/chat", tags=["Chat"])
@limiter.limit("10/minute")
async def api_chat_completion(chat_request: ChatRequest, request: Request, user: dict = Depends(get_current_user)):
    """Generate chat completion."""
    try:
        user_id = user.get("email", chat_request.user_id)
        if not await check_credits(user_id, CREDIT_COSTS["chat"]):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        if not OPENROUTER_API_KEY:
            raise HTTPException(status_code=500, detail="OpenRouter API not configured")
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": chat_request.query}],
            "max_tokens": 1000,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(OPENROUTER_API_URL, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
        answer = result["choices"][0]["message"]["content"]
        await save_chat_history(user_id, chat_request.query, answer)
        return {
            "success": True,
            "response": answer,
            "credits_remaining": await get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter API error: {e}")
        raise HTTPException(status_code=500, detail="Chat service unavailable")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/multi-agent", tags=["Multi-Agent"])
@limiter.limit("5/minute")
async def api_multi_agent_task(multi_request: MultiAgentRequest, request: Request, user: dict = Depends(get_current_user)):
    """Process task with multi-agent system."""
    try:
        user_id = user.get("email", multi_request.user_id)
        if not await check_credits(user_id, CREDIT_COSTS["multi-agent"]):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        result = await multi_agent.process_multi_agent_task(
            multi_request.task,
            multi_request.use_multi_agent,
            user_id
        )
        await save_chat_history(user_id, multi_request.task, result.get("solution", result.get("response", "Error")))
        return {
            "success": result.get("success", False),
            "task_id": result.get("task_id"),
            "task": result.get("task"),
            "task_type": result.get("task_type"),
            "solution": result.get("solution"),
            "format": result.get("format"),
            "explanation": result.get("explanation"),
            "iterations": result.get("iterations"),
            "multi_agent_results": result.get("multi_agent_results"),
            "models_used": result.get("models_used"),
            "processing_time": result.get("processing_time"),
            "credits_remaining": await get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Multi-agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/credits", tags=["Credits"])
async def api_get_user_credits(user: dict = Depends(get_current_user)):
    """Get user credits."""
    user_id = user.get("email")
    credits = await get_credits(user_id)
    return {
        "success": True,
        "credits": credits,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/virtusim/orders/history", tags=["VirtuSim"])
async def get_virtusim_order_history(user: dict = Depends(get_current_user), limit: int = Query(20, ge=1, le=100)):
    """Get VirtuSim order history."""
    user_id = user.get("email")
    try:
        if DATABASE_URL.startswith("sqlite"):
            with sqlite3.connect(DATABASE_URL.replace("sqlite:///", "")) as conn:
                c = conn.cursor()
                c.execute(
                    "SELECT order_id, service, operator, number, status, created_at FROM virtusim_orders WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit)
                )
                history = [
                    {"order_id": row[0], "service": row[1], "operator": row[2], "number": row[3], "status": row[4], "created_at": row[5]}
                    for row in c.fetchall()
                ][::-1]
        else:
            async with asyncpg.create_pool(dsn=DATABASE_URL) as pool:
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        "SELECT order_id, service, operator, number, status, created_at FROM virtusim_orders WHERE user_id = $1 ORDER BY id DESC LIMIT $2",
                        user_id, limit
                    )
                    history = [
                        {"order_id": row["order_id"], "service": row["service"], "operator": row["operator"], "number": row["number"], "status": row["status"], "created_at": row["created_at"]}
                        for row in rows
                    ][::-1]
        logger.info(f"Retrieved {len(history)} VirtuSim orders for user {user_id}")
        return {
            "success": True,
            "history": history,
            "total": len(history),
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error getting VirtuSim order history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# VirtuSim Endpoints
@app.get("/virtusim/balance", tags=["VirtuSim"])
async def get_balance(user: dict = Depends(get_current_user)):
    """Get VirtuSim balance."""
    try:
        result = await virtusim_service.check_balance()
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Balance check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/balance/logs", tags=["VirtuSim"])
async def get_balance_logs(user: dict = Depends(get_current_user)):
    """Get VirtuSim balance logs."""
    try:
        result = await virtusim_service.get_balance_logs()
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Balance logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/activity", tags=["VirtuSim"])
async def get_recent_activity(user: dict = Depends(get_current_user)):
    """Get VirtuSim recent activity."""
    try:
        result = await virtusim_service.get_recent_activity()
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Recent activity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/services", tags=["VirtuSim"])
async def get_services(user: dict = Depends(get_current_user), country: str = Query("indonesia")):
    """Get VirtuSim services."""
    try:
        result = await virtusim_service.get_available_services(country)
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Services error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/countries", tags=["VirtuSim"])
async def get_countries(user: dict = Depends(get_current_user)):
    """Get available countries."""
    try:
        result = await virtusim_service.get_countries()
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Countries error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/operators", tags=["VirtuSim"])
async def get_operators(user: dict = Depends(get_current_user), country: str = Query(...)):
    """Get operators."""
    try:
        result = await virtusim_service.get_operators(country)
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Operators error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/virtusim/orders/active", tags=["VirtuSim"])
async def get_active_orders(user: dict = Depends(get_current_user)):
    """Get active orders."""
    try:
        result = await virtusim_service.get_active_orders()
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Active orders error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/orders/create", tags=["VirtuSim"])
@limiter.limit("5/minute")
async def create_order(order_request: VirtuSimOrderRequest, request: Request, user: dict = Depends(get_current_user)):
    """Create VirtuSim order."""
    try:
        user_id = user.get("email")
        result = await virtusim_service.create_order(user_id, order_request.service, order_request.operator)
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Create order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/orders/reactive", tags=["VirtuSim"])
@limiter.limit("5/minute")
async def reactive_order(reactive_request: VirtuSimReactiveRequest, request: Request, user: dict = Depends(get_current_user)):
    """Reactivate order."""
    try:
        result = await virtusim_service.reactive_order(reactive_request.order_id)
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Reactive order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/virtusim/orders/check", tags=["VirtuSim"])
@limiter.limit("10/minute")
async def check_order(check_request: VirtuSimCheckRequest, request: Request, user: dict = Depends(get_current_user)):
    """Check order status."""
    try:
        result = await virtusim_service.check_order_status(check_request.order_id)
        return {
            "success": result.get("status", False),
            "data": result.get("data", {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Check order error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional Endpoints
@app.get("/chat/history", tags=["Chat"])
async def get_user_chat_history(user: dict = Depends(get_current_user), limit: int = Query(20, ge=1, le=100)):
    """Get chat history."""
    user_id = user.get("email")
    history = await get_chat_history(user_id, limit)
    return {
        "success": True,
        "history": history,
        "total": len(history),
        "user_id": user_id
    }

@app.post("/image/generate", tags=["Image"])
@limiter.limit("5/minute")
async def generate_image(image_request: ImageRequest, request: Request, user: dict = Depends(get_current_user)):
    """Generate image."""
    try:
        user_id = user.get("email", image_request.user_id)
        if not await check_credits(user_id, CREDIT_COSTS["image"]):
            raise HTTPException(status_code=402, detail="Insufficient credits")
        if not STABILITY_API_KEY:
            raise HTTPException(status_code=500, detail="Stability AI API not configured")
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(STABILITY_API_URL, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()
        image_b64 = result["artifacts"][0]["base64"]
        image_data = base64.b64decode(image_b64)
        resized_image = resize_image(image_data)
        final_b64 = base64.b64encode(resized_image).decode()
        return {
            "success": True,
            "image": final_b64,
            "prompt": image_request.prompt,
            "credits_remaining": await get_credits(user_id),
            "timestamp": datetime.now().isoformat()
        }
    except httpx.HTTPStatusError as e:
        logger.error(f"Stability AI error: {e}")
        raise HTTPException(status_code=500, detail="Image generation unavailable")
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/multi-agent/status", tags=["Multi-Agent"])
async def get_multi_agent_status():
    """Get multi-agent status."""
    return {
        "success": True,
        "status": "active",
        "agents": {name: {"models": config["models"], "role": config["role"]} for name, config in multi_agent.agents.items()},
        "cost": f"{CREDIT_COSTS['multi-agent']} credits per task",
        "processing": "Iterative collaboration (Analyzer → Researcher → Synthesizer)"
    }

# Initialize Database
@app.on_event("startup")
async def startup_event():
    """Initialize database and logging."""
    await ensure_db_and_log()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        access_log=True
    )
