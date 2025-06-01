import os
import gradio as gr
import requests
from datetime import datetime
import sqlite3
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
import json
import logging
import sys
import time
from starlette.middleware.sessions import SessionMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "supersecretkey"))

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://backend-cb98.onrender.com/auth/callback")
SCOPES = ["openid", "email", "profile"]

client_config = {
    "web": {
        "client_id": CLIENT_ID,
        "project_id": "YOUR_PROJECT_ID",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI]
    }
}

ADMIN_USERS = ["admin@kugy.ai", "testadmin"]

def init_db():
    try:
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
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    finally:
        conn.close()

init_db()

def check_credits(user_id, required_credits):
    if not user_id:
        return False
    if user_id in ADMIN_USERS:
        return True
    try:
        conn = sqlite3.connect("credits.db")
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        if not result or result[0] < required_credits:
            return False
        c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (required_credits, user_id))
        conn.commit()
        return True
    except Exception as e:
        return False
    finally:
        conn.close()

def get_credits(user_id):
    if not user_id:
        return "0 (Invalid User)"
    if user_id in ADMIN_USERS:
        return "∞ (Admin)"
    try:
        conn = sqlite3.connect("credits.db")
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        return str(result[0]) if result else "0"
    except Exception as e:
        return "0"
    finally:
        conn.close()

def top_up_credits(user_id, user_name, amount):
    if not user_id:
        return "User ID tidak valid."
    if user_id.startswith("guest_"):
        return "Guest tidak bisa top up. Login dulu!"
    if user_id in ADMIN_USERS:
        return "Admin unlimited kredit."
    try:
        conn = sqlite3.connect("credits.db")
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (user_id, user_name, 0, 0, datetime.now().strftime("%Y-%m-%d"), 0, ''))
        c.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (amount, user_id))
        conn.commit()
        total = get_credits(user_id)
        return f"Top up sukses! Total kredit: {total}"
    except Exception as e:
        return "Top up gagal."
    finally:
        conn.close()

def check_login_streak(user_id, user_name):
    if not user_id or user_id.startswith("guest_"):
        return "Login/register untuk bonus harian!"
    try:
        conn = sqlite3.connect("credits.db")
        c = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("SELECT login_streak, last_login, last_reward_date FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        bonus_message = ""
        if not result:
            initial_credits = 50  # USER BARU LOGIN GOOGLE DAPAT 50 KREDIT
            c.execute("INSERT INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (user_id, user_name, initial_credits, 1, today, 0, today))
            conn.commit()
            bonus_message = f"Selamat datang, {user_name}! Dapat 50 kredit AI gratis! 😸"
        else:
            streak, last_login, last_reward_date = result
            if last_reward_date == today:
                bonus_message = f"Streak: {streak} hari. Bonus harian sudah diambil."
            else:
                if (datetime.now() - datetime.strptime(last_login, "%Y-%m-%d")).days == 1:
                    streak += 1
                else:
                    streak = 1
                daily_bonus = 1
                streak_bonus = 2 if streak % 5 == 0 else 0
                total_bonus = daily_bonus + streak_bonus
                c.execute("UPDATE users SET credits = credits + ?, login_streak = ?, last_login = ?, last_reward_date = ? WHERE user_id = ?",
                          (total_bonus, streak, today, today, user_id))
                conn.commit()
                bonus_message = f"Login harian! Bonus {daily_bonus} kredit. "
                if streak_bonus:
                    bonus_message += f"Streak {streak} hari! Bonus {streak_bonus} kredit! 🎉 Total: {get_credits(user_id)} 💰"
                else:
                    bonus_message += f"Streak: {streak} hari. Total: {get_credits(user_id)} 💰"
        return bonus_message
    except Exception as e:
        return "Error cek login streak."
    finally:
        conn.close()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
API_KEYS = {"openrouter": os.getenv("OPENROUTER_API_KEY")}

def chat_with_openrouter(message, history, user_id, model_select):
    try:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            return history + [("Kugy.ai: OpenRouter API Key missing! Set OPENROUTER_API_KEY in environment.", None)]
        if not check_credits(user_id, 1):
            return history + [("Kugy.ai: Not enough credits (perlu 1)! Top up! 💰", None)]
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "KugyAI"
        }
        model_map = {
            "OpenRouter (Grok 3 Mini Beta)": "x-ai/grok-3-mini-beta",
            "OpenRouter (Gemini 2.0 Flash)": "google/gemini-flash-1.5"
        }
        model_id = model_map.get(model_select, "x-ai/grok-3-mini-beta")
        messages = [{"role": "user", "content": message}]
        payload = {"model": model_id, "messages": messages, "temperature": 0.7}
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            error_msg = response.text[:200]
            return history + [(f"Kugy.ai: OpenRouter API error (status {response.status_code}): {error_msg}", None)]
        reply = response.json()["choices"][0]["message"]["content"]
        return history + [(f"🤖 {reply}", None)]
    except Exception as e:
        return history + [(f"Kugy.ai: Unexpected error: {str(e)}", None)]

@app.get("/login")
async def login(request: Request):
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
    authorization_url, state = flow.authorization_url(prompt='consent', access_type='offline', include_granted_scopes='true')
    request.session['state'] = state
    return RedirectResponse(url=authorization_url)

@app.get("/auth/callback")
async def auth_callback(request: Request):
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    session_state = request.session.get("state")
    if not state or not code or state != session_state:
        return HTMLResponse("Login gagal, coba ulangi dari awal.", status_code=400)
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
    flow.fetch_token(code=code)
    credentials = flow.credentials
    credentials.refresh(GoogleRequest())
    user_info_resp = requests.get("https://www.googleapis.com/userinfo/v2/me", headers={"Authorization": f"Bearer {credentials.token}"})
    if user_info_resp.status_code != 200:
        return HTMLResponse("Gagal mengambil data Google. Coba lagi.", status_code=400)
    user_info = user_info_resp.json()
    user_id = user_info.get("email")
    user_name = user_info.get("name", user_id.split("@")[0] if user_id else "User")
    request.session['user_id'] = user_id
    request.session['user_name'] = user_name
    # Inisialisasi user baru dengan 50 kredit di check_login_streak (Gradio akan otomatis panggil)
    return RedirectResponse(url="/?tab=Chat")

@app.get("/whoami")
async def whoami(request: Request):
    user_id = request.session.get("user_id")
    user_name = request.session.get("user_name")
    return {"user_id": user_id, "user_name": user_name}

@app.get('/logout')
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")

@app.post('/topup')
async def topup(request: Request, user_id: str = Form(...), user_name: str = Form(...), amount: int = Form(...)):
    hasil = top_up_credits(user_id, user_name, amount)
    return {"message": hasil}

def create_gradio_interface():
    with gr.Blocks(theme="soft") as demo:
        gr.HTML("<div style='font-size:30px;font-weight:bold;color:#4A90E2;text-align:center;'>kugy.ai — Your Cute Assistant 💙</div>")
        user_id_state = gr.State(value=None)
        user_name_state = gr.State(value=None)
        chat_state = gr.State(value={})

        with gr.Tabs(selected="Welcome") as tabs:
            with gr.Tab("Welcome", id="Welcome") as welcome_tab:
                gr.Markdown("### Selamat datang di Kugy.ai!\nLogin dengan Google untuk akses chat AI.")
                with gr.Row():
                    gr.HTML('<a href="/login" class="oauth-link" style="background:#4285F4;color:white;padding:10px 20px;border-radius:5px;text-decoration:none;font-weight:bold;">🔑 Login with Google</a>')
                    guest_button = gr.Button("👤 Guest Mode", variant="secondary")
                welcome_message = gr.Textbox("", label="Status", interactive=False)

                def start_guest_mode(chat_history_state):
                    if not isinstance(chat_history_state, dict):
                        chat_history_state = {}
                    guest_id = f"guest_{int(time.time())}"
                    chat_history_state[guest_id] = []
                    conn = sqlite3.connect("credits.db")
                    c = conn.cursor()
                    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (guest_id, "Guest", 25, 0, datetime.now().strftime("%Y-%m-%d"), int(time.time()), ''))
                    conn.commit()
                    conn.close()
                    return guest_id, chat_history_state, f"Guest Mode ({guest_id}) aktif! Anda dapat 25 kredit.", gr.update(selected="Chat")
                guest_button.click(
                    fn=start_guest_mode,
                    inputs=[chat_state],
                    outputs=[user_id_state, chat_state, welcome_message, tabs]
                )

            with gr.Tab("Chat", id="Chat") as chat_tab:
                with gr.Row():
                    credit_display = gr.Textbox("Credit: 0 💰", interactive=False, label="Credits")
                chatbot = gr.Chatbot(label="", height=500)
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        ["OpenRouter (Grok 3 Mini Beta)", "OpenRouter (Gemini 2.0 Flash)"],
                        value="OpenRouter (Grok 3 Mini Beta)",
                        label="Pilih AI Model"
                    )
                    textbox = gr.Textbox(placeholder="Tulis pesan...", label="")
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Reset Chatbot", variant="secondary")

                def load_chat_data(user_id, chat_history_state):
                    if not user_id:
                        return "Credit: 0 💰", [], chat_history_state
                    if not isinstance(chat_history_state, dict):
                        chat_history_state = {}
                    credits = get_credits(user_id)
                    history = chat_history_state.get(user_id, [])
                    if not history:
                        history = [("🤖 Hi bro! Bagaimana saya bisa bantu?", None)]
                        chat_history_state[user_id] = history
                    return f"Credit: {credits} 💰", history, chat_history_state

                chat_tab.select(
                    fn=load_chat_data,
                    inputs=[user_id_state, chat_state],
                    outputs=[credit_display, chatbot, chat_state]
                )

                def chat(message, history, user_id, chat_history_state, model_select):
                    if not isinstance(history, list):
                        history = []
                    updated_history = chat_with_openrouter(message, history, user_id, model_select)
                    if not isinstance(updated_history, list):
                        updated_history = history
                    if isinstance(chat_history_state, dict) and user_id:
                        chat_history_state[user_id] = updated_history
                    return updated_history, chat_history_state

                send_btn.click(fn=chat, inputs=[textbox, chatbot, user_id_state, chat_state, model_dropdown], outputs=[chatbot, chat_state])
                textbox.submit(fn=chat, inputs=[textbox, chatbot, user_id_state, chat_state, model_dropdown], outputs=[chatbot, chat_state])

                def clear_chat(user_id, chat_history_state):
                    if not isinstance(chat_history_state, dict):
                        chat_history_state = {}
                    if user_id:
                        chat_history_state[user_id] = [("🤖 Chat cleared! Ada lagi?", None)]
                        return chat_history_state[user_id], chat_history_state
                    else:
                        return [("Kugy.ai: Please login or guest mode.", None)], chat_history_state
                clear_btn.click(fn=clear_chat, inputs=[user_id_state, chat_state], outputs=[chatbot, chat_state])

                # Top up form (khusus user login, bukan guest/admin)
                with gr.Row():
                    topup_amount = gr.Number(value=10, label="Nominal Top Up", minimum=1)
                    topup_btn = gr.Button("Top Up Kredit")
                    topup_status = gr.Textbox(label="Status Top Up", interactive=False)

                def do_topup(user_id, user_name, amount):
                    try:
                        msg = top_up_credits(user_id, user_name, int(amount))
                        return msg
                    except Exception as e:
                        return f"Top up gagal: {e}"

                topup_btn.click(fn=do_topup, inputs=[user_id_state, user_name_state, topup_amount], outputs=[topup_status])

            # Tab admin info
            with gr.Tab("Admin Tools", id="Admin"):
                gr.Markdown("### Only for Admins (admin@kugy.ai, testadmin):\nKredit unlimited, fitur top up, dsb.")
                admin_info = gr.Textbox(label="Info User", interactive=False)
                def get_admin_info(user_id):
                    if user_id in ADMIN_USERS:
                        return "Anda admin, kredit unlimited."
                    return "Anda bukan admin."
                admin_info_btn = gr.Button("Cek Status Admin")
                admin_info_btn.click(fn=get_admin_info, inputs=[user_id_state], outputs=[admin_info])

        def check_initial_login():
            # Cek ke endpoint /whoami untuk tahu user udah login Google apa belum
            try:
                resp = requests.get("http://localhost:7860/whoami" if os.getenv("ENV") != "prod" else "https://backend-cb98.onrender.com/whoami", timeout=2)
                data = resp.json()
                user_id = data.get("user_id")
                user_name = data.get("user_name")
                if user_id:
                    streak_msg = check_login_streak(user_id, user_name)
                    return user_id, user_name, {}, f"Welcome back! {streak_msg}", gr.update(selected="Chat")
                else:
                    return "", "", {}, "Please login atau guest mode.", gr.update(selected="Welcome")
            except Exception as e:
                return "", "", {}, "Silakan login atau guest.", gr.update(selected="Welcome")

        demo.load(
            fn=check_initial_login,
            inputs=[],
            outputs=[user_id_state, user_name_state, chat_state, welcome_message, tabs]
        )

    return demo

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)