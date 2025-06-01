import os
import gradio as gr
import requests
from datetime import datetime
import sqlite3
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_CLIENT_REDIRECT_URI", os.getenv("GOOGLE_REDIRECT_URI", "https://backend-cb98.onrender.com/auth/callback"))
SCOPES = [
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid"
]
client_config = {
    "web": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
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
        logger.info(f"Admin user {user_id}: bypassing credit check")
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
        logger.error(f"Error checking credits for user {user_id}: {str(e)}")
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
        logger.error(f"Error getting credits for user {user_id}: {str(e)}")
        return "0"
    finally:
        conn.close()

def top_up_credits(user_id, user_name, amount):
    if not user_id or user_id.startswith("guest_"):
        return "Kugy.ai: Guests can't top up. Please register/login."
    if user_id in ADMIN_USERS:
        return "Kugy.ai: Admin has unlimited credits! 😎"
    try:
        conn = sqlite3.connect("credits.db")
        c = conn.cursor()
        c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (user_id, user_name, 0, 0, datetime.now().strftime("%Y-%m-%d"), 0, ''))
        c.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (amount, user_id))
        conn.commit()
        return f"Kugy.ai: Added {amount} credits! Total: {get_credits(user_id)} 💰"
    except Exception as e:
        logger.error(f"Error topping up credits for user {user_id}: {str(e)}")
        return "Kugy.ai: Failed to top up credits."
    finally:
        conn.close()

def check_login_streak(user_id, user_name):
    if not user_id or user_id.startswith("guest_"):
        return "Login/register for daily bonuses!"
    try:
        conn = sqlite3.connect("credits.db")
        c = conn.cursor()
        today = datetime.now().strftime("%Y-%m-%d")
        c.execute("SELECT login_streak, last_login, last_reward_date FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        bonus_message = ""
        if not result:
            initial_credits = 10
            c.execute("INSERT INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (user_id, user_name, initial_credits, 1, today, 0, today))
            conn.commit()
            bonus_message = f"Welcome, {user_name}! Got {initial_credits} free credits! 😸"
        else:
            streak, last_login, last_reward_date = result
            if last_reward_date == today:
                bonus_message = f"Streak: {streak} days. Daily bonus already claimed today."
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
                bonus_message = f"Daily login! Got {daily_bonus} credit. "
                if streak_bonus:
                    bonus_message += f"Streak {streak} days! Bonus {streak_bonus} credits! 🎉 Total: {get_credits(user_id)} 💰"
                else:
                    bonus_message += f"Streak: {streak} days. Total: {get_credits(user_id)} 💰"
        return bonus_message
    except Exception as e:
        logger.error(f"Error checking login streak for user {user_id}: {str(e)}")
        return "Error checking login streak."
    finally:
        conn.close()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

API_KEYS = {
    "openrouter": os.getenv("OPENROUTER_API_KEY")
}

def validate_api_keys():
    openrouter_status = "Not set"
    if API_KEYS["openrouter"]:
        headers = {"Authorization": f"Bearer {API_KEYS['openrouter']}", "Content-Type": "application/json", "HTTP-Referer": "http://localhost", "X-Title": "KugyAI"}
        payload = {"model": "x-ai/grok-3-mini-beta", "messages": [{"role": "user", "content": "test"}], "temperature": 0.7}
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=5)
            openrouter_status = f"Status {response.status_code}: {'Valid' if response.status_code == 200 else response.text[:100]}"
        except Exception as e:
            openrouter_status = f"Error: {str(e)[:100]}"
    logger.info(f"OPENROUTER_API_KEY: {openrouter_status}")
    return openrouter_status,

try:
    openrouter_status = validate_api_keys()
except Exception as e:
    logger.error(f"Failed to validate API keys: {str(e)}")
    openrouter_status = "Error",

def chat_with_openrouter(message, history, user_id, model_select):
    try:
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_key:
            logger.error("OpenRouter API Key is missing or not set in environment!")
            return history + [("Kugy.ai: OpenRouter API Key missing! Set OPENROUTER_API_KEY in environment.", None)]
        if not check_credits(user_id, 1):
            return history + [("Kugy.ai: Not enough credits (need 1)! Top up! 💰", None)]
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
            logger.error(f"OpenRouter error: {error_msg}")
            return history + [(f"Kugy.ai: OpenRouter API error (status {response.status_code}): {error_msg}", None)]
        reply = response.json()["choices"][0]["message"]["content"]
        return history + [(f"🤖 {reply}", None)]
    except Exception as e:
        logger.error(f"Unexpected error with OpenRouter: {str(e)}")
        return history + [(f"Kugy.ai: Unexpected error with OpenRouter: {str(e)}", None)]

def handle_google_callback(code, state):
    try:
        if not code or not state:
            return None, None, "Error: Missing code or state in callback.", gr.update(selected="Welcome")
        flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
        flow.fetch_token(code=code)
        credentials = flow.credentials
        credentials.refresh(GoogleRequest())
        user_info_resp = requests.get("https://www.googleapis.com/userinfo/v2/me", headers={"Authorization": f"Bearer {credentials.token}"})
        logger.info(f"Google response: {user_info_resp.status_code}, {user_info_resp.text}")
        if user_info_resp.status_code != 200:
            return None, None, f"Error: Gagal mengambil data Google: {user_info_resp.text}", gr.update(selected="Welcome")
        user_info = user_info_resp.json()
        logger.info(f"User info: {user_info}")
        user_id = user_info.get("email")
        if not user_id:
            return None, None, "Error: Email tidak ditemukan di Google response.", gr.update(selected="Welcome")
        user_name = user_info.get("name", user_id.split("@")[0])
        logger.info(f"User logged in: {user_id} ({user_name})")
        return user_id, user_name, "Login berhasil! Kamu sudah bisa chat.", gr.update(selected="Chat")
    except Exception as e:
        logger.error(f"Error in Google callback: {str(e)}")
        return None, None, f"Error during login: {str(e)}", gr.update(selected="Welcome")

@app.get('/auth/login')
async def login():
    try:
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI,
        )
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            prompt='consent',
            include_granted_scopes='true'
        )
        logger.info(f"Redirecting user to Google OAuth: {authorization_url}")
        return RedirectResponse(url=authorization_url)
    except Exception as e:
        logger.error(f"Error in OAuth login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error: Failed to start OAuth login")

@app.get('/auth/callback')
async def oauth_callback(request: Request):
    try:
        code = request.query_params.get('code')
        state = request.query_params.get('state')
        logger.info(f"[OAuth Callback] code={code}, state={state}")
        if not code or not state:
            return HTMLResponse("Missing code or state parameter.", status_code=400)
        gradio_base_url = "/"
        redirect_url = f"{gradio_base_url}?code={code}&state={state}"
        return RedirectResponse(url=redirect_url)
    except Exception as e:
        logger.error(f"Error in oauth_callback: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error: Failed to handle OAuth callback")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error: An unexpected error occurred."}
    )

def create_gradio_interface():
    with gr.Blocks(
        css="""
        #title { font-size: 30px; font-weight: bold; color: #4A90E2; text-align: center; }
        .credit-display { background: linear-gradient(to right, #FFD700, #FFA500); color: black; padding: 8px 12px; border-radius: 15px; text-align: center; font-weight: bold; }
        .oauth-link { display: inline-block; background-color: #4285F4; color: white; padding: 10px 20px; border-radius: 5px; text-decoration: none; font-weight: bold; }
        .oauth-link:hover { background-color: #3267D6; }
        .user-badge { position: absolute; top: 15px; right: 30px; background: #f0f0f0; color: #333; padding: 8px 16px; border-radius: 20px; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);}
        """,
        theme="soft",
    ) as demo:
        user_id_state = gr.State(value=None)
        user_name_state = gr.State(value=None)
        chat_state = gr.State(value={})

        # Header: judul dan badge user login di pojok atas
        with gr.Row():
            gr.HTML("<div id='title'>kugy.ai — Your Cute Assistant 💙</div>")
            user_badge = gr.HTML("", elem_id="user_badge")

        with gr.Tabs() as tabs:
            with gr.Tab("Welcome", id="Welcome") as welcome_tab:
                # Konten welcome/guest hanya muncul jika user belum login
                welcome_box = gr.Column(visible=True)
                with welcome_box:
                    gr.Markdown("### Welcome to Kugy.ai!")
                    gr.Markdown("Login dengan Google untuk simpan history & dapat bonus harian, atau coba Guest Mode (history sementara).")
                    with gr.Row():
                        gr.HTML(f'<a class="oauth-link" href="/auth/login">Login with Google</a>')
                        guest_button = gr.Button("👤 Guest Mode", variant="secondary")
                    welcome_message = gr.Textbox("", label="Status", interactive=False)
                    with gr.Row():
                        code_input = gr.Textbox(placeholder="Paste the 'code' from URL", label="Code")
                        state_input = gr.Textbox(placeholder="Paste the 'state' from URL (should be \'xyz\')", label="State")
                        callback_btn = gr.Button("Submit Callback")
                # Jika user sudah login, tampilkan info login berhasil saja
                success_msg = gr.Textbox("Login berhasil! Kamu sudah bisa chat.", visible=False, interactive=False, label="Status")

                # Fungsi guest
                def start_guest_mode(chat_history_state):
                    try:
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
                        return guest_id, chat_history_state, f"Guest Mode ({guest_id}) active! You have 25 free credits.", gr.update(selected="Chat")
                    except Exception as e:
                        logger.error(f"Error in start_guest_mode: {str(e)}")
                        return None, chat_history_state, f"Error starting guest mode: {str(e)}", gr.update(selected="Welcome")

                guest_button.click(
                    fn=start_guest_mode,
                    inputs=[chat_state],
                    outputs=[user_id_state, chat_state, welcome_message, tabs]
                )

                callback_btn.click(
                    fn=handle_google_callback,
                    inputs=[code_input, state_input],
                    outputs=[user_id_state, user_name_state, welcome_message, tabs]
                )

            with gr.Tab("Chat", id="Chat") as chat_tab:
                with gr.Row():
                    credit_display = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], label="Credits")
                chatbot = gr.Chatbot(label="", height=500)
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        ["OpenRouter (Grok 3 Mini Beta)", "OpenRouter (Gemini 2.0 Flash)"],
                        value="OpenRouter (Grok 3 Mini Beta)",
                        label="Choose AI Model"
                    )
                    textbox = gr.Textbox(placeholder="Type your message...", label="")
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Reset Chatbot", variant="secondary")

                def load_chat_data(user_id, chat_history_state):
                    try:
                        if not user_id:
                            return "Credit: 0 💰", [], chat_history_state
                        if not isinstance(chat_history_state, dict):
                            chat_history_state = {}
                        credits = get_credits(user_id)
                        history = chat_history_state.get(user_id, [])
                        if not history:
                            history = [("🤖 Hi bro! How can I help?", None)]
                            chat_history_state[user_id] = history
                        return f"Credit: {credits} 💰", history, chat_history_state
                    except Exception as e:
                        logger.error(f"Error loading chat data: {str(e)}")
                        return "Credit: Error", [], chat_history_state

                chat_tab.select(
                    fn=load_chat_data,
                    inputs=[user_id_state, chat_state],
                    outputs=[credit_display, chatbot, chat_state]
                )

                def chat(message, history, user_id, chat_history_state, model_select):
                    try:
                        if not isinstance(history, list):
                            history = []
                        updated_history = chat_with_openrouter(message, history, user_id, model_select)
                        if not isinstance(updated_history, list):
                            updated_history = history
                        chat_history_state[user_id] = updated_history
                        return updated_history, chat_history_state
                    except Exception as e:
                        logger.error(f"Error in chat: {str(e)}")
                        return history + [("Kugy.ai: Error processing chat.", None)], chat_history_state

                send_btn.click(fn=chat, inputs=[textbox, chatbot, user_id_state, chat_state, model_dropdown], outputs=[chatbot, chat_state])
                textbox.submit(fn=chat, inputs=[textbox, chatbot, user_id_state, chat_state, model_dropdown], outputs=[chatbot, chat_state])

                def clear_chat(user_id, chat_history_state):
                    try:
                        if not isinstance(chat_history_state, dict):
                            chat_history_state = {}
                        chat_history_state[user_id] = [("🤖 Chat cleared! What's next?", None)]
                        return chat_history_state[user_id], chat_history_state
                    except Exception as e:
                        logger.error(f"Error clearing chat: {str(e)}")
                        return [("Kugy.ai: Error clearing chat.", None)], chat_history_state

                clear_btn.click(fn=clear_chat, inputs=[user_id_state, chat_state], outputs=[chatbot, chat_state])

            with gr.Tab("Debug", id="Debug"):
                debug_output = gr.Textbox(label="Debug Info", interactive=False)

                def get_debug_info():
                    try:
                        debug_info = (
                            f"Python Version: {sys.version}\n"
                            f"Gradio Version: {gr.__version__}\n"
                            f"FastAPI Version: {FastAPI.__version__}\n"
                            f"GOOGLE_CLIENT_ID: {'[Value Hidden]' if os.getenv('GOOGLE_CLIENT_ID') else 'Not found'}\n"
                            f"GOOGLE_CLIENT_SECRET: {'[Value Hidden]' if os.getenv('GOOGLE_CLIENT_SECRET') else 'Not found'}\n"
                            f"OPENROUTER_API_KEY: {openrouter_status}\n"
                        )
                        return debug_info
                    except Exception as e:
                        logger.error(f"Error getting debug info: {str(e)}")
                        return f"Error getting debug info: {str(e)}"

                debug_btn = gr.Button("Check API Keys")
                debug_btn.click(fn=get_debug_info, inputs=[], outputs=[debug_output])

        def check_initial_login(user_id_state, user_name_state, chat_history_state):
            try:
                user_id = user_id_state.value if hasattr(user_id_state, "value") else user_id_state
                user_name = user_name_state.value if hasattr(user_name_state, "value") else user_name_state
                # Sembunyikan Welcome box & tampilkan badge user jika sudah login
                badge_html = f'<span class="user-badge">👤 {user_id}</span>' if user_id else ""
                # Welcome tab
                if user_id:
                    # Sembunyikan seluruh konten welcome, tampilkan pesan sukses
                    demo.get_component("welcome_box").update(visible=False)
                    demo.get_component("success_msg").update(visible=True, value="Login berhasil! Kamu sudah bisa chat.")
                    demo.get_component("user_badge").update(value=badge_html)
                    streak_msg = check_login_streak(user_id, user_name)
                    return user_id, user_name, chat_history_state, f"Welcome back! {streak_msg}", gr.update(selected="Chat")
                else:
                    demo.get_component("welcome_box").update(visible=True)
                    demo.get_component("success_msg").update(visible=False)
                    demo.get_component("user_badge").update(value="")
                    return "", "", chat_history_state, "Please login or select guest mode.", gr.update(selected="Welcome")
            except Exception as e:
                logger.error(f"Error in check_initial_login: {str(e)}")
                return "", "", chat_history_state, f"Error checking login: {str(e)}", gr.update(selected="Welcome")

        demo.load(
            fn=check_initial_login,
            inputs=[user_id_state, user_name_state, chat_state],
            outputs=[user_id_state, user_name_state, chat_state, welcome_message, tabs]
        )

    logger.info("Gradio interface created successfully.")
    return demo

try:
    logger.info("Attempting to create Gradio interface...")
    gradio_app = create_gradio_interface()
    logger.info("Attempting to mount Gradio app to FastAPI...")
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    logger.info("Gradio app mounted successfully.")
except Exception as e:
    logger.error(f"Failed to mount Gradio app: {str(e)}")
    raise

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    logger.info(f"Starting server on port: {port}")
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Failed to start Uvicorn server: {str(e)}")
        raise