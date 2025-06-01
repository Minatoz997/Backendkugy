import os
import gradio as gr
import requests
import sqlite3
import sys
import time
import logging
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse

from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request as GoogleRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_CLIENT_REDIRECT_URI", os.getenv("GOOGLE_REDIRECT_URI", "https://backend-cb98.onrender.com/auth/callback"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]

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

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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

def check_login_streak(user_id, user_name):
    if not user_id or user_id.startswith("guest_"):
        return "Login/register for daily bonuses!"
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT login_streak, last_login, last_reward_date FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    if not result:
        initial_credits = 10
        c.execute("INSERT INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (user_id, user_name, initial_credits, 1, today, 0, today))
        conn.commit()
        conn.close()
        return f"Welcome, {user_name}! Got {initial_credits} free credits! 😸"
    streak, last_login, last_reward_date = result
    if last_reward_date == today:
        conn.close()
        return f"Streak: {streak} days. Daily bonus already claimed today."
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
    conn.close()
    msg = f"Daily login! Got {daily_bonus} credit. "
    if streak_bonus:
        msg += f"Streak {streak} days! Bonus {streak_bonus} credits! 🎉 Total: {get_credits(user_id)} 💰"
    else:
        msg += f"Streak: {streak} days. Total: {get_credits(user_id)} 💰"
    return msg

def chat_with_openrouter(messages, user_id, model_select):
    if not OPENROUTER_API_KEY:
        return messages + [{"role": "assistant", "content": "Kugy.ai: OpenRouter API Key missing! Set OPENROUTER_API_KEY in environment."}]
    if not check_credits(user_id, 1):
        return messages + [{"role": "assistant", "content": "Kugy.ai: Not enough credits (need 1)! Top up! 💰"}]
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
    model_id = model_map.get(model_select, "x-ai/grok-3-mini-beta")
    payload = {"model": model_id, "messages": messages, "temperature": 0.7}
    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return messages + [{"role": "assistant", "content": f"Kugy.ai: OpenRouter API error (status {response.status_code}): {response.text[:200]}"}]
        reply = response.json()["choices"][0]["message"]["content"]
        return messages + [{"role": "assistant", "content": reply}]
    except Exception as e:
        return messages + [{"role": "assistant", "content": f"Kugy.ai: Unexpected error with OpenRouter: {str(e)}"}]

def handle_google_callback(code, state):
    try:
        if not code or not state:
            return None, None, "Error: Missing code or state in callback.", gr.update(selected="Welcome"), True, False, ""
        flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
        flow.fetch_token(code=code)
        credentials = flow.credentials
        credentials.refresh(GoogleRequest())
        user_info_resp = requests.get("https://www.googleapis.com/userinfo/v2/me", headers={"Authorization": f"Bearer {credentials.token}"})
        user_info = user_info_resp.json()
        user_id = user_info.get("email")
        if not user_id:
            return None, None, "Error: Email tidak ditemukan di Google response.", gr.update(selected="Welcome"), True, False, ""
        user_name = user_info.get("name", user_id.split("@")[0])
        badge_html = f'<span class="user-badge">👤 {user_id}</span>' if user_id else ""
        return user_id, user_name, "Login berhasil! Kamu sudah bisa chat.", gr.update(selected="Chat"), False, True, badge_html
    except Exception as e:
        return None, None, f"Error during login: {str(e)}", gr.update(selected="Welcome"), True, False, ""

@app.get('/auth/login')
async def login():
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI)
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        prompt='consent',
        include_granted_scopes='true'
    )
    return RedirectResponse(url=authorization_url)

@app.get('/auth/callback')
async def oauth_callback(request: Request):
    code = request.query_params.get('code')
    state = request.query_params.get('state')
    if not code or not state:
        return HTMLResponse("Missing code or state parameter.", status_code=400)
    gradio_base_url = "/"
    redirect_url = f"{gradio_base_url}?code={code}&state={state}"
    return RedirectResponse(url=redirect_url)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error: An unexpected error occurred."}
    )

def create_gradio_interface():
    with gr.Blocks(
        css="""
        #title { font-size: 30px; font-weight: bold; color: #4A90E2; text-align: center; }
        .credit-display { background: linear-gradient(to right, #FFD700, #FFA500); color: black; padding: 8px 12px; border-radius: 15px; text-align: center; font-weight: bold; }
        .user-badge { position: absolute; top: 15px; right: 30px; background: #f0f0f0; color: #333; padding: 8px 16px; border-radius: 20px; font-size: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.07);}
        """,
        theme="soft",
    ) as demo:
        user_id_state = gr.State(value=None)
        user_name_state = gr.State(value=None)
        chat_state = gr.State(value={})

        with gr.Row():
            gr.HTML("<div id='title'>kugy.ai — Your Cute Assistant 💙</div>")
            user_badge = gr.HTML("", elem_id="user_badge")

        with gr.Tabs() as tabs:
            with gr.Tab("Welcome", id="Welcome"):
                welcome_col = gr.Column(visible=True)
                with welcome_col:
                    gr.Markdown("### Welcome to Kugy.ai!\nLogin dengan Google untuk simpan history & dapat bonus harian, atau coba Guest Mode (history sementara).")
                    with gr.Row():
                        gr.HTML('<a href="/auth/login" style="text-decoration:none">'
                                '<button style="background:#4285F4;color:white;padding:10px 20px;border:none;border-radius:5px;font-weight:bold;">'
                                'Login with Google</button></a>')
                        guest_btn = gr.Button("👤 Guest Mode")
                    welcome_message = gr.Textbox("", label="Status", interactive=False)
                    code_input = gr.Textbox(placeholder="Paste the 'code' from URL", label="Code")
                    state_input = gr.Textbox(placeholder="Paste the 'state' from URL (should be \'xyz\')", label="State")
                    callback_btn = gr.Button("Submit Callback")
                success_col = gr.Column(visible=False)
                with success_col:
                    success_msg = gr.Markdown("")

            with gr.Tab("Chat", id="Chat"):
                with gr.Row():
                    credit_display = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], label="Credits")
                chatbot = gr.Chatbot(label="", height=500, type='messages')
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
                    if not user_id:
                        return "Credit: 0 💰", [], chat_history_state
                    if not isinstance(chat_history_state, dict):
                        chat_history_state = {}
                    credits = get_credits(user_id)
                    history = chat_history_state.get(user_id, [])
                    if not history:
                        history = [{"role": "assistant", "content": "Hi bro! How can I help?"}]
                        chat_history_state[user_id] = history
                    return f"Credit: {credits} 💰", history, chat_history_state

                def chat_tab_select(user_id_state, chat_state):
                    return load_chat_data(user_id_state, chat_state)
                tabs.select(
                    fn=chat_tab_select,
                    inputs=[user_id_state, chat_state],
                    outputs=[credit_display, chatbot, chat_state],
                    queue=False
                )

                def chat(message, history, user_id, chat_history_state, model_select):
                    if not isinstance(history, list):
                        history = []
                    messages = history + [{"role": "user", "content": message}]
                    updated_history = chat_with_openrouter(messages, user_id, model_select)
                    chat_history_state[user_id] = updated_history
                    return updated_history, chat_history_state

                send_btn.click(fn=chat, inputs=[textbox, chatbot, user_id_state, chat_state, model_dropdown], outputs=[chatbot, chat_state])
                textbox.submit(fn=chat, inputs=[textbox, chatbot, user_id_state, chat_state, model_dropdown], outputs=[chatbot, chat_state])

                def clear_chat(user_id, chat_history_state):
                    if not isinstance(chat_history_state, dict):
                        chat_history_state = {}
                    chat_history_state[user_id] = [{"role": "assistant", "content": "Chat cleared! What's next?"}]
                    return chat_history_state[user_id], chat_history_state

                clear_btn.click(fn=clear_chat, inputs=[user_id_state, chat_state], outputs=[chatbot, chat_state])

            with gr.Tab("Debug", id="Debug"):
                debug_output = gr.Textbox(label="Debug Info", interactive=False)
                def get_debug_info():
                    debug_info = (
                        f"Python Version: {sys.version}\n"
                        f"Gradio Version: {gr.__version__}\n"
                        f"GOOGLE_CLIENT_ID: {'[Value Hidden]' if os.getenv('GOOGLE_CLIENT_ID') else 'Not found'}\n"
                        f"GOOGLE_CLIENT_SECRET: {'[Value Hidden]' if os.getenv('GOOGLE_CLIENT_SECRET') else 'Not found'}\n"
                        f"OPENROUTER_API_KEY: {'[SET]' if os.getenv('OPENROUTER_API_KEY') else 'Not found'}\n"
                    )
                    return debug_info
                debug_btn = gr.Button("Check API Keys")
                debug_btn.click(fn=get_debug_info, inputs=[], outputs=[debug_output])

        def check_initial_login(user_id_state, user_name_state, chat_state):
            user_id = user_id_state.value if hasattr(user_id_state, "value") else user_id_state
            user_name = user_name_state.value if hasattr(user_name_state, "value") else user_name_state
            badge_html = f'<span class="user-badge">👤 {user_id}</span>' if user_id else ""
            if user_id:
                streak_msg = check_login_streak(user_id, user_name)
                return user_id, user_name, chat_state, f"Welcome back! {streak_msg}", gr.update(selected="Chat"), gr.update(visible=False), gr.update(visible=True), badge_html
            else:
                return "", "", chat_state, "Please login or select guest mode.", gr.update(selected="Welcome"), gr.update(visible=True), gr.update(visible=False), ""

        def guest_mode(chat_history_state):
            guest_id = f"guest_{int(time.time())}"
            if not isinstance(chat_history_state, dict):
                chat_history_state = {}
            chat_history_state[guest_id] = [{"role": "assistant", "content": "Hi bro! How can I help?"}]
            conn = sqlite3.connect("credits.db")
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (guest_id, "Guest", 25, 0, datetime.now().strftime("%Y-%m-%d"), int(time.time()), ''))
            conn.commit()
            conn.close()
            return guest_id, chat_history_state, "Guest Mode aktif! Kamu mendapat 25 credit gratis.", gr.update(selected="Chat"), gr.update(visible=False), gr.update(visible=True), ""

        # guest_btn tetap tombol Gradio, login pakai HTML <a>
        guest_btn.click(
            guest_mode,
            inputs=[chat_state],
            outputs=[user_id_state, chat_state, welcome_message, tabs, welcome_col, success_col, user_badge]
        )

        callback_btn.click(
            handle_google_callback,
            inputs=[code_input, state_input],
            outputs=[user_id_state, user_name_state, welcome_message, tabs, welcome_col, success_col, user_badge]
        )

        demo.load(
            fn=check_initial_login,
            inputs=[user_id_state, user_name_state, chat_state],
            outputs=[user_id_state, user_name_state, chat_state, welcome_message, tabs, welcome_col, success_col, user_badge]
        )
    return demo

try:
    gradio_app = create_gradio_interface()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
except Exception as e:
    logger.error(f"Failed to mount Gradio app: {str(e)}")
    raise

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)