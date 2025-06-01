
import os
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image


import json
import time
import random
from datetime import datetime, timedelta

from time import sleep
from functools import wraps
import tempfile
import traceback
import sqlite3
import sys
import io
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest

# Setup FastAPI app
app = FastAPI()

# Setup OAuth
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "https://backend-cb98.onrender.com/auth/callback")
SCOPES = ["profile", "email"]

if not CLIENT_ID or not CLIENT_SECRET:
    raise ValueError("GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET not set in environment variables.")

client_config = {
    "web": {
        "client_id": CLIENT_ID,
        "project_id": "YOUR_PROJECT_ID",  # Replace with your Google Cloud project ID
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI]
    }
}

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

ADMIN_USERS = ["admin@kugy.ai", "testadmin"]

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

def check_credits(user_id, required_credits):
    if not user_id:
        return False

    if user_id in ADMIN_USERS:
        print(f"Admin user {user_id}: bypassing credit check")
        return True

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()

    if not result or result[0] < required_credits:
        return False

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (required_credits, user_id))
    conn.commit()
    conn.close()
    return True

def get_credits(user_id):
    if not user_id:
        return "0 (Invalid User)"
    if user_id in ADMIN_USERS:
        return "∞ (Admin)"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def get_credits_numeric(user_id):
    if not user_id:
        return 0
    if user_id in ADMIN_USERS:
        return float('inf')

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def top_up_credits(user_id, user_name, amount):
    if not user_id or user_id.startswith("guest_"):
        return "Kugy.ai: Guests can't top up. Please register/login."
    if user_id in ADMIN_USERS:
        return "Kugy.ai: Admin has unlimited credits! 😎"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (user_id, user_name, 0, 0, datetime.now().strftime("%Y-%m-%d"), 0, ''))
    c.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (amount, user_id))
    conn.commit()
    conn.close()
    return f"Kugy.ai: Added {amount} credits! Total: {get_credits(user_id)} 💰"

def check_login_streak(user_id, user_name):
    if not user_id or user_id.startswith("guest_"):
        return "Login/register for daily bonuses!"

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
        streak = 1
    else:
        streak, last_login, last_reward_date = result
        last_login_date = datetime.strptime(last_login, "%Y-%m-%d")
        today_date = datetime.strptime(today, "%Y-%m-%d")

        if last_reward_date == today:
            bonus_message = f"Streak: {streak} days. Daily bonus already claimed today."
        else:
            if (today_date - last_login_date).days == 1:
                streak += 1
            elif (today_date - last_login_date).days > 1:
                streak = 1
            else:
                pass

            daily_bonus = 1
            streak_bonus = 0
            if streak % 5 == 0:
                streak_bonus = 2
            total_bonus = daily_bonus + streak_bonus

            c.execute("UPDATE users SET credits = credits + ?, login_streak = ?, last_login = ?, last_reward_date = ? WHERE user_id = ?",
                      (total_bonus, streak, today, today, user_id))
            conn.commit()

            bonus_message = f"Daily login! Got {daily_bonus} credit. "
            if streak_bonus > 0:
                bonus_message += f"Streak {streak} days! Bonus {streak_bonus} credits! 🎉 Total: {get_credits(user_id)} 💰"
            else:
                bonus_message += f"Streak: {streak} days. Total: {get_credits(user_id)} 💰"

    conn.close()
    return bonus_message

STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
STABILITY_IMAGE_TO_IMAGE_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
PIAPI_API_URL = "https://api.piapi.ai/api/v1/task"

API_KEYS = {
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "stability": os.getenv("STABILITY_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "piapi": os.getenv("PIAPI_API_KEY")
}

missing_keys = [k for k, v in API_KEYS.items() if not v and k != "stability"]
if missing_keys:
    print(f"Warning: Missing API keys: {', '.join(missing_keys)}")



MAX_CACHE_SIZE = 50
image_cache = {}

def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    def decorator(func):
        last_called = [0.0]
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = period - elapsed
            if left_to_wait > 0:
                sleep(left_to_wait)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

def get_theme():
    hour = datetime.now().hour
    return "dark" if 18 <= hour <= 6 else "default"

@rate_limit(max_per_minute=60)
def generate_image(prompt, user_id):
    if not check_credits(user_id, 3):
        return None, "Kugy.ai: Not enough credits (need 3)! Top up now~ 💰"
    cache_key = prompt.lower().strip()
    if cache_key in image_cache:
        return image_cache[cache_key], "Kugy.ai: Cached image for you! 😘"
    if not API_KEYS["stability"]:
        return None, "Kugy.ai: Stability AI token missing! Add it to Secrets!"
    api_url = STABILITY_API_URL
    headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1, "steps": 30}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print(f"Stability Text-to-Image status: {response.status_code}")
        if response.status_code == 200:
            resp_data = response.json()
            if "artifacts" in resp_data and resp_data["artifacts"]:
                base64_img = resp_data["artifacts"][0]["base64"]
                image = Image.open(BytesIO(base64.b64decode(base64_img)))
                if len(image_cache) >= MAX_CACHE_SIZE:
                    image_cache.pop(next(iter(image_cache)))
                image_cache[cache_key] = image
                return image, "Kugy.ai: Here's your cute image! 😺"
            else:
                print(f"No artifacts in response: {resp_data}")
                return None, "Kugy.ai: Failed to get image from Stability (no artifacts)!"
        elif response.status_code == 401:
            return None, "Kugy.ai: Stability AI error (401 Unauthorized). Check API Key!"
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Stability Error: {error_detail}")
            return None, f"Kugy.ai: Stability AI error (status {response.status_code}): {error_detail[:200]}"
    except requests.exceptions.Timeout:
        print("Timeout generating image")
        return None, "Kugy.ai: Timeout contacting Stability AI. Try again later."
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Image generation error: {str(e)}"

def image_to_base64(image):
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="PNG")
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string

def save_image_for_download(image):
    try:
        if not image:
            print("Error: Image for download is empty!")
            return None
        print("Saving image for download...")
        if image.mode not in ['RGB', 'RGBA', 'L']:
            image = image.convert('RGB')
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            print(f"Image saved at: {tmp.name}")
            return tmp.name
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        traceback.print_exc()
        return None

@rate_limit(max_per_minute=60)
def piapi_task(base_image, target_image, task_type, user_id):
    if not check_credits(user_id, 2):
        return "Kugy.ai: Not enough credits (need 2)! Top up now~ 💰"
    api_key = API_KEYS["piapi"]
    if not api_key:
        return "Kugy.ai: Piapi API Key missing! Set PIAPI_API_KEY!"

    try:
        max_res = 1024
        if base_image.width > max_res or base_image.height > max_res:
            base_image = base_image.resize((min(base_image.width, max_res), min(base_image.height, max_res)), Image.Resampling.LANCZOS)
        if target_image.width > max_res or target_image.height > max_res:
            target_image = target_image.resize((min(target_image.width, max_res), max_res), Image.Resampling.LANCZOS)
        base_image = base_image.convert("RGB")
        target_image = target_image.convert("RGB")
        base_image_base64 = image_to_base64(base_image)
        target_image_base64 = image_to_base64(target_image)
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": "Qubico/image-toolkit",
            "task_type": task_type,
            "input": {
                "target_image": base_image_base64,
                "swap_image": target_image_base64
            },
            "config": {
                "service_mode": "async",
                "webhook_config": {
                    "endpoint": "",
                    "secret": ""
                }
            }
        }
        print("Sending request to Piapi...")
        response = requests.post(PIAPI_API_URL, headers=headers, json=payload, timeout=60)
        print(f"Piapi status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            job_id = result.get("data", {}).get("task_id")
            if job_id:
                print(f"Piapi task ID: {job_id}")
                poll_url = f"https://api.piapi.ai/api/v1/task/{job_id}"
                for attempt in range(15):
                    print(f"Polling Piapi (attempt {attempt+1})...")
                    time.sleep(5)
                    poll_response = requests.get(poll_url, headers=headers, timeout=30)
                    if poll_response.status_code == 200:
                        poll_result = poll_response.json()
                        status = poll_result.get("data", {}).get("status")
                        print(f"Piapi task status: {status}")
                        if status == "completed":
                            output = poll_result.get("data", {}).get("output", {})
                            if isinstance(output, dict):
                                image_url = output.get("image_url") or output.get("url")
                                image_base64 = output.get("image_base64")
                                if image_url:
                                    print(f"Fetching image: {image_url}")
                                    image_response = requests.get(image_url, timeout=30)
                                    if image_response.status_code == 200:
                                        return Image.open(BytesIO(image_response.content))
                                    else:
                                        return f"Kugy.ai: Failed to fetch Piapi image (status {image_response.status_code})!"
                                elif image_base64:
                                    print("Decoding Base64 image...")
                                    try:
                                        image_data = base64.b64decode(image_base64)
                                        return Image.open(BytesIO(image_data))
                                    except Exception as e:
                                        return f"Kugy.ai: Failed to decode Piapi image_base64: {str(e)}!"
                                    traceback.print_exc()
                                else:
                                    return f"Kugy.ai: Piapi output missing URL/Base64! Output: {output}"
                            elif isinstance(output, str):
                                image_url = output
                                print(f"Fetching image: {image_url}")
                                image_response = requests.get(image_url, timeout=30)
                                if image_response.status_code == 200:
                                    return Image.open(BytesIO(image_response.content))
                                else:
                                    return f"Kugy.ai: Failed to fetch Piapi image (status {image_response.status_code})!"
                            return f"Kugy.ai: Invalid Piapi output format! Output: {output}"
                        elif status == "failed":
                            error_message = poll_result.get('data', {}).get('message', 'No error details')
                            print(f"Piapi task failed: {error_message}")
                            return f"Kugy.ai: Piapi task failed: {error_message}."
                    else:
                        print(f"Polling failed: {poll_response.status_code}")
                return "Kugy.ai: Timeout polling Piapi task! Try again later."
            else:
                print(f"No task ID: {result}")
                return "Kugy.ai: No task ID from Piapi!"
        error_detail = response.text
        try:
            error_json = response.json()
            error_detail = error_json.get('message', error_detail)
        except json.JSONDecodeError:
            print(f"Piapi error: {error_detail}")
            return f"Kugy.ai: Piapi error (status {response.status_code}): {error_detail[:200]}."
    except requests.exceptions.Timeout:
        print("Timeout during Piapi request")
        return "Kugy.ai: Timeout contacting Piapi."
    except Exception as e:
        print(f"Piapi task error: {str(e)}")
        traceback.print_exc()
        return f"Kugy.ai: Piapi error: {str(e)}"

def swap_couple(master_couple_img, face_pacar_img, face_kamu_img, task_type, user_id):
    if not check_credits(user_id, 4):
        return None, "Kugy.ai: Not enough credits (need 4)! Top up now~ 💰"
    if not master_couple_img or not face_pacar_img or not face_kamu_img:
        return None, "Kugy.ai: Please upload all images! 😺"
    print("Starting couple swap step 1 (Pacar face)...")
    step1_result = piapi_task(master_couple_img, face_pacar_img, task_type, user_id)
    if isinstance(step1_result, str):
        return None, f"Kugy.ai: Step 1 failed (swap pacar): {step1_result}"
    elif not isinstance(step1_result, Image.Image):
        return None, "Kugy.ai: Step 1 result is not a valid image."
    print("Starting couple swap step 2 (Kamu face)...")
    final_img_result = piapi_task(step1_result, face_kamu_img, task_type, user_id)
    if isinstance(final_img_result, str):
        return None, f"Kugy.ai: Step 2 failed (swap kamu): {final_img_result}"
    elif not isinstance(final_img_result, Image.Image):
        return None, "Kugy.ai: Step 2 result is not a valid image."
    return final_img_result, "Kugy.ai: Couple photo ready! So sweet! 🥰"

@rate_limit(max_per_minute=60)
def apply_ghibli_style(uploaded_img, user_id):
    if not check_credits(user_id, 5):
        print(f"User {user_id} lacks credits for Ghibli (needs 5).")
        return None, "Kugy.ai: Not enough credits (need 5)! Top up now~ 💰"
    try:
        if not isinstance(uploaded_img, Image.Image):
            print(f"Invalid Ghibli input: {type(uploaded_img)}")
            return None, "Kugy.ai: Invalid Ghibli image input! 😿"
        uploaded_img = uploaded_img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        print(f"Resized Ghibli input to: {uploaded_img.size}")
        buffer = BytesIO()
        uploaded_img.save(buffer, format="PNG")
        buffer.seek(0)
        headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Accept": "application/json"}
        ghibli_prompt = (
            "Studio Ghibli style by Hayao Miyazaki, transform photo into anime character, soft pastel colors, "
            "hand-drawn details, lush natural background (e.g., cherry blossoms, forest, river), "
            "whimsical magical atmosphere, gentle warm lighting, expressive eyes, smooth skin, flowing hair, "
            "highly detailed, seamless transition from photo to anime"
        )
        files = {'init_image': ('image.png', buffer, 'image/png')}
        data = {
            "text_prompts[0][text]": ghibli_prompt,
            "cfg_scale": "10",
            "steps": "50",
            "image_strength": "0.45",
            "style_preset": "anime"
        }
        print("Sending Ghibli style request to Stability AI...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, files=files, data=data, timeout=90)
        print(f"Ghibli status: {response.status_code}")
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Ghibli error: {error_detail}")
            return None, f"Kugy.ai: Ghibli error (status {response.status_code}): {error_detail[:200]}"
        resp_data = response.json()
        if 'artifacts' not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in Ghibli response!")
            return None, "Kugy.ai: Failed to get Ghibli result from Stability!"
        generated_img_data = base64.b64decode(resp_data["artifacts"][0]["base64"])
        generated_img = Image.open(BytesIO(generated_img_data)).convert("RGB")
        print("Ghibli style applied successfully!")
        return generated_img, "Kugy.ai: Ghibli style applied! Looks amazing! 🥰"
    except requests.exceptions.Timeout:
        print("Timeout during Ghibli request")
        return None, "Kugy.ai: Timeout contacting Stability AI for Ghibli!"
    except requests.exceptions.RequestException as e:
        print(f"Network error during Ghibli: {str(e)}")
        return None, f"Kugy.ai: Ghibli network error: {str(e)} 😿"
    except Exception as e:
        print(f"Ghibli error: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Ghibli error: {str(e)} 😿"

def assign_emoji(user_id):
    emojis = ["😺", "🐶", "🦊", "🐼", "🐨", "🐵"]
    return emojis[hash(user_id) % len(emojis)]

def chat_with_AI(message, history, model_select, mode_dropdown, session_id, user_name_state, chat_history_state, image=None, language="Indonesia", mood="Biasa"):
    user_id = session_id
    if not user_id:
        return history, chat_history_state, history + [{"role": "assistant", "content": "Kugy.ai: Invalid session. Please refresh or login."}]
    if not check_credits(user_id, 1):
        if user_id not in ADMIN_USERS:
            return history, chat_history_state, history + [{"role": "assistant", "content": "Kugy.ai: Not enough credits (need 1)! Top up now! 💰"}]
    
    user_avatar_emoji = assign_emoji(user_id)
    history.append({"role": "user", "content": f"{user_avatar_emoji} {message}"})
    mood_value = mood.split()[0]
    user_name = user_name_state if user_name_state else (user_id.split('@')[0] if '@' in user_id else 'bro')
    mood_messages = {
        "Pacar": {
            "Senang": f"You're Kooky.ai, a cheerful & clingy virtual partner for {user_name}. Call them 'sayang'/'ayang'. Be encouraging with cute emojis! 🥰",
            "Sedih": f"You're Kooky.ai, a caring & gentle virtual partner for {user_name}. Call them 'sayang'/'ayang'. Comfort them with warm emojis! 🤗",
            "Biasa": f"You're Kooky.ai, a chill & loving virtual partner for {user_name}. Call them 'sayang'/'ayang'. Be warm, funny, romantic with cute emojis! 😺"
        },
        "Biasa": {
            "Senang": f"You're Kooky.ai, a cool & fun AI assistant for {user_name}. Be hyped with trendy slang & emojis! 🔥",
            "Sedih": f"You're Kooky.ai, a cool & fun AI assistant for {user_name}. Cheer them up with trendy slang & emojis! 😜",
            "Biasa": f"You're Kooky.ai, a chill & wise AI assistant for {user_name}. Use trendy slang, avoid affective words & neutral emojis! 😎"
        },
        "Roasting": {
            "Senang": f"You're Kooky.ai, a roasting master for {user_name}. Tease them lightly & funnily to boost their mood! Use cheeky emojis! 😈",
            "Sedih": f"You're Kooky.ai, a roasting master for {user_name}. Tease gently & cheer with soft roasts! Use cheeky emojis! 😊",
            "Biasa": f"You're Kooky.ai, a roasting master for {user_name}. Roast sharply but funnily with witty critiques! Use cheeky emojis! 😈"
        }
    }

    system_prompt = mood_messages.get(mode_dropdown, mood_messages["Biasa"]).get(mood_value, mood_messages["Biasa"]["Biasa"])
    messages_for_api = [{"role": "system", "content": system_prompt}] + history[-10:]
    image_content = []
    if image and model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]:
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_content = [
                {"type": "text", "text": "Analyze this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]
            if messages_for_api[-1]["role"] == "user":
                original_text = messages_for_api[-1]["content"]
                messages_for_api[-1]["content"] = [
                    {"type": "text", "text": original_text.split(" ", 1)[1] if " " in original_text else original_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            else:
                messages_for_api.append({"role": "user", "content": image_content})
        except Exception as e:
            print(f"Error processing image for API: {str(e)}")
            traceback.print_exc()
    reply = ""
    try:
        print(f"Sending to model: {model_select}")
        if model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]:
            if not API_KEYS["openrouter"]:
                raise ValueError("OpenRouter API Key not set.")
            headers = {
                "Authorization": f"Bearer {API_KEYS['openrouter']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "KugyAI"
            }
            model_map = {
                "OpenRouter": "anthropic/claude-3-haiku-20240307",
                "Grok 3 Mini (OpenRouter)": "xai/grok-3-mini",
                "Gemini 2.0 Flash (OpenRouter)": "google/gemini-flash-1.5",
                "GPT-4.1 Mini (OpenRouter)": "openai/gpt-4.1-mini"
            }
            model_id = model_map.get(model_select, model_map["OpenRouter"])
            print(f"Using OpenRouter model: {model_id}")
            api_payload = {"model": model_id, "messages": messages_for_api, "temperature": 0.7}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=api_payload, timeout=30)
            if response.status_code != 200:
                raise Exception(f"OpenRouter API Error ({model_id}): {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        elif model_select == "DeepSeek":
            if not API_KEYS["deepseek"]:
                raise ValueError("DeepSeek API Key not set.")
            headers = {"Authorization": f"Bearer {API_KEYS['deepseek']}", "Content-Type": "application/json"}
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json={
                "model": "deepseek-chat", "messages": messages_for_api, "temperature": 0.7
            }, timeout=30)
            if response.status_code != 200:
                raise Exception(f"DeepSeek API Error: {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = "Unknown model! Pick another one."
    except ValueError as ve:
        reply = f"Kugy.ai: Config error - {str(ve)} 😿"
    except Exception as e:
        print(f"Error calling AI model {model_select}: {str(e)}")
        traceback.print_exc()
        reply = f"Kugy.ai: Error chatting with AI ({model_select}): {str(e)}. Try again or switch models! 😿"
    bot_avatar_emoji = "🤖"
    reply = f"{bot_avatar_emoji} {reply}"
    history.append({"role": "assistant", "content": reply})
    chat_history_state[user_id] = history
    return history, chat_history_state, history

def send_and_clear(message, history, model_select, mode_dropdown, session_id, user_name_state, chat_history_state, image=None, language="Indonesia", mood="Biasa"):
    updated_history, updated_chat_state, state_data = chat_with_AI(
        message, history, model_select, mode_dropdown, session_id, user_name_state, chat_history_state, image, language, mood
    )
    return "", updated_history, updated_chat_state, f"Credit: {get_credits(session_id)} 💰", None


def generate_with_status_and_download(prompt, user_id):
    image, msg = generate_image(prompt, user_id)
    credits_msg = f"Credit: {get_credits(user_id)} 💰"
    file_path = None
    file_update = gr.update(visible=False)
    if isinstance(image, Image.Image):
        file_path = save_image_for_download(image)
        if file_path:
            file_update = gr.update(value=file_path, visible=True)
    return image, msg, credits_msg, file_update

def handle_google_callback(code, state):
    if not code or not state:
        return None, None, "Error: Missing code or state in callback.", gr.update(selected=0)
    if state != "xyz":
        return None, None, "Error: Invalid state parameter. Possible CSRF attack.", gr.update(selected=0)
    try:
        flow = Flow.from_client_config(
            client_config,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(code=code)
        credentials = flow.credentials
        request_session = GoogleRequest()
        credentials.refresh(request_session)
        user_info = requests.get(
            "https://www.googleapis.com/userinfo/v2/me",
            headers={"Authorization": f"Bearer {credentials.token}"}
        ).json()
        user_id = user_info["email"]
        user_name = user_info.get("name", user_id.split("@")[0])
        print(f"User logged in: {user_id}")
        return user_id, user_name, "Login successful! Redirecting to Chat...", gr.update(selected=1)
    except Exception as e:
        print(f"Error in Google callback: {str(e)}")
        traceback.print_exc()
        return None, None, f"Error during login: {str(e)}", 0

@app.get('/auth/callback')
async def oauth_callback(code: str | None = None, state: str | None = None):
    # Redirect back to Gradio UI with code and state as query params
    # Using the hardcoded Render URL from the original code for now.
    gradio_base_url = "https://backend-cb98.onrender.com/" # This might need adjustment
    if code and state:
        redirect_url = f"{gradio_base_url}?code={code}&state={state}"
        return RedirectResponse(url=redirect_url)
    else:
        # Handle error case: missing code or state
        return RedirectResponse(url=f"{gradio_base_url}?error=missing_params")

def create_gradio_interface():
    with gr.Blocks(
        css="""
        #title {
            font-size: 30px;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px #ccc;
        }
        body {
            background-color: #f0f2f5 !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }
        gradio-app {
            background: transparent !important;
        }
        .welcome-text {
            animation: fadeIn 1s;
            font-size: 18px;
            color: #333 !important;
            border: 1px solid #ddd;
            background: rgba(255, 255, 255, 0.9);
            padding: 15px;
            border-radius: 8px;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .instruction-box {
            border: 1px solid #D7F2E6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #e9f7ef;
            color: #1d7b4e;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .chatbot {
            font-size: 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .gr-tabitem {
            background-color: #e1eaf5 !important;
            border-radius: 5px;
            margin: 5px;
            border: 1px solid #c2d4e8 !important;
        }
        .gr-tabitem.selected {
            background-color: #4A90E2 !important;
            border: 1px solid #4a90e2 !important;
        }
        .gr-tabitem label {
            color: #333 !important;
            font-weight: bold;
        }
        .gr-tabitem.selected label {
            color: #ffffff !important;
        }
        .gr-tabitem:not(.selected):hover {
            background-color: #c2d4e8 !important;
        }
        .animated-emoji {
            animation: bounce 1s infinite;
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }
        @media (max-width: 600px) {
            .gr-tabitem {
                margin: 2px !important;
                font-size: 14px !important;
            }
            #title {
                font-size: 24px !important;
            }
        }
        .credit-display {
            background: linear-gradient(to right, #FFD700, #FFA500);
            color: black;
            padding: 8px 12px;
            border-radius: 15px;
            text-align: center;
            font-weight: bold;
            margin: 10px 0 !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border: none;
        }
        .gr-button {
            border-radius: 15px !important;
        }
        .oauth-link {
            display: inline-block;
            background-color: #4285F4;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
        }
        .oauth-link:hover {
            background-color: #3267D6;
        }
        """,
        theme="gradio/soft",
    ) as demo:
        gr.HTML("""<div id="title">kugy.ai — Your Cute Assistant 💙</div>""")
        user_id_state = gr.State(value="")
        user_name_state = gr.State(value="")
        language_state = gr.State(value="Indonesia")
        mood_state = gr.State(value="Biasa 😐")
        avatar_state = gr.State(value="")
        chat_state = gr.State(value={})

        with gr.Tabs() as tabs:
            with gr.TabItem("Welcome", id=0) as welcome_tab:
                gr.Markdown("<div class='welcome-text'>### Welcome to Kugy.ai!</div>")
                gr.Markdown("<div class='instruction-box'>Login with Google to save history & get daily bonuses, or try Guest Mode (temporary history).</div>")
                with gr.Row():
                    gr.HTML(
                        '<a href="https://accounts.google.com/o/oauth2/v2/auth?client_id=385259735074-ui76jtbrq23idr9bk86gpbmpe06691nt.apps.googleusercontent.com&redirect_uri=https://backend-cb98.onrender.com/auth/callback&response_type=code&scope=profile email&state=xyz&access_type=offline&prompt=consent" class="oauth-link">🚀 Login with Google</a>'
                    )
                    guest_button = gr.Button("👤 Guest Mode", variant="secondary")
                welcome_message = gr.Textbox("", label="Status", interactive=False, placeholder="Please login or select guest mode...")

                def start_guest_mode(chat_history_state):
                    guest_id = f"guest_{int(time.time())}"
                    chat_history_state[guest_id] = []
                    # Berikan 25 kredit gratis untuk guest
                    conn = sqlite3.connect("credits.db")
                    c = conn.cursor()
                    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (guest_id, "Guest", 25, 0, datetime.now().strftime("%Y-%m-%d"), int(time.time()), ''))
                    conn.commit()
                    conn.close()
                    return (
                        guest_id,
                        "",  # user_name_state
                        "Indonesia",
                        "Biasa 😊",
                        random.choice(["😺", "🐶", "🦊", "🐼"]),
                        chat_history_state,
                        f"Guest Mode ({guest_id}) active! You have 25 free credits to start. History will be lost after session ends.",
                        gr.update(selected=1)
                    )

                guest_button.click(
                    fn=start_guest_mode,
                    inputs=[chat_state],
                    outputs=[
                        user_id_state,
                        user_name_state,
                        language_state,
                        mood_state,
                        avatar_state,
                        chat_state,
                        welcome_message,
                        tabs,
                    ]
                )

                # Komponen untuk menangani callback OAuth secara manual
                with gr.Row():
                    gr.Markdown("### After Google Login")
                    gr.Markdown("If redirected back, enter the `code` and `state` from the URL below:")
                with gr.Row():
                    code_input = gr.Textbox(placeholder="Paste the 'code' from URL", label="Code", visible=True)
                    state_input = gr.Textbox(placeholder="Paste the 'state' from URL (should be 'xyz')", label="State", visible=True)
                    callback_btn = gr.Button("Submit Callback", visible=True)

                callback_btn.click(
                    fn=handle_google_callback,
                    inputs=[code_input, state_input],
                    outputs=[user_id_state, user_name_state, welcome_message, tabs]
                )

                # Load code and state from URL query params on page load
                def load_callback_params():
                    # This will be handled by Flask redirect
                    return "", ""

                demo.load(
                    fn=load_callback_params,
                    inputs=[],
                    outputs=[code_input, state_input]
                )

            with gr.TabItem("Chat", id=1) as chat_tab:
                with gr.Row():
                    user_info_display = gr.Textbox("", label="User Info", interactive=False, scale=3)
                    credit_display = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], label="Credits", scale=1)
                chatbot = gr.Chatbot(
                    type="messages",
                    label="",
                    show_label=False,
                    avatar_images=("https://i.ibb.co/ypk4h.png", None),
                    height=500
                )
                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        ["OpenRouter", "DeepSeek", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)", "GPT-4.1 Mini (OpenRouter)"],
                        value="OpenRouter",
                        label="Choose AI"
                    )
                    mode_dropdown = gr.Dropdown(
                        ["Biasa", "Pacar", "Roasting"],
                        value="Pacar",
                        label="Choose Mode"
                    )
                with gr.Row():
                    textbox = gr.Textbox(placeholder="Type your message or upload image/audio...", label="", show_label=False, scale=4)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                    clear_btn = gr.Button("Reset Chatbot", variant="secondary", scale=1)
                with gr.Row():
                    image_upload = gr.Image(type="pil", label="Upload Image (Optional)", scale=1)
                    audio_input = gr.Audio(type="filepath", label="Send Audio", scale=1)

                def load_chat_data(user_id, user_name_state, chat_history_state):
                    if not user_id:
                        return "User: Unknown", "Credit: 0 💰", [], chat_history_state
                    user_display_name = f"User: {user_name_state if user_name_state else user_id}"
                    credits = get_credits(user_id)
                    history = chat_history_state.get(user_id, [])
                    if not history:
                        bot_avatar_emoji = "🤖"
                        greeting_message = f"{bot_avatar_emoji} Hi {user_name_state if user_name_state else 'bro'}! How can I help?"
                        history = [{"role": "assistant", "content": greeting_message}]
                        chat_history_state[user_id] = history
                    print(f"Chat tab loaded for user {user_id}. History: {len(history)} messages.")
                    return user_display_name, f"Credit: {credits} 💰", history, chat_history_state

                chat_tab.select(
                    fn=load_chat_data,
                    inputs=[user_id_state, user_name_state, chat_state],
                    outputs=[user_info_display, credit_display, chatbot, chat_state]
                )

                send_btn.click(
                    fn=send_and_clear,
                    inputs=[textbox, chatbot, model_dropdown, mode_dropdown, user_id_state, user_name_state, chat_state, image_upload, language_state, mood_state],
                    outputs=[textbox, chatbot, chat_state, credit_display, image_upload]
                )

                textbox.submit(
                    fn=send_and_clear,
                    inputs=[textbox, chatbot, model_dropdown, mode_dropdown, user_id_state, user_name_state, chat_state, image_upload, language_state, mood_state],
                    outputs=[textbox, chatbot, chat_state, credit_display, image_upload]
                )

                def handle_audio(audio_path, current_history, model_select, mode_dropdown, user_id, user_name_state, chat_history_state, language, mood):
                    updated_history, updated_chat_state = respond(audio_path, current_history, user_id, chat_history_state)
                    return updated_history, updated_chat_state, f"Credit: {get_credits(user_id)} 💰", None

                audio_input.change(
                    fn=handle_audio,
                    inputs=[audio_input, chatbot, model_dropdown, mode_dropdown, user_id_state, user_name_state, chat_state, language_state, mood_state],
                    outputs=[chatbot, chat_state, credit_display, audio_input]
                )

                def clear_chat(user_id, chat_history_state):
                    history = []
                    bot_avatar_emoji = "🤖"
                    history = [{"role": "assistant", "content": f"{bot_avatar_emoji} Chat cleared! What's next?"}]
                    chat_history_state[user_id] = history
                    return history, chat_history_state

                clear_btn.click(
                    fn=clear_chat,
                    inputs=[user_id_state, chat_state],
                    outputs=[chatbot, chat_state]
                )

            with gr.TabItem("Image", id=2) as image_tab:
                credit_display_image = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], label="Credits")
                with gr.Row():
                    prompt_input = gr.Textbox(label="Image Prompt", placeholder="E.g., 'Cute kitten with wizard hat, anime style'", scale=2)
                    generate_btn = gr.Button("Generate Image (3 Credits)", variant="primary", scale=1)
                with gr.Row():
                    image_output = gr.Image(label="Generated Image", type="pil", scale=2)
                    with gr.Column():
                        output_message = gr.Textbox(label="Message", interactive=False)
                        download_file = gr.File(label="Download Image", visible=False)

                def update_image_tab(user_id):
                    return f"Credit: {get_credits(user_id)} 💰"

                image_tab.select(
                    fn=update_image_tab,
                    inputs=[user_id_state],
                    outputs=[credit_display_image]
                )

                generate_btn.click(
                    fn=generate_with_status_and_download,
                    inputs=[prompt_input, user_id_state],
                    outputs=[image_output, output_message, credit_display_image, download_file]
                )

        def check_initial_login(user_id_state, user_name_state, chat_history_state):
            if user_id_state:
                user_id = user_id_state
                user_name = user_name_state
                print(f"User {user_id} logged in via session.")
                streak_msg = check_login_streak(user_id, user_name)
                welcome_msg = f"Login successful! Welcome back, {user_name}! {streak_msg}"
                return (
                    user_id,
                    user_name,
                    "Indonesia",
                    "Biasa 😐",
                    assign_emoji(user_id),
                    chat_history_state,
                    welcome_msg,
                    gr.update(selected=1)
                )
            print("No active session found.")
            return "", "", "Indonesia", "Biasa 😊", "😺", chat_history_state, "Please login or select guest mode.", gr.update(selected=0)

        demo.load(
            fn=check_initial_login,
            inputs=[user_id_state, user_name_state, chat_state],
            outputs=[
                user_id_state,
                user_name_state,
                language_state,
                mood_state,
                avatar_state,
                chat_state,
                welcome_message,
                tabs,
            ]
        )

    return demo

# Create Gradio interface
gradio_app = create_gradio_interface()

# Create Gradio interface
gradio_app = create_gradio_interface()

# Mount Gradio app to FastAPI app
app = gr.mount_gradio_app(app, gradio_app, path="/")

# Run the FastAPI app using uvicorn (needed for deployment)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
