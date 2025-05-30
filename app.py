import zipfile
import os
from flask import Flask, redirect, request, session, url_for
import gradio as gr
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from faster_whisper import WhisperModel
import json
import time
import random
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
from time import sleep
from functools import wraps
import tempfile
import traceback
import sqlite3
import sys
import io
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

port = int(os.getenv("PORT", 10000))

# Setup OAuth
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"https://<your-render-domain>/auth/callback")  # Update in Render
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]

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

@app.route("/auth/google")
def google_login():
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    session["oauth_state"] = state
    return redirect(authorization_url)

@app.route("/auth/callback")
def google_callback():
    state = session.get("oauth_state")
    if not state or state != request.args.get("state"):
        return "Invalid state parameter.", 400

    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    try:
        flow.fetch_token(authorization_response=request.url)
    except Exception as e:
        print(f"Error fetching token: {e}")
        return f"Authentication failed: {e}", 400

    credentials = flow.credentials
    session["credentials"] = credentials_to_dict(credentials)

    try:
        user_info_service = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        )
        user_info = user_info_service.json()
        session["user_email"] = user_info["email"]
        session["user_name"] = user_info["name"]
        print(f"User logged in: {user_info['email']}")
        return redirect("/")
    except Exception as e:
        print(f"Error getting user info: {e}")
        return f"Could not fetch user info: {e}", 500

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

zip_path = "anime_faces.zip"
extract_to = "./anime_faces"

current_dir = os.path.abspath(os.getcwd())
print(f"Current working directory: {current_dir}")
print(f"ZIP path: {os.path.abspath(zip_path)}")
print(f"Extract to path: {os.path.abspath(extract_to)}")

if not os.path.exists(extract_to):
    os.makedirs(extract_to, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Extracted to: {os.path.abspath(extract_to)}")

            image_files_found = []
            print(f"Searching for images in {extract_to}...")
            for root, dirs, files in os.walk(extract_to):
                print(f"  Checking: {root}")
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files_found.append(os.path.join(root, file))

            if not image_files_found:
                print("Warning: No image files (PNG/JPG/JPEG) found!")
                all_items = []
                for root, dirs, files in os.walk(extract_to):
                    for item in dirs + files:
                        all_items.append(os.path.join(root, item))
                print(f"Contents of {extract_to}: {all_items}")
            else:
                print(f"Found {len(image_files_found)} images")

    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
    except FileNotFoundError:
        print(f"Error: {zip_path} not found in {current_dir}.")
    except Exception as e:
        print(f"Extraction error: {str(e)}")
else:
    print(f"Folder {extract_to} already exists.")
    image_files_found = []
    print(f"Searching for images in {extract_to}...")
    for root, dirs, files in os.walk(extract_to):
        print(f"  Checking: {root}")
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files_found.append(os.path.join(root, file))

    if not image_files_found:
        print("Warning: No image files (PNG/JPG/JPEG) found!")
        all_items = []
        for root, dirs, files in os.walk(extract_to):
            for item in dirs + files:
                all_items.append(os.path.join(root, item))
        print(f"Contents of {extract_to}: {all_items}")
    else:
        print(f"Found {len(image_files_found)} images")

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
    if not user_id or user_id.startswith("guest_"):
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
    if not user_id or user_id.startswith("guest_"):
        return "0 (Guest)"
    if user_id in ADMIN_USERS:
        return "∞ (Admin)"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def get_credits_numeric(user_id):
    if not user_id or user_id.startswith("guest_"):
        return 0
    if user_id in ADMIN_USERS:
        return float('inf')

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def top_up_credits(user_id, amount):
    if not user_id or user_id.startswith("guest_"):
        return "Kugy.ai: Guests can't top up. Please register/login."
    if user_id in ADMIN_USERS:
        return "Kugy.ai: Admin has unlimited credits! 😎"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (user_id, session.get('user_name', user_id), 0, 0, datetime.now().strftime("%Y-%m-%d"), 0, ''))
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

try:
    whisper_model = WhisperModel("tiny")
except Exception as e:
    print(f"Warning: Failed to load Whisper: {str(e)}")
    whisper_model = None

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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def blend_images_with_stability(img1, img2, prompt, user_id):
    if not check_credits(user_id, 5):
        return None, "Kugy.ai: Not enough credits (need 5)! Top up now~ 💰"
    if not API_KEYS["stability"]:
        return None, "Kugy.ai: Stability AI token missing! Add it to Secrets!"

    try:
        if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
            return None, "Kugy.ai: Invalid input images! 😿"
        img1 = img1.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        img2 = img2.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        buffer1 = BytesIO()
        img1.save(buffer1, format="PNG")
        buffer1.seek(0)
        headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Accept": "application/json"}
        files = {'init_image': ('image1.png', buffer1, 'image/png')}
        data = {
            "text_prompts[0][text]": prompt or "Blend this image with elements of another style or character, anime style",
            "text_prompts[0][weight]": "1.0",
            "cfg_scale": "7",
            "steps": "50",
            "image_strength": "0.5",
            "style_preset": "anime"
        }
        print("Sending blending request to Stability AI...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, files=files, data=data, timeout=90)
        print(f"Blending status: {response.status_code}")
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Blending failed: {error_detail}")
            return None, f"Kugy.ai: Stability AI Blending error (status {response.status_code}): {error_detail[:200]}"
        resp_data = response.json()
        if "artifacts" not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in blending response")
            return None, "Kugy.ai: Failed to get blending result from Stability!"
        base64_img = resp_data["artifacts"][0]["base64"]
        final_img = Image.open(BytesIO(base64.b64decode(base64_img)))
        return final_img, "Kugy.ai: Anime images blended successfully! 🥰"
    except requests.exceptions.Timeout:
        print("Timeout during blending")
        return None, "Kugy.ai: Timeout contacting Stability AI for blending!"
    except requests.exceptions.RequestException as e:
        print(f"Network error during blending: {str(e)}")
        return None, f"Kugy.ai: Network error during blending: {str(e)} 😿"
    except Exception as e:
        print(f"Unexpected error during blending: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Blending error: {str(e)} 😿"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
                            print(f"Failed to fetch image: {str(e)}")
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
        except:
            json.JSONDecodeError
            print(f"Piapi error: {error_detail}")
            return f"Kugy.ai: Piapi error (status {response.status_code}): {error_detail[:200]}."
    except requests.exceptions.Timeout:
        print("Timeout during Piapi request")
        return "Kugy.ai: Timeout contacting Piapi."
    except Exception as e:
        print(f"Piapi task error: {str(e)}")
        traceback.print_exc()
        return f"Kugy.ai: Piapi error: {str(e)}")

def swap_couple(master_couple_img, face_pacar_img, face_kamu_img, task_type, user_id):
    if not check_credits(user_id, 4):
        return None, "Kugy.ai: Not enough credits (need 4)! Top up now! 💰"
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

def load_random_anime_face():
    try:
        dataset_path = "./anime_faces"
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"Loading random anime face from: {abs_dataset_path}")
        try:
            if not os.path.exists(abs_dataset_path):
                print(f"Folder {abs_dataset_path} not found!")
                return None, f"Kugy.ai: Folder {dataset_path} not found!"
        image_files = []
        for root, _, files in os.walk(abs_dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        image_files.append(full_path)
                    else:
                        print(f"Warning: Path {full_path} doesn't exist.")
            print(f"Found {len(image_files)} valid images")
            if not image_files:
                all_items = []
                for root, dirs, files in os.walk(abs_dataset_path):
                    for item in dirs + files:
                        all_items.append(os.path.join(root, item))
                print(f"Folder contents: {all_items}")
                return None, f"Kugy.ai: No valid images in {dataset_path}!")
        random_image_path = random.choice(image_files)
        print(f"Loading: {random_image_path}")
        img = Image.open(random_image_path).convert("RGB")
        return img, "Kugy.ai: Random anime face loaded! 😺"
    except Exception as e:
        print(f"Error loading random image: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Failed to load random image: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def generate_anime_from_dataset(user_id):
    if not check_credits(user_id, 3):
        return None, "Kugy.ai: Not enough credits (need 3)! Top up now~ 💰")
    try:
        dataset_path = "./anime_faces"
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"Checking dataset: {abs_dataset_path}")
        if not os.path.exists(abs_dataset_path):
            print(f"Dataset folder {abs_dataset_path} not found!")
            reference_style = "anime style"
        else:
            ref_img, reference_style = find_reference_image(dataset_path, "")
            print(f"Reference style: {reference_style}")
        print("Generating anime image via Stability AI...")
        hair_color = random.choice(['blue', 'red', 'black', 'blonde', 'pink', 'green', 'silver'])
        hair_style = random.choice(['short', 'long', 'medium length', 'ponytail', 'twin tails'])
        eye_color = random.choice(['blue', 'green', 'brown', 'red', 'purple', 'golden'])
        setting = random.choice(['in a vibrant city', 'in a magical forest', 'by the sea', 'in a classroom', 'under starry sky'])
        prompt = f"Anime character with {hair_style} {hair_color} hair and {eye_color} eyes, {setting}, {reference_style}, detailed background, high quality"
        img, msg = generate_image(prompt, user_id)
        return img, msg
    except Exception as e:
        print(f"Error generating from dataset: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Failed to generate from dataset: {str(e)}")

def find_reference_image(dataset_path, prompt):
    try:
        abs_dataset_path = os.path.abspath(dataset_path)
        if not os.path.exists(abs_dataset_path):
            return None, "default anime style"
        image_files = []
        for root, dirs, files in os.walk(abs_dataset_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        image_files.append(full_path)
        if not image_files:
            return None, "default anime style"
        if prompt:
            prompt_keywords = prompt.lower().split()
            for img_file in image_files:
                file_name = os.path.basename(img_file).lower()
                if any(keyword in file_name for keyword in prompt_keywords):
                    ref_img = Image.open(img_file).convert('RGB')
                    prompt_tag = file_name.split('.')[0].replace('_', ' ').lower()
                    return ref_img, f"{prompt_tag}, detailed anime style"
        ref_img_path = random.choice(image_files)
        ref_img = Image.open(ref_img_path).convert('RGB')
        prompt_tag = os.path.basename(ref_img_path).split('.')[0].replace('_', ' ').lower()
        return ref_img, f"{prompt_tag}, detailed anime style"
    except Exception as e:
        print(f"Error finding reference image: {str(e)}")
        traceback.print_exc()
        return None, f"default anime style"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
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
            except:
                json.JSONDecodeError
            print(f"Ghibli error: {error_detail}")
            return None, f"Kugy.ai: Ghibli error (status {status_code}): {error_detail[:200]}")
        resp_data = Response.json()
        if 'artifacts' not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in Ghibli response!")
            return None, "Kugy.ai: Failed to get Ghibli result from Stability!")
        generated_img_data = base64.b64decode(resp_data["artifacts"]["0"]["base64"])
        generated_img = Image.open(BytesIO(generated_img_data)).convert("RGB")
        print("Ghibli style applied successfully!")
        return generated_img, "Kugy.ai: Ghibli style applied! Looks amazing! 🥰"
    except requests.exceptions.Timeout:
        print("Timeout during Ghibli request")
        return None, "Kugy.ai: Timeout contacting Stability AI for Ghibli!)
    except requests.exceptions.RequestException as e:
        print(f"Network error during Ghibli: {str(e)}")
        return None, f"Kugy.ai: Ghibli network error: {str(e)} 😿)
    except Exception as e:
        print(f"Ghibli error: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Ghibli error: {str(e)} 😿")

def assign_emoji(user_id):
    emojis = ["😺", "🐶", "🦊", "🐼", "🐨", "🐵"]
    return emojis[hash(user_id) % len(emojis)]

def chat_with_AI(message, history, model_select, mode_dropdown, session_id, image=None, language="Indonesia", mood="Biasa"):
    user_id = session_id
    if not user_id:
        return history, history + [{"role": "assistant", "content": "Kugy.ai: Invalid session. Please refresh or login."}]
    if not check_credits(user_id, 1):
        if user_id not in ADMIN_USERS:
            return history, history + [{"role": "assistant", "content": "Kugy.ai: Not enough credits (need 1)! Top up now! 💰"}]
    if 'chat_history' not in session:
        session['chat_history'] = {}
    if user_id not in session['chat_history']:
        session['chat_history'][user_id] = []
    current_history = session['chat_history'][user_id]
    user_avatar_emoji = assign_emoji(user_id)
    current_history.append({"role": "user", "content": f"{user_avatar_emoji} {message}"})
    mood_value = mood.split()[0]
    user_name = session.get('user_name', user_id.split('@')[0] if '@' in user_id else 'bro')
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
    messages_for_api = [{"role": "system", "content": system_prompt}] + current_history[-10:]
    image_content = []
    if image and model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]:
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_content = [
                {"type": "text", "text": "Analyze this image."},
                {"type": "image_url", "image_url": {"url": f"data:image_png;base64,{img_str}"}
            ]
            if messages_for_api[-1]["role"] == "user":
                original_text = messages_for_api[-1]["content"]
                messages_for_api[-1]["content"] = [
                    {"type": "text", "text": original_text.split(" ", 1)[1] if " " in original_text else original_text},
                    {"type": "image_url", "image_url": {"url": f"data:image_png;base64,{img_str}"}
                ]
            else:
                messages_for_api.append({"role": "user", "content": image_content})
        except Exception as e:
            print(f"Error processing image for API: {str(e)}")
            traceback.print_exc()
    reply = ""
    try:
        print(f"Sending to model: {model_select}")
        if model_select == "Mistral":
            if not API_KEYS["mistral"]:
                raise ValueError("Mistral API Key not set.")
            from mistral.client import MistralClient
            client = MistralClient(api_key=API_KEYS["mistral"])
            response = client.chat(model="mistral-large-latest", messages=messages_for_api, temperature=0.7)
            reply = response.choices[0]["message"].content
        elif model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]:
            if not API_KEYS["openrouter"]:
                raise ValueError("OpenRouter API Key not set.")
            headers = {
                "Authorization": f"Bearer {API_KEYS['openrouter']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "KugyAI"
            }
            model_map = {
                "OpenRouter": ["anthropic/claude-3-haiku-20240307",
                "Grok": ["groq/llama3-8b-8192",
                "Gemini": ["google/gemini-flash-1.5"]
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
    current_history.append({"role": "assistant", "content": reply})
    session['chat_history'][user_id] = current_history
    session.modified = True
    return current_history, current_history

def send_and_clear(message, history, model_select, mode_dropdown, session_id, image=None, language="Indonesia", mood="Biasa"):
    updated_history, state_data = chat_with_ai(message, history, model_select, mode_dropdown, session_id, image, language, mood)
    return "", updated_history, state_data, f"Credit: {get_credits(session_id)} 💰", None

def respond(audio_path, history, session_id):
    if not whisper_model:
        return history + [{"role": "assistant", "content": "Kugy.ai: Audio feature disabled due to Whisper model failure."}]
    if not audio_path:
        return history + [{"role": "assistant", "content": "Kugy.ai: Empty audio! Try again!"}]
    try:
        print(f"Transcribing audio: {audio_path}")
        segments, info = whisper_model.transcribe(audio_path, language='id')
        message = " ".join(seg.text for seg in segments).strip()
        print(f"Transcription: {message}")
        if not message:
            return history + [{"role": "assistant", "content": "Kugy.ai: Audio empty or not detected!"}]
        if 'chat_history' not in session:
            session['chat_history'] = {}
        if session_id not in session['chat_history']:
            session['chat_history'][session_id] = []
        user_avatar_emoji = assign_emoji(session_id)
        session['chat_history'][session_id].append({"role": "user", "content": f"{user_avatar_emoji} {message}"})
        session.modified = True
        return session['chat_history'][session_id]
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return history + [{"role": "assistant", "content": f"Error processing audio: {str(e)}"}]

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
        """,
        theme="gradio/soft",
    ) as demo:
        gr.HTML("""<div id="title">kugy.ai — Your Cute Assistant 💙</div>""")
        user_id_state = gr.State(value="")
        language_state = gr.State(value="Indonesia")
        mood_state = gr.State(value="Biasa 😐")
        avatar_state = gr.State(value="")
        chat_state = gr.State(value=[])
        redirect_url = gr.State(value="")  # State to handle redirect
        with gr.Tabs() as tabs:
            with gr.TabItem("Welcome", id=0) as welcome_tab:
                gr.Markdown("<div class='welcome-text'>### Welcome to Kugy.ai!</div>")
                gr.Markdown("<div class='instruction-box'>Login with Google to save history & get daily bonuses, or try Guest Mode (temporary history).</div>")
                with gr.Row():
                    login_google_btn = gr.Button("🚀 Login with Google", variant="primary")
                    guest_button = gr.Button("👤 Guest Mode", variant="secondary")
                welcome_message = gr.Textbox("", label="Status", interactive=False, placeholder="Please login or select guest mode...")
                
                def start_google_login():
                    return "/auth/google", "Initiating Google Login..."
                
                login_google_btn.click(
                    fn=start_google_login,
                    inputs=None,
                    outputs=[redirect_url, welcome_message],
                )
                
                # Handle redirect when redirect_url changes
                def handle_redirect(url):
                    if url:
                        return gr.update(value=f"Redirecting to {url}..."), url
                    return gr.update(), ""
                
                redirect_url.change(
                    fn=handle_redirect,
                    inputs=[redirect_url],
                    outputs=[welcome_message, redirect_url],
                    _js="function(url) { if (url) { window.location.href = url; } }"
                )
                
                def start_guest_mode():
                    guest_id = f"guest_{int(time.time())}"
                    return (
                        guest_id,
                        "Indonesia",
                        "Biasa 😊",
                        random.choice(["😺", "🐶", "🦊", "🐼"]),
                        [],
                        f"Guest Mode ({guest_id}) active! History will be lost after session ends.",
                        gr.update(selected=1)
                    )
                
                guest_button.click(
                    fn=start_guest_mode,
                    inputs=[],
                    outputs=[
                        user_id_state,
                        language_state,
                        mood_state,
                        avatar_state,
                        chat_state,
                        welcome_message,
                        tabs,
                    ]
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
                        ["Mistral", "OpenRouter", "DeepSeek", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"],
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
                
                def load_chat_data(user_id):
                    user_display_name = f"User: {session.get('user_name', user_id)}"
                    credits = get_credits(user_id)
                    history = session.get('chat_history', {}).get(user_id, [])
                    if not history:
                        bot_avatar_emoji = "🤖"
                        greeting_message = f"{bot_avatar_emoji} Hi {session.get('user_name', 'bro')}! How can I help?"
                        history = [{"role": "assistant", "content": greeting_message}]
                        if user_id in session.get('chat_history', {}):
                            session['chat_history'][user_id] = history
                            session.modified = True
                    return user_display_name, f"Credit: {credits} 💰", history, history
                
                chat_tab.select(
                    fn=load_chat_data,
                    inputs=[user_id_state],
                    outputs=[user_info_display, credit_display, chatbot, chat_state]
                )
                
                send_btn.click(
                    fn=send_and_clear,
                    inputs=[textbox, chat_state, model_dropdown, mode_dropdown, user_id_state, image_upload, language_state, mood_state],
                    outputs=[textbox, chatbot, chat_state, credit_display, image_upload]
                )
                
                textbox.submit(
                    fn=send_and_clear,
                    inputs=[textbox, chat_state, model_dropdown, mode_dropdown, user_id_state, image_upload, language_state, mood_state],
                    outputs=[textbox, chatbot, chat_state, credit_display, image_upload]
                )
                
                def handle_audio(audio_path, current_history, model_select, mode_dropdown, user_id, language, mood):
                    updated_history = respond(audio_path, current_history, user_id)
                    return updated_history, updated_history, f"Credit: {get_credits(user_id)} 💰", None
                
                audio_input.change(
                    fn=handle_audio,
                    inputs=[audio_input, chat_state, model_dropdown, mode_dropdown, user_id_state, language_state, mood_state],
                    outputs=[chatbot, chat_state, credit_display, audio_input]
                )
                
                def clear_chat(user_id):
                    history = []
                    bot_avatar_emoji = "🤖"
                    history = [{"role": "assistant", "content": f"{bot_avatar_emoji} Chat cleared! What's next?"}]
                    if user_id in session.get('chat_history', {}):
                        session['chat_history'][user_id] = history
                        session.modified = True
                    return history, history
                
                clear_btn.click(
                    fn=clear_chat,
                    inputs=[user_id_state],
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
            with gr.TabItem("Swap Couple", id=3) as image_v2_tab:
                credit_display_v2 = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], label="Credits")
                gr.Markdown("Upload master couple image, partner's face, and your face.")
                with gr.Row():
                    master_couple = gr.Image(label="Master Couple", type="pil", scale=1)
                    face_pacar = gr.Image(label="Partner's Face", type="pil", scale=1)
                    face_kamu = gr.Image(label="Your Face", type="pil", scale=1)
                task_type = gr.Dropdown(["face-swap"], label="Task Type", value="face-swap")
                gen_btn = gr.Button("Generate Swap (4 Credits)", variant="primary")
                with gr.Row():
                    output_v2 = gr.Image(label="Swap Result", type="pil", scale=2)
                    with gr.Column(scale=1):
                        swap_message = gr.Textbox(label="Message", interactive=False)
                        download_file_v2 = gr.File(label="Download Swap", visible=False)
                
                def update_v2_tab(user_id):
                    return f"Credit: {get_credits(user_id)} 💰"
                
                image_v2_tab.select(
                    fn=update_v2_tab,
                    inputs=[user_id_state],
                    outputs=[credit_display_v2]
                )
                
                def swap_with_status_and_download(master_couple_img, face_pacar_img, face_kamu_img, task_type, user_id):
                    image, msg = swap_couple(master_couple_img, face_pacar_img, face_kamu_img, task_type, user_id)
                    credits_msg = f"Credit: {get_credits(user_id)} 💰"
                    file_path = None
                    file_update = gr.update(visible=False)
                    if isinstance(image, Image.Image):
                        file_path = save_image_for_download(image)
                        if file_path:
                            file_update = gr.update(value=file_path, visible=True)
                    return image, msg, credits_msg, file_update
                
                gen_btn.click(
                    fn=swap_with_status_and_download,
                    inputs=[master_couple, face_pacar, face_kamu, task_type, user_id_state],
                    outputs=[output_v2, swap_message, credit_display_v2, download_file_v2]
                )
            with gr.TabItem("Ghibli Style", id=4) as image_v3_tab:
                credit_display_v3 = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], label="Credits")
                gr.Markdown("Upload your photo to transform it into Ghibli style!")
                with gr.Row():
                    user_image = gr.Image(label="Upload Photo", type="pil", scale=1)
                    ghibli_output = gr.Image(label="Ghibli Style Result", type="pil", scale=1)
                ghibli_btn = gr.Button("Apply Ghibli Style (5 Credits)", variant="primary")
                with gr.Column():
                    ghibli_msg = gr.Textbox(label="Message", interactive=False)
                    ghibli_download = gr.File(label="Download Ghibli Image", visible=False)
                
                def update_v3_tab(user_id):
                    return f"Credit: {get_credits(user_id)} 💰"
                
                image_v3_tab.select(
                    fn=update_v3_tab,
                    inputs=[user_id_state],
                    outputs=[credit_display_v3]
                )
                
                def apply_ghibli_with_download(uploaded_img, user_id):
                    image, msg = apply_ghibli_style(uploaded_img, user_id)
                    credits_msg = f"Credit: {get_credits(user_id)} 💰"
                    file_path = None
                    file_update = gr.update(visible=False)
                    if isinstance(image, Image.Image):
                        file_path = save_image_for_download(image)
                        if file_path:
                            file_update = gr.update(value=file_path, visible=True)
                    return image, msg, credits_msg, file_update
                
                ghibli_btn.click(
                    fn=apply_ghibli_with_download,
                    inputs=[user_image, user_id_state],
                    outputs=[ghibli_output, ghibli_msg, credit_display_v3, ghibli_download]
                )
        
        def handle_user_login(user_id, user_name):
            streak_msg = check_login_streak(user_id, user_name)
            welcome_msg = f"Login successful! Welcome back, {user_name}! {streak_msg}"
            return (
                user_id,
                "Indonesia",
                "Biasa 😐",
                assign_emoji(user_id),
                [],
                welcome_msg,
                gr.update(selected=1)
            )
        
        def check_initial_login():
            if 'user_email' in session:
                user_id = session['user_email']
                user_name = session.get('user_name', user_id)
                print(f"User {user_id} logged in via session.")
                return handle_user_login(user_id, user_name)
            print("No active session found.")
            return "", "Indonesia", "Biasa 😊", "😺", [], "Please login or select guest mode.", gr.update(selected=0)
        
        demo.load(
            fn=check_initial_login,
            inputs=[],
            outputs=[
                user_id_state,
                language_state,
                mood_state,
                avatar_state,
                chat_state,
                welcome_message,
                tabs,
            ]
        )
    return demo

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)