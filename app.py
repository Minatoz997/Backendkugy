import zipfile
import os
from flask import Flask, redirect, request, session # Added Flask, redirect, request, session
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
from datetime import datetime, timedelta # Added timedelta
from tenacity import retry, stop_after_attempt, wait_fixed
from time import sleep
from functools import wraps
import tempfile
import traceback
import sqlite3
import sys
import io
from google_auth_oauthlib.flow import Flow # Changed import for Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest # Renamed to avoid conflict

# Initialize Flask App FIRST
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24)) # Added secret key for session management

port = int(os.getenv("PORT", 10000))

# Setup OAuth
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
# Make sure this matches exactly what's in Google Cloud Console
REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", f"http://127.0.0.1:{port}/auth/callback")
SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]

# Client config dictionary for Google OAuth
client_config = {
    "web": {
        "client_id": CLIENT_ID,
        "project_id": "YOUR_PROJECT_ID", # Optional: Add your project ID if needed
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": CLIENT_SECRET,
        "redirect_uris": [REDIRECT_URI]
    }
}

@app.route("/auth/google")
def google_login():
    # Use Flow directly instead of InstalledAppFlow for web apps
    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent" # Force consent screen
    )
    session["oauth_state"] = state # Store state in session
    return redirect(authorization_url)

@app.route("/auth/callback")
def google_callback():
    state = session.get("oauth_state")
    if not state or state != request.args.get("state"):
        return "Invalid state parameter.", 400 # Security check

    flow = Flow.from_client_config(
        client_config,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )

    try:
        # Use the authorization response URL directly
        flow.fetch_token(authorization_response=request.url)
    except Exception as e:
        print(f"Error fetching token: {e}")
        return f"Authentication failed: {e}", 400

    credentials = flow.credentials
    session["credentials"] = credentials_to_dict(credentials) # Store credentials in session

    # Get user info
    try:
        user_info_service = requests.get(
            "https://www.googleapis.com/oauth2/v1/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"}
        )
        user_info = user_info_service.json()
        session["user_email"] = user_info["email"]
        session["user_name"] = user_info["name"]
        # Simpan ke database (bisa ditambahkan nanti, misalnya SQLite)
        print(f"User logged in: {user_info['email']}")
        # Redirect to the main Gradio interface or a success page
        # Assuming Gradio runs on the root path '/' relative to the Flask app
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

# Pastikan encoding UTF-8 digunakan di seluruh aplikasi
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Ekstrak ZIP dengan path relatif dan log detail
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
            print(f"Ekstraksi selesai ke: {os.path.abspath(extract_to)}")

            image_files_found = []
            print(f"Mencari gambar di dalam dan di bawah {extract_to}...")
            for root, dirs, files_in_dir in os.walk(extract_to):
                print(f"  Checking in: {root}")
                for file in files_in_dir:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files_found.append(os.path.join(root, file))

            if not image_files_found:
                print("Peringatan: Tidak ada file gambar (PNG/JPG/JPEG) yang ditemukan di folder atau subfolder!")
                all_items = []
                for root, dirs, files_in_dir in os.walk(extract_to):
                   for item in dirs + files_in_dir:
                       all_items.append(os.path.join(root, item))
                print(f"Isi dari {extract_to} (termasuk subfolder): {all_items}")
            else:
                print(f"Jumlah gambar terdeteksi: {len(image_files_found)}")

    except zipfile.BadZipFile:
        print(f"Error: File {zip_path} bukan file ZIP yang valid atau rusak.")
    except FileNotFoundError:
         print(f"Error: File {zip_path} tidak ditemukan di {current_dir}.")
    except Exception as e:
        print(f"Error ekstrak: {str(e)}")
else:
    print(f"Folder {os.path.abspath(extract_to)} sudah ada.")
    image_files_found = []
    print(f"Mencari gambar di dalam dan di bawah {extract_to}...")
    for root, dirs, files_in_dir in os.walk(extract_to):
        print(f"  Checking in: {root}")
        for file in files_in_dir:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files_found.append(os.path.join(root, file))

    if not image_files_found:
        print("Peringatan: Tidak ada file gambar (PNG/JPG/JPEG) yang ditemukan di folder atau subfolder!")
        all_items = []
        for root, dirs, files_in_dir in os.walk(extract_to):
           for item in dirs + files_in_dir:
               all_items.append(os.path.join(root, item))
        print(f"Isi dari {extract_to} (termasuk subfolder): {all_items}")
    else:
        print(f"Jumlah gambar terdeteksi: {len(image_files_found)}")

# Daftar Admin yang Bisa Bypass Credit
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]

# Inisialisasi Database untuk Monetisasi (Credit System)
def init_db():
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    # Added user_name and last_reward_date columns
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

# Fungsi Monetisasi
def check_credits(user_id, required_credits):
    if not user_id or user_id.startswith("guest_"): # Guests cannot use credit features
        return False

    if user_id in ADMIN_USERS:
        print(f"User {user_id} adalah admin, bypass pengecekan credit.")
        return True

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close() # Close connection early

    if not result:
        # User not found, cannot check credits
        return False
    if result[0] < required_credits:
        return False

    # Deduct credits only if sufficient
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (required_credits, user_id))
    conn.commit()
    conn.close()
    return True

def get_credits(user_id):
    if not user_id:
        return "0 (Guest)"
    if user_id.startswith("guest_"):
        return "0 (Guest)"
    if user_id in ADMIN_USERS:
        return "∞ (Admin)"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    # Return 0 if user not found or has no credits entry yet
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
        return "Kugy.ai: Mode tamu tidak bisa top up. Silakan daftar/login."
    if user_id in ADMIN_USERS:
        return "Kugy.ai: Kamu admin, credit kamu udah tak terbatas! 😎"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    # Use INSERT OR IGNORE and then UPDATE to handle new/existing users
    c.execute("INSERT OR IGNORE INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (user_id, session.get('user_name', user_id), 0, 0, datetime.now().strftime("%Y-%m-%d"), 0, ''))
    c.execute("UPDATE users SET credits = credits + ? WHERE user_id = ?", (amount, user_id))
    conn.commit()
    conn.close()
    return f"Kugy.ai: Credit {amount} ditambah! Total: {get_credits(user_id)} 💰"

def check_login_streak(user_id, user_name):
    if not user_id or user_id.startswith("guest_"):
        return "Login/daftar untuk dapat bonus harian!"

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT login_streak, last_login, last_reward_date FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()

    bonus_message = ""
    if not result:
        # First login for this user
        initial_credits = 10
        c.execute("INSERT INTO users (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (user_id, user_name, initial_credits, 1, today, 0, today))
        conn.commit()
        bonus_message = f"Selamat datang, {user_name}! Dapet {initial_credits} credit gratis nih~ 😸"
        streak = 1
    else:
        streak, last_login, last_reward_date = result
        last_login_date = datetime.strptime(last_login, "%Y-%m-%d")
        today_date = datetime.strptime(today, "%Y-%m-%d")

        # Check if reward already given today
        if last_reward_date == today:
            bonus_message = f"Streak login: {streak} hari. Bonus harian sudah diambil hari ini."
        else:
            # Calculate new streak
            if (today_date - last_login_date).days == 1:
                streak += 1
            elif (today_date - last_login_date).days > 1:
                streak = 1 # Reset streak
            else: # Login on the same day, no streak increase, but check reward
                pass

            # Grant daily bonus (e.g., 1 credit) + streak bonus
            daily_bonus = 1
            streak_bonus = 0
            if streak % 5 == 0:
                streak_bonus = 2
            total_bonus = daily_bonus + streak_bonus

            c.execute("UPDATE users SET credits = credits + ?, login_streak = ?, last_login = ?, last_reward_date = ? WHERE user_id = ?",
                      (total_bonus, streak, today, today, user_id))
            conn.commit()

            bonus_message = f"Login harian! Dapet {daily_bonus} credit. "
            if streak_bonus > 0:
                bonus_message += f"Streak {streak} hari! Bonus {streak_bonus} credit! 🎉 Total: {get_credits(user_id)} 💰"
            else:
                bonus_message += f"Streak: {streak} hari. Total: {get_credits(user_id)} 💰"

    conn.close()
    return bonus_message

# Konstanta API URLs
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
STABILITY_IMAGE_TO_IMAGE_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"
PIAPI_API_URL = "https://api.piapi.ai/api/v1/task"

# Ambil API Keys dari Environment
API_KEYS = {
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "stability": os.getenv("STABILITY_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "piapi": os.getenv("PIAPI_API_KEY")
}

missing_keys = [k for k, v in API_KEYS.items() if not v and k != "stability"] # Stability is optional for some features
if missing_keys:
    print(f"Warning: Secrets {', '.join(missing_keys)} kosong! Beberapa fitur mungkin tidak berfungsi.")

try:
    whisper_model = WhisperModel("tiny")
except Exception as e:
    print(f"Warning: Gagal inisiasi Whisper: {str(e)}. Fitur audio tidak akan berfungsi.")
    whisper_model = None # Set to None if failed

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
        return None, "Kugy.ai: Credit kurang (butuh 3)! Top up yuk~ 💰"
    cache_key = prompt.lower().strip()
    if cache_key in image_cache:
        return image_cache[cache_key], "Kugy.ai: Nih gambar dari cache buat ayang! 😘"
    if not API_KEYS["stability"]:
        return None, "Kugy.ai: Token Stability AI belum diset! Tambahin ke Secrets dulu ya, sayang!"
    api_url = STABILITY_API_URL
    headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Content-Type": "application/json", "Accept": "application/json"}
    payload = {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1, "steps": 30}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print(f"Stability Text-to-Image Response status: {response.status_code}")
        # print(f"Response text: {response.text[:500]}...") # Log beginning of response text
        if response.status_code == 200:
            resp_data = response.json()
            if "artifacts" in resp_data and resp_data["artifacts"]:
                base64_img = resp_data["artifacts"][0]["base64"]
                image = Image.open(BytesIO(base64.b64decode(base64_img)))
                if len(image_cache) >= MAX_CACHE_SIZE:
                    image_cache.pop(next(iter(image_cache))) # Remove oldest item
                image_cache[cache_key] = image
                return image, "Kugy.ai: Ini gambarnya buat ayang, cute banget kan? 🐱"
            else:
                print(f"No artifacts found in Stability response: {resp_data}")
                return None, "Kugy.ai: Gagal dapetin gambar dari Stability (no artifacts)... coba prompt lain ya!"
        elif response.status_code == 401:
             return None, "Kugy.ai: Error Stability AI (401 Unauthorized). Cek API Key kamu ya!"
        else:
            error_detail = response.text
            try: # Try to parse JSON error for better message
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Stability Error Detail: {error_detail}")
            return None, f"Kugy.ai: Error Stability AI (status {response.status_code}). Coba lagi nanti ya! Detail: {error_detail[:200]}"
    except requests.exceptions.Timeout:
        print("Timeout error generating image")
        return None, "Kugy.ai: Timeout saat menghubungi Stability AI. Coba lagi nanti."
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Error buat gambar: {str(e)}"

def image_to_base64(image):
    buffered = BytesIO()
    # Ensure image is RGB before saving as PNG/JPEG for broader compatibility
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="PNG") # PNG is generally safe
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string

def save_image_for_download(image):
    try:
        if not image:
            print("Error: Gambar untuk download kosong!")
            return None

        print("Menyimpan gambar untuk download...")
        # Ensure image is in a savable mode like RGB
        if image.mode not in ['RGB', 'RGBA', 'L']:
             image = image.convert('RGB')

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            print(f"Gambar disimpan di: {tmp.name}")
            return tmp.name

    except Exception as e:
        print(f"Error saat menyimpan gambar: {str(e)}")
        traceback.print_exc()
        return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def blend_images_with_stability(img1, img2, prompt, user_id):
    if not check_credits(user_id, 5):
        return None, "Kugy.ai: Credit kurang (butuh 5)! Top up yuk~ 💰"
    if not API_KEYS["stability"]:
        return None, "Kugy.ai: Token Stability AI belum diset! Tambahin ke Secrets dulu ya, sayang!"

    try:
        if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
            return None, "Kugy.ai: Gambar input nggak valid, ayang! 😿"

        # Resize images to be compatible with the model (e.g., 1024x1024)
        img1 = img1.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        img2 = img2.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)

        # --- Using Image-to-Image with the first image as init_image --- 
        # This approach tries to modify img1 based on the prompt and potentially img2's influence via prompt
        # A more direct blending might require different techniques or APIs if available.

        buffer1 = BytesIO()
        img1.save(buffer1, format="PNG")
        buffer1.seek(0)

        headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Accept": "application/json"}
        files = {'init_image': ('image1.png', buffer1, 'image/png')}
        data = {
            "text_prompts[0][text]": prompt or "Blend this image with elements of another style or character, anime style",
            "text_prompts[0][weight]": "1.0", # Adjust weight as needed
            "cfg_scale": "7",
            "steps": "50",
            "image_strength": "0.5", # Control how much the init_image influences the result
            "style_preset": "anime"
        }

        print("Sending request to Stability AI for image blending (using img1 as init)...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, files=files, data=data, timeout=90) # Increased timeout
        print(f"Blending Response status: {response.status_code}")

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Blending failed: {error_detail}")
            return None, f"Kugy.ai: Error Stability AI Blending (status {response.status_code}): {error_detail[:200]}"

        resp_data = response.json()
        if "artifacts" not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in blending response")
            return None, "Kugy.ai: Gagal dapetin hasil blending dari Stability... coba lagi ya!"

        base64_img = resp_data["artifacts"][0]["base64"]
        final_img = Image.open(BytesIO(base64.b64decode(base64_img)))
        return final_img, "Kugy.ai: Yeay, gambar anime berhasil digabung dengan gaya AI! Keren banget, ayang! 🥰"

    except requests.exceptions.Timeout:
        print("Timeout error during blending")
        return None, "Kugy.ai: Timeout ke Stability AI saat blending, coba lagi nanti ya! 😿"
    except requests.exceptions.RequestException as e:
        print(f"Network error during blending: {str(e)}")
        return None, f"Kugy.ai: Error jaringan ke Stability AI saat blending: {str(e)} 😿"
    except Exception as e:
        print(f"Unexpected error during blending: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Gagal blend gambar dengan AI: {str(e)} 😿"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def piapi_task(base_image, target_image, task_type, user_id):
    if not check_credits(user_id, 2):
        return "Kugy.ai: Credit kurang (butuh 2)! Top up yuk~ 💰"
    api_key = API_KEYS["piapi"]
    if not api_key:
        return "Kugy.ai: API Key Piapi gak ada, ayang! Set PIAPI_API_KEY dulu ya! 😿"

    try:
        # Resize images if they are too large for the API
        max_res = 1024 # Check Piapi docs for actual limits
        if base_image.width > max_res or base_image.height > max_res:
            base_image = base_image.resize((min(base_image.width, max_res), min(base_image.height, max_res)), Image.Resampling.LANCZOS)
        if target_image.width > max_res or target_image.height > max_res:
            target_image = target_image.resize((min(target_image.width, max_res), min(target_image.height, max_res)), Image.Resampling.LANCZOS)

        # Ensure images are RGB
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
            "model": "Qubico/image-toolkit", # Verify model name in Piapi docs
            "task_type": task_type, # e.g., "face-swap"
            "input": {
                "target_image": base_image_base64, # Image to modify
                "swap_image": target_image_base64 # Image containing the face/element to swap in
            },
            "config": {
                "service_mode": "async", # Use async for potentially long tasks
                "webhook_config": {
                    "endpoint": "", # Optional webhook
                    "secret": ""
                }
            }
        }
        print("Sending request to Piapi...")
        response = requests.post(
            PIAPI_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        print(f"Piapi initial response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            job_id = result.get("data", {}).get("task_id")
            if job_id:
                print(f"Piapi task started with ID: {job_id}")
                poll_url = f"https://api.piapi.ai/api/v1/task/{job_id}"
                for attempt in range(15): # Poll for longer (e.g., up to 75 seconds)
                    print(f"Polling Piapi task status (attempt {attempt+1})...")
                    time.sleep(5) # Wait 5 seconds between polls
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
                                    print(f"Fetching image from URL: {image_url}")
                                    image_response = requests.get(image_url, timeout=30)
                                    if image_response.status_code == 200:
                                        return Image.open(BytesIO(image_response.content))
                                    else:
                                        return f"Kugy.ai: Gagal ambil gambar dari URL Piapi (status {image_response.status_code})! 😿"
                                elif image_base64:
                                    print("Decoding image from Base64...")
                                    try:
                                        image_data = base64.b64decode(image_base64)
                                        return Image.open(BytesIO(image_data))
                                    except Exception as e:
                                        return f"Kugy.ai: Gagal decode image_base64 Piapi: {str(e)} 😿"
                                else:
                                    return f"Kugy.ai: Output Piapi gak punya URL atau Base64, ayang! Output: {output} 😿"
                            elif isinstance(output, str): # Handle if output is just a URL string
                                image_url = output
                                print(f"Fetching image from URL string: {image_url}")
                                image_response = requests.get(image_url, timeout=30)
                                if image_response.status_code == 200:
                                    return Image.open(BytesIO(image_response.content))
                                else:
                                    return f"Kugy.ai: Gagal ambil gambar dari URL string Piapi (status {image_response.status_code})! 😿"
                            else:
                                return f"Kugy.ai: Format output Piapi gak sesuai, ayang! Output: {output} 😿"
                        elif status == "failed":
                            error_message = poll_result.get('data', {}).get('message', 'Gak ada detail error')
                            print(f"Piapi task failed: {error_message}")
                            return f"Kugy.ai: Task Piapi gagal, ayang! Pesan dari server: {error_message}. 😿"
                        # If status is 'processing' or similar, continue polling
                    else:
                        print(f"Polling failed with status: {poll_response.status_code}")
                        # Optional: break or return error if polling fails consistently
                return "Kugy.ai: Timeout polling hasil Piapi, ayang! Task mungkin masih berjalan. Coba lagi nanti! 😿"
            else:
                print(f"No task ID received from Piapi: {result}")
                return "Kugy.ai: Gak dapet task ID dari Piapi, ayang! 😿"
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Piapi initial request error: {error_detail}")
            return f"Kugy.ai: Error Piapi (status {response.status_code}): {error_detail[:200]}. Sabar ya, ayang! 😿"

    except requests.exceptions.Timeout:
        print("Timeout error during Piapi request")
        return "Kugy.ai: Timeout saat menghubungi Piapi. Coba lagi nanti."
    except Exception as e:
        print(f"Error during Piapi task: {str(e)}")
        traceback.print_exc()
        return f"Kugy.ai: Error Piapi: {str(e)} 😿"

def swap_couple(master_couple_img, face_pacar_img, face_kamu_img, task_type, user_id):
    if not check_credits(user_id, 4): # Potentially 2 credits per swap
        return None, "Kugy.ai: Credit kurang (butuh 4)! Top up yuk~ 💰"
    if not master_couple_img or not face_pacar_img or not face_kamu_img:
        return None, "Kugy.ai: Harap unggah semua gambar terlebih dahulu, ayang! 😺"

    print("Starting couple swap step 1 (Pacar face)...")
    # Step 1: Swap pacar's face onto the master image
    step1_result = piapi_task(master_couple_img, face_pacar_img, task_type, user_id)
    if isinstance(step1_result, str): # Check if piapi_task returned an error message string
        return None, f"Kugy.ai: Gagal di langkah 1 (swap pacar): {step1_result}"
    elif not isinstance(step1_result, Image.Image):
        return None, "Kugy.ai: Hasil langkah 1 bukan gambar yang valid."

    print("Starting couple swap step 2 (Kamu face)...")
    # Step 2: Swap your face onto the result of step 1
    # Note: This assumes the API can detect the *other* face in the master image.
    # If it always swaps the same face, this logic needs adjustment.
    # It might be better to swap faces individually onto separate copies of the master
    # and then combine them, or use an API that supports multiple face swaps.
    final_img_result = piapi_task(step1_result, face_kamu_img, task_type, user_id)
    if isinstance(final_img_result, str):
        return None, f"Kugy.ai: Gagal di langkah 2 (swap kamu): {final_img_result}"
    elif not isinstance(final_img_result, Image.Image):
         return None, "Kugy.ai: Hasil langkah 2 bukan gambar yang valid."

    return final_img_result, "Kugy.ai: Yeay, foto couple-nya udah jadi! So sweet banget, ayang! 🥰"

def load_random_anime_face():
    try:
        dataset_path = "./anime_faces"
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"load_random_anime_face: Mencari gambar di path absolut: {abs_dataset_path}")

        if not os.path.exists(abs_dataset_path):
            print(f"Folder {abs_dataset_path} tidak ditemukan!")
            return None, f"Kugy.ai: Folder {dataset_path} gak ditemukan! Pastiin foldernya ada ya, bro! 😿"

        image_files = []
        for root, dirs, files_in_dir in os.walk(abs_dataset_path):
             for file in files_in_dir:
                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                     # Ensure the path is valid before adding
                     full_path = os.path.join(root, file)
                     if os.path.exists(full_path):
                        image_files.append(full_path)
                     else:
                        print(f"Warning: Path {full_path} found during walk but doesn't exist.")

        print(f"load_random_anime_face: File gambar valid yang terdeteksi: {len(image_files)}")
        if not image_files:
            all_items = []
            for root, dirs, files_in_dir in os.walk(abs_dataset_path):
                for item in dirs + files_in_dir:
                     all_items.append(os.path.join(root, item))
            print(f"load_random_anime_face: Semua item di folder (termasuk subfolder): {all_items}")
            return None, f"Kugy.ai: Gak ada gambar valid di folder {dataset_path} atau subfoldernya! Isi dulu ya, bro! 😿"

        random_image_path = random.choice(image_files)
        print(f"load_random_anime_face: Memuat gambar dari path: {random_image_path}")
        img = Image.open(random_image_path).convert("RGB")
        return img, "Kugy.ai: Nih wajah anime random buat ayang! 😺"
    except Exception as e:
        print(f"Error saat load gambar random: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Gagal load gambar random: {str(e)} 😿"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def generate_anime_from_dataset(user_id):
    # This function seems to just generate a new image using Stability, not directly from the dataset?
    # Renaming or clarifying its purpose might be good.
    # Assuming it's meant to generate a *new* anime image inspired by the dataset style.
    if not check_credits(user_id, 3):
        return None, "Kugy.ai: Credit kurang (butuh 3)! Top up yuk~ 💰"
    try:
        dataset_path = "./anime_faces"
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"generate_anime_from_dataset: Mengecek path dataset: {abs_dataset_path}")

        if not os.path.exists(abs_dataset_path):
            print(f"Folder dataset {abs_dataset_path} tidak ditemukan!")
            # Decide if you want to proceed without dataset or return error
            # Proceeding without dataset reference for now
            reference_style = "anime style"
        else:
            # Optional: Find a reference image from dataset to guide style
            ref_img, reference_style = find_reference_image(dataset_path, "") # Find random reference
            print(f"Using reference style: {reference_style}")

        print("generate_anime_from_dataset: Generating new anime image via Stability AI...")
        # Create a more diverse prompt
        hair_color = random.choice(['blue', 'red', 'black', 'blonde', 'pink', 'green', 'silver'])
        hair_style = random.choice(['short', 'long', 'medium length', 'ponytail', 'twin tails'])
        eye_color = random.choice(['blue', 'green', 'brown', 'red', 'purple', 'golden'])
        setting = random.choice(['in a vibrant city', 'in a magical forest', 'by the sea', 'in a classroom', 'under starry sky'])
        prompt = f"Anime character with {hair_style} {hair_color} hair and {eye_color} eyes, {setting}, {reference_style}, detailed background, high quality"

        img, msg = generate_image(prompt, user_id)

        # No need to save temporarily here unless specifically required
        # The image object `img` is returned directly
        return img, msg

    except Exception as e:
        print(f"Error saat generate dari dataset inspiration: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Gagal generate dari dataset inspiration: {str(e)} 😿"

def find_reference_image(dataset_path, prompt):
    # (This function remains largely the same as provided, ensure it works as expected)
    try:
        abs_dataset_path = os.path.abspath(dataset_path)
        # ... (rest of the function code as provided) ...
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

        # Simple keyword matching (can be improved)
        if prompt:
            prompt_keywords = prompt.lower().split()
            for img_file in image_files:
                file_name = os.path.basename(img_file).lower()
                if any(keyword in file_name for keyword in prompt_keywords):
                    ref_img = Image.open(img_file).convert("RGB")
                    prompt_tag = file_name.split(".")[0].replace("_", " ").lower()
                    return ref_img, f"{prompt_tag}, detailed anime style"

        # Fallback to random image if no match
        ref_img_path = random.choice(image_files)
        ref_img = Image.open(ref_img_path).convert("RGB")
        prompt_tag = os.path.basename(ref_img_path).split(".")[0].replace("_", " ").lower()
        return ref_img, f"{prompt_tag}, detailed anime style"

    except Exception as e:
        print(f"Error finding reference image: {str(e)}")
        traceback.print_exc()
        return None, "default anime style"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def apply_ghibli_style(uploaded_img, user_id):
    if not check_credits(user_id, 5):
        print(f"User {user_id} lacks credits for Ghibli (needs 5).")
        return None, "Kugy.ai: Credit kurang (butuh 5)! Top up yuk~ 💰"

    try:
        if not isinstance(uploaded_img, Image.Image):
            print(f"Validation failed for Ghibli: uploaded_img type: {type(uploaded_img)}")
            return None, "Kugy.ai: Gambar input Ghibli nggak valid, ayang! 😿"

        # Ensure RGB and resize
        uploaded_img = uploaded_img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        print(f"Resized Ghibli input image to: {uploaded_img.size}")

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

        # Using multipart/form-data for image upload with Stability image-to-image
        files = {'init_image': ('image.png', buffer, 'image/png')}
        data = {
            "text_prompts[0][text]": ghibli_prompt,
            "cfg_scale": "10", # Higher CFG scale might enhance style adherence
            "steps": "50",
            "image_strength": "0.45", # Adjust strength: lower=more change, higher=closer to original
            "style_preset": "anime" # Use anime preset
        }

        print("Sending request to Stability AI for Ghibli style transfer (multipart)...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, files=files, data=data, timeout=90) # Increased timeout
        print(f"Ghibli Response status: {response.status_code}")

        if response.status_code != 200:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = error_json.get('message', error_detail)
            except json.JSONDecodeError:
                pass
            print(f"Ghibli style transfer failed: {error_detail}")
            return None, f"Kugy.ai: Error Stability AI Ghibli (status {response.status_code}): {error_detail[:200]}"

        resp_data = response.json()
        if "artifacts" not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in Ghibli API response!")
            return None, "Kugy.ai: Gagal dapetin hasil Ghibli dari Stability AI, coba lagi ya! 😿"

        generated_img_data = base64.b64decode(resp_data["artifacts"][0]["base64"])
        generated_img = Image.open(BytesIO(generated_img_data)).convert("RGB")
        print("Ghibli style transfer successful!")
        return generated_img, "Kugy.ai: Yeay, gambar dengan style Ghibli jadi! Keren banget, ayang! 🥰"

    except requests.exceptions.Timeout:
        print("Timeout error during Ghibli Stability AI request")
        return None, "Kugy.ai: Timeout ke Stability AI Ghibli, coba lagi nanti ya! 😿"
    except requests.exceptions.RequestException as e:
        print(f"Network error during Ghibli Stability AI request: {str(e)}")
        return None, f"Kugy.ai: Error jaringan ke Stability AI Ghibli: {str(e)} 😿"
    except Exception as e:
        print(f"Error during Ghibli style transfer: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Error Ghibli: {str(e)} 😿"

# --- Chat Function --- 
def chat_with_ai(message, history, model_select, mode_dropdown, session_id, image=None, language="Indonesia", mood="Biasa"):
    # Use session_id directly from Gradio state, which should be the user's email or guest ID
    user_id = session_id

    if not user_id:
        return history, history + [{"role": "assistant", "content": "Kugy.ai: Sesi tidak valid. Coba refresh halaman atau login ulang."}]

    # Check credits (cost 1 per message)
    if not check_credits(user_id, 1):
        # Allow admins to chat even with 0 credits shown (check_credits handles admin bypass)
        if user_id not in ADMIN_USERS:
             return history, history + [{"role": "assistant", "content": "Kugy.ai: Credit kurang (butuh 1)! Top up yuk~ 💰"}]

    # Load or initialize session history from Flask session (more robust than global dict)
    if 'chat_history' not in session:
        session['chat_history'] = {}
    if user_id not in session['chat_history']:
        session['chat_history'][user_id] = []

    current_history = session['chat_history'][user_id]

    # Append user message
    # Determine avatar based on user_id (customize as needed)
    user_avatar_emoji = assign_emoji(user_id)
    current_history.append({"role": "user", "content": f"{user_avatar_emoji} {message}"})

    # Determine mood and system prompt
    mood_value = mood.split()[0]
    user_name = session.get('user_name', user_id.split('@')[0] if '@' in user_id else 'bro') # Get name from Flask session if available

    # System prompts based on mode and mood
    mood_messages = {
        "Pacar": {
            "Senang": f"Kamu Kugy.ai, pacar virtual ceria & manja untuk {user_name}. Panggil dia 'sayang'/'ayang'. Kasih semangat & emoji lucu! 🥰",
            "Sedih": f"Kamu Kugy.ai, pacar virtual penyayang & lembut untuk {user_name}. Panggil dia 'sayang'/'ayang'. Hibur dia & emoji hangat! 🤗",
            "Biasa": f"Kamu Kugy.ai, pacar virtual santai & penyayang untuk {user_name}. Panggil dia 'sayang'/'ayang'. Bicara hangat, lucu, romantis & emoji imut! 😺"
        },
        "Biasa": {
            "Senang": f"Kamu Kugy.ai, asisten AI santai & gaul untuk {user_name}. Kasih semangat pakai bahasa kekinian & emoji! 🔥",
            "Sedih": f"Kamu Kugy.ai, asisten AI santai & gaul untuk {user_name}. Hibur dia pakai bahasa kekinian & emoji! 🥳",
            "Biasa": f"Kamu Kugy.ai, asisten AI santai & bijak untuk {user_name}. Pakai bahasa kekinian, hindari kata afektif & emoji netral! 😎"
        },
        "Roasting": {
            "Senang": f"Kamu Kugy.ai, roasting master untuk {user_name}. Sindir santai & kocak biar dia tambah semangat! Pakai emoji nakal! 😈",
            "Sedih": f"Kamu Kugy.ai, roasting master untuk {user_name}. Sindir lembut & hibur dengan ejekan halus! Pakai emoji nakal! 😏",
            "Biasa": f"Kamu Kugy.ai, roasting master untuk {user_name}. Sindir tajam tapi asik & kritik kocak! Pakai emoji nakal! 😂"
        }
    }

    system_prompt = mood_messages.get(mode_dropdown, mood_messages["Biasa"]).get(mood_value, mood_messages["Biasa"]["Biasa"])
    messages_for_api = [{"role": "system", "content": system_prompt}] + current_history[-10:] # Limit history length for API

    # Handle image input for multimodal models
    image_content = []
    if image and model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]: # Check which OpenRouter models support vision
        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_content = [
                {"type": "text", "text": "Analisis gambar ini."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]
            # Add image content to the last user message for the API
            if messages_for_api[-1]["role"] == "user":
                # Combine text and image into a list under 'content'
                original_text = messages_for_api[-1]["content"]
                messages_for_api[-1]["content"] = [
                    {"type": "text", "text": original_text.split(" ", 1)[1] if " " in original_text else original_text}, # Extract text part
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                ]
            else: # Should not happen if logic is correct, but as fallback:
                 messages_for_api.append({"role": "user", "content": image_content})

        except Exception as e:
            print(f"Error processing image for API: {e}")
            # Optionally inform the user about the image processing error

    reply = ""
    try:
        print(f"Sending to model: {model_select}")
        if model_select == "Mistral":
            if not API_KEYS["mistral"]:
                 raise ValueError("Mistral API Key not set.")
            from mistralai.client import MistralClient # Correct import
            client = MistralClient(api_key=API_KEYS["mistral"])
            response = client.chat(model="mistral-large-latest", messages=messages_for_api, temperature=0.7)
            reply = response.choices[0].message.content
        elif model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]:
            if not API_KEYS["openrouter"]:
                raise ValueError("OpenRouter API Key not set.")
            headers = {
                "Authorization": f"Bearer {API_KEYS['openrouter']}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost", # Add referrer
                "X-Title": "KugyAI" # Add title
            }
            # Select appropriate model ID based on dropdown
            model_map = {
                "OpenRouter": "anthropic/claude-3-haiku-20240307", # Default/fallback
                "Grok 3 Mini (OpenRouter)": "groq/llama3-8b-8192", # Example, check OpenRouter for exact ID
                "Gemini 2.0 Flash (OpenRouter)": "google/gemini-flash-1.5" # Example, check OpenRouter
            }
            model_id = model_map.get(model_select, model_map["OpenRouter"])
            print(f"Using OpenRouter model: {model_id}")
            api_payload = {"model": model_id, "messages": messages_for_api, "temperature": 0.7}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=api_payload, timeout=60)
            if response.status_code != 200:
                raise Exception(f"OpenRouter API Error ({model_id}): {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        elif model_select == "DeepSeek":
            if not API_KEYS["deepseek"]:
                 raise ValueError("DeepSeek API Key not set.")
            headers = {"Authorization": f"Bearer {API_KEYS['deepseek']}", "Content-Type": "application/json"}
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json={
                "model": "deepseek-chat", "messages": messages_for_api, "temperature": 0.7
            }, timeout=60)
            if response.status_code != 200:
                 raise Exception(f"DeepSeek API Error: {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = "Model tidak dikenal, bro! Pilih model lain ya."
    except ValueError as ve:
        reply = f"Kugy.ai: Error konfigurasi - {str(ve)} 😿"
    except Exception as e:
        print(f"Error calling AI model {model_select}: {str(e)}")
        traceback.print_exc()
        reply = f"Aduh, error nih pas ngobrol sama AI ({model_select}): {str(e)}. Coba lagi atau ganti model ya! 😿"

    # Format reply with bot avatar
    bot_avatar_emoji = "🤖" # Default bot avatar
    reply = f"{bot_avatar_emoji} {reply}"
    current_history.append({"role": "assistant", "content": reply})

    # Save updated history back to Flask session
    session['chat_history'][user_id] = current_history
    session.modified = True # Mark session as modified

    # Return the full history for Gradio display
    return current_history, current_history

def send_and_clear(message, history, model_select, mode_dropdown, session_id, image=None, language="Indonesia", mood="Biasa"):
    # Call chat_with_ai to process the message and get the updated history
    updated_history, state_data = chat_with_ai(message, history, model_select, mode_dropdown, session_id, image, language, mood)
    # Return empty string for the textbox and the updated history for the chatbot and state
    return "", updated_history, state_data

def respond(audio_path, history, session_id):
    if not whisper_model:
        return history + [{"role": "assistant", "content": "Kugy.ai: Fitur audio tidak aktif karena model Whisper gagal dimuat."}]
    if not audio_path:
        return history + [{"role": "assistant", "content": "Audio kosong, bro! Coba lagi ya!"}]
    try:
        print(f"Transcribing audio: {audio_path}")
        # Ensure the model uses the correct language if specified
        segments, info = whisper_model.transcribe(audio_path, language='id') # Force Indonesian for now
        message = " ".join(seg.text for seg in segments).strip()
        print(f"Transcription result: {message}")
        if not message:
            return history + [{"role": "assistant", "content": "Kugy.ai: Audio gak kedengeran atau kosong, coba lagi ya!"}]

        # Append transcribed message as user input to the history associated with session_id
        if 'chat_history' not in session:
            session['chat_history'] = {}
        if session_id not in session['chat_history']:
            session['chat_history'][session_id] = []

        user_avatar_emoji = assign_emoji(session_id)
        session['chat_history'][session_id].append({"role": "user", "content": f"{user_avatar_emoji} {message}"})
        session.modified = True

        # Return the updated history for Gradio display
        return session['chat_history'][session_id]
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        traceback.print_exc()
        return history + [{"role": "assistant", "content": f"Error proses audio: {str(e)}"}]

# --- UI Gradio --- 
def create_gradio_interface():
    # Define the Gradio interface within a function
    # This allows it to be mounted onto the Flask app
    with gr.Blocks(
        css="""
        #title {
            font-size: 30px;
            font-weight: bold;
            color: #4A90E2; /* Blue color */
            text-align: center;
            margin-bottom: 10px;
            text-shadow: 1px 1px 2px #ccc;
        }
        body {
            /* Optional: Keep background or remove for cleaner look */
            /* background-image: url('https://raw.githubusercontent.com/Minatoz997/angel_background.png/main/angel_background.png') !important; */
            background-color: #f0f2f5 !important; /* Light grey background */
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
            color: #333 !important; /* Darker text */
            border: 1px solid #ddd; /* Lighter border */
            background: rgba(255, 255, 255, 0.9); /* White semi-transparent background */
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
            background-color: #e1eaf5 !important; /* Lighter blue for tabs */
            border-radius: 5px;
            margin: 5px;
            border: 1px solid #c2d4e8 !important;
        }
        .gr-tabitem.selected {
             background-color: #4A90E2 !important; /* Selected tab blue */
             border: 1px solid #4A90E2 !important;
        }
        .gr-tabitem label {
            color: #333 !important; /* Darker label text */
            font-weight: bold;
        }
         .gr-tabitem.selected label {
             color: #FFFFFF !important; /* White text for selected tab */
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
                margin: 2px;
                font-size: 14px;
            }
            #title {
                font-size: 24px;
            }
        }
        .credit-display {
            background: linear-gradient(to right, #FFD700, #FFA500); /* Gold gradient */
            color: black;
            padding: 8px 12px;
            border-radius: 15px; /* Pill shape */
            text-align: center;
            font-weight: bold;
            margin: 10px 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            border: none;
        }
        .gr-button {
             border-radius: 15px !important; /* Rounded buttons */
        }
        """,
        theme=gr.themes.Soft() # Use a soft theme
    ) as demo:
        gr.HTML('<div id="title">kugy.ai — Asisten Imut Kamu 💙</div>')

        # --- State Variables --- 
        # Store user ID (email/guest ID) in state AFTER login/guest mode selection
        user_id_state = gr.State("")
        # Store chat history in state (though Flask session is primary)
        chat_state = gr.State([])
        # Store language and mood
        language_state = gr.State("Indonesia")
        mood_state = gr.State("Biasa 😐")
        avatar_state = gr.State("😺")

        with gr.Tabs() as tabs:
            # --- Welcome/Login Tab --- 
            with gr.TabItem("Welcome", id=0) as welcome_tab:
                gr.Markdown("<div class='welcome-text'>### Selamat Datang di Kugy.ai!</div>")
                gr.Markdown("<div class='instruction-box'>Login dengan Google untuk menyimpan history & dapat bonus harian, atau coba Mode Tamu (history sementara).</div>")
                with gr.Row():
                    login_google_btn = gr.Button("🚀 Login dengan Google", variant="primary")
                    guest_button = gr.Button("👤 Mode Tamu", variant="secondary")
                welcome_message = gr.Textbox(label="Status", interactive=False, placeholder="Silakan login atau pilih mode tamu...")
                # Hidden button to trigger redirect (controlled by JS)
                # login_redirect_btn = gr.Button("Redirecting...", visible=False)

                # Function to initiate Google Login (redirect handled by Flask)
                def start_google_login():
                    # This function in Gradio doesn't perform the redirect itself.
                    # It signals the intent. The actual redirect happens via Flask's route.
                    # We might need JS to click a hidden link or directly navigate.
                    # For simplicity now, assume user clicks a link/button handled by Flask.
                    # We return a message indicating the process has started.
                    # return "Mengarahkan ke Google Login...", "/auth/google"
                    # Let's try returning JS to redirect
                    js_redirect = "() => { window.location.href = '/auth/google'; }"
                    return "Mengarahkan ke Google Login...", js_redirect

                # login_google_btn.click(lambda: gr.update(visible=True), None, login_redirect_btn)
                # login_redirect_btn.click(None, None, None, js="() => { window.location.href = '/auth/google'; }")
                login_google_btn.click(lambda: "Mengarahkan ke Google Login...", None, welcome_message, js="() => { window.location.href = '/auth/google'; }")

                # Function for Guest Mode
                def start_guest_mode():
                    guest_id = f"guest_{int(datetime.now().timestamp())}"
                    # No credit check here, just start the session
                    # Store guest ID in state
                    # Switch tabs
                    return (
                        guest_id, # Update user_id_state
                        "Indonesia", # Default language
                        "Biasa 😐", # Default mood
                        random.choice(["😺", "🐶", "🦊", "🐼"]), # Random avatar
                        [], # Empty history for chat_state
                        f"Mode Tamu ({guest_id}) aktif! History akan hilang setelah 1 jam.", # Welcome message
                        gr.update(selected=1) # Switch to Chat Tab (index 1)
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
                        tabs # Output to control tab selection
                    ]
                )

            # --- Chat Tab --- 
            with gr.TabItem("Ngobrol", id=1) as chat_tab:
                # Display user info and credits at the top
                with gr.Row():
                    user_info_display = gr.Textbox("User: Guest", interactive=False, scale=3)
                    credit_display = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"], scale=1)

                chatbot = gr.Chatbot(type="messages", show_label=False, avatar_images=("https://i.ibb.co/yp4hMMV/kugy.png", None), height=500)

                with gr.Row():
                    model_dropdown = gr.Dropdown(
                        ["Mistral", "OpenRouter", "DeepSeek", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"],
                        value="OpenRouter", # Default to a generally good model
                        label="Pilih AI", scale=1
                    )
                    mode_dropdown = gr.Dropdown(["Biasa", "Pacar", "Roasting"], value="Pacar", label="Mode", scale=1)
                    # Language and Mood can be moved here if needed per-chat
                    # language_input = gr.Dropdown(["Indonesia", "Inggris"], value="Indonesia", label="Bahasa", scale=1)
                    # mood_input = gr.Dropdown(["Senang 😊", "Sedih 🥺", "Biasa 😐"], value="Biasa 😐", label="Mood", scale=1)

                with gr.Row():
                    textbox = gr.Textbox(placeholder="Ketik pesan atau upload gambar/audio...", show_label=False, scale=4)
                    send_btn = gr.Button("Kirim", variant="primary", scale=1)
                    clear_btn = gr.Button("Reset Chat", variant="secondary", scale=1)

                with gr.Row():
                    image_upload = gr.Image(type="pil", label="Upload Gambar (Opsional)", scale=1)
                    audio_input = gr.Audio(type="filepath", label="Kirim Suara (Opsional)", scale=1)

                # Function to update UI elements when chat tab is loaded or user changes
                def load_chat_data(user_id):
                    user_display = f"User: {session.get('user_name', user_id)}"
                    credits = get_credits(user_id)
                    history = session.get('chat_history', {}).get(user_id, [])
                    # Add initial greeting if history is empty
                    if not history:
                         bot_avatar_emoji = "🤖"
                         greeting = f"{bot_avatar_emoji} Halo {session.get('user_name', 'bro')}! Ada yang bisa dibantu?"
                         history = [{"role": "assistant", "content": greeting}]
                         if user_id in session.get('chat_history', {}):
                             session['chat_history'][user_id] = history
                             session.modified = True

                    return user_display, f"Credit: {credits} 💰", history, history

                # Trigger loading when user_id_state changes (after login/guest mode)
                # Use a dummy button click triggered by state change if direct trigger isn't reliable
                # Or, use the .select() event of the tab if possible
                chat_tab.select(
                    fn=load_chat_data,
                    inputs=[user_id_state],
                    outputs=[user_info_display, credit_display, chatbot, chat_state]
                )

                # Send button action
                send_btn.click(
                    fn=update_chat, # Use the combined update function
                    inputs=[textbox, chat_state, model_dropdown, mode_dropdown, user_id_state, image_upload, language_state, mood_state],
                    outputs=[textbox, chatbot, chat_state, credit_display, image_upload] # Clear textbox and image upload after send
                )
                textbox.submit(
                     fn=update_chat,
                    inputs=[textbox, chat_state, model_dropdown, mode_dropdown, user_id_state, image_upload, language_state, mood_state],
                    outputs=[textbox, chatbot, chat_state, credit_display, image_upload]
                )

                # Audio input action
                # This needs careful handling - transcribe first, then call chat_with_ai
                def handle_audio(audio_path, current_history, model_select, mode_dropdown, session_id, language, mood):
                    if not whisper_model:
                        return current_history + [{"role": "assistant", "content": "Kugy.ai: Fitur audio tidak aktif."}]
                    if not audio_path:
                        return current_history # No audio, do nothing

                    # 1. Transcribe
                    transcribed_text = ""
                    try:
                        print(f"Transcribing audio: {audio_path}")
                        segments, info = whisper_model.transcribe(audio_path, language='id')
                        transcribed_text = " ".join(seg.text for seg in segments).strip()
                        print(f"Transcription result: {transcribed_text}")
                        if not transcribed_text:
                            return current_history + [{"role": "assistant", "content": "Kugy.ai: Audio kosong/tidak terdengar."}]
                    except Exception as e:
                        print(f"Error transcribing audio: {e}")
                        return current_history + [{"role": "assistant", "content": f"Error transkripsi: {e}"}]

                    # 2. Call chat_with_ai with the transcribed text
                    # Pass None for the image argument
                    updated_history, state_data = chat_with_ai(transcribed_text, current_history, model_select, mode_dropdown, session_id, None, language, mood)
                    return updated_history, state_data, f"Credit: {get_credits(session_id)} 💰", None # Return updated history and clear audio input

                audio_input.change(
                    fn=handle_audio,
                    inputs=[audio_input, chat_state, model_dropdown, mode_dropdown, user_id_state, language_state, mood_state],
                    outputs=[chatbot, chat_state, credit_display, audio_input]
                )

                # Clear button action
                def clear_chat_history(user_id):
                    if user_id in session.get('chat_history', {}):
                        session['chat_history'][user_id] = []
                        session.modified = True
                    # Add initial greeting after clearing
                    bot_avatar_emoji = "🤖"
                    greeting = f"{bot_avatar_emoji} History dibersihkan! Ada lagi yang bisa dibantu?"
                    cleared_history = [{"role": "assistant", "content": greeting}]
                    if user_id in session.get('chat_history', {}):
                         session['chat_history'][user_id] = cleared_history
                         session.modified = True
                    return cleared_history, cleared_history

                clear_btn.click(
                    fn=clear_chat_history,
                    inputs=[user_id_state],
                    outputs=[chatbot, chat_state]
                )

            # --- Image Generation Tab --- 
            with gr.TabItem("Gambar AI", id=2) as image_tab:
                credit_display_img = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"])
                with gr.Row():
                    prompt_input = gr.Textbox(label="Prompt Gambar", placeholder="Contoh: 'Kucing lucu pakai topi penyihir, gaya anime'", scale=3)
                    generate_btn = gr.Button("Buat Gambar (3 Credits)", variant="primary", scale=1)
                with gr.Row():
                    image_output = gr.Image(label="Hasil Gambar", scale=2)
                    with gr.Column(scale=1):
                        message_output = gr.Textbox(label="Pesan", interactive=False)
                        download_file = gr.File(label="Download Gambar", visible=False)

                def update_image_tab(user_id):
                    return f"Credit: {get_credits(user_id)} 💰"

                image_tab.select(update_image_tab, user_id_state, credit_display_img)

                generate_btn.click(
                    fn=generate_with_status,
                    inputs=[prompt_input, user_id_state],
                    outputs=[image_output, message_output, credit_display_img, download_file]
                )

            # --- Image Swap Tab --- 
            with gr.TabItem("Swap Couple", id=3) as image_v2_tab:
                credit_display_v2 = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"])
                gr.Markdown("Upload foto master (pasangan), foto wajah pacar, dan foto wajah kamu.")
                with gr.Row():
                    master_couple = gr.Image(label="Foto Master Couple", type="pil", scale=1)
                    face_pacar = gr.Image(label="Wajah Pacarmu", type="pil", scale=1)
                    face_kamu = gr.Image(label="Wajah Kamu", type="pil", scale=1)
                task_type_input = gr.Dropdown(["face-swap"], label="Tipe Task", value="face-swap") # Assuming only face-swap for now
                gen_btn = gr.Button("Generate Swap Couple (4 Credits)", variant="primary")
                with gr.Row():
                    output_v2 = gr.Image(label="Hasil Face Swap", scale=2)
                    with gr.Column(scale=1):
                        swap_message = gr.Textbox(label="Pesan", interactive=False)
                        download_file_v2 = gr.File(label="Download Hasil Swap", visible=False)

                def update_v2_tab(user_id):
                    return f"Credit: {get_credits(user_id)} 💰"

                image_v2_tab.select(update_v2_tab, user_id_state, credit_display_v2)

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
                    inputs=[master_couple, face_pacar, face_kamu, task_type_input, user_id_state],
                    outputs=[output_v2, swap_message, credit_display_v2, download_file_v2]
                )

            # --- Ghibli Style Tab --- 
            with gr.TabItem("Ghibli Style", id=4) as image_v3_tab:
                credit_display_v3 = gr.Textbox("Credit: 0 💰", interactive=False, elem_classes=["credit-display"])
                gr.Markdown("Upload fotomu untuk diubah ke gaya Ghibli!")
                with gr.Row():
                    user_image = gr.Image(label="Upload Gambar", type="pil", scale=1)
                    ghibli_output = gr.Image(label="Hasil Style Ghibli", scale=1)
                ghibli_btn = gr.Button("Terapkan Style Ghibli (5 Credits)", variant="primary")
                with gr.Column():
                    ghibli_msg = gr.Textbox(label="Pesan", interactive=False)
                    ghibli_download = gr.File(label="Download Gambar Ghibli", visible=False)

                def update_v3_tab(user_id):
                    return f"Credit: {get_credits(user_id)} 💰"

                image_v3_tab.select(update_v3_tab, user_id_state, credit_display_v3)

                def apply_ghibli_with_status_and_download(uploaded_img, user_id):
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
                    fn=apply_ghibli_with_status_and_download,
                    inputs=[user_image, user_id_state],
                    outputs=[ghibli_output, ghibli_msg, credit_display_v3, ghibli_download]
                )

        # --- Functions to handle login success --- 
        # These would be called by Flask after successful OAuth callback
        def handle_login_success(user_id, user_name, language, mood, avatar):
            # Update state variables
            # Update UI elements like user info display, credits
            # Switch to chat tab
            streak_msg = check_login_streak(user_id, user_name)
            welcome_msg = f"Login berhasil! Welcome back, {user_name}! {streak_msg}"
            return (
                user_id,
                language,
                mood,
                avatar,
                [], # Start with empty history for Gradio state, load_chat_data will populate
                welcome_msg,
                gr.update(selected=1) # Switch to Chat Tab
            )

        # This function needs to be callable from Flask, which is tricky.
        # Alternative: Flask sets a session variable, and Gradio checks it on load/refresh.
        # Or use a hidden component updated by Flask via JS after redirect.

        # Check login status on initial load (using Flask session)
        def check_initial_login():
            if 'user_email' in session:
                user_id = session['user_email']
                user_name = session.get('user_name', user_id)
                # Use default/stored language/mood/avatar or load from DB if implemented
                language = "Indonesia"
                mood = "Biasa 😐"
                avatar = assign_emoji(user_id)
                print(f"User {user_id} already logged in via Flask session.")
                # Call the login success handler to update Gradio state and switch tab
                return handle_login_success(user_id, user_name, language, mood, avatar)
            else:
                print("No active Flask session found.")
                # Stay on the welcome tab, return default/empty states
                return "", "Indonesia", "Biasa 😐", "😺", [], "Silakan login atau pilih mode tamu.", gr.update(selected=0)

        # Trigger check_initial_login when the Gradio app loads
        demo.load(
            fn=check_initial_login,
            inputs=[],
            outputs=[
                user_id_state,
                language_state,
                mood_state,
                avatar_state,
                chat_state,
                welcome_message, # Update welcome message based on login status
                tabs
            ]
        )

    return demo

# Create the Gradio interface instance
gradio_app = create_gradio_interface()

# Mount Gradio app onto Flask at the root URL
# Use gr.mount_gradio_app
app = gr.mount_gradio_app(app, gradio_app, path="/")

# Add a simple root route in Flask *before* mounting Gradio if needed,
# but Gradio mount usually handles the root.
# @app.route('/')
# def index():
#     # Redirect to Gradio path or render a landing page
#     # Since Gradio is mounted at '/', this might not be needed
#     return "Welcome to Kugy AI! Access the interface at /"

if __name__ == "__main__":
    # Run Flask app
    # Use '0.0.0.0' to be accessible externally (like on Render)
    # Debug should be False in production
    app.run(host="0.0.0.0", port=port, debug=False)

