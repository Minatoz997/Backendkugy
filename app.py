import zipfile
import os
port = int(os.getenv("PORT", 10000))
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
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_fixed
from time import sleep
from functools import wraps
import tempfile
import traceback
import sqlite3
import sys
import io
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import requests

# Setup OAuth
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = "https://backend-cb98.onrender.com/auth/callback"
SCOPES = ["profile", "email"]

@app.route("/auth/google")
def google_login():
    flow = InstalledAppFlow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = REDIRECT_URI
    authorization_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true"
    )
    return redirect(authorization_url)

@app.route("/auth/callback")
def google_callback():
    code = request.args.get("code")
    flow = InstalledAppFlow.from_client_config(
        {
            "web": {
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "redirect_uris": [REDIRECT_URI],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token"
            }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = REDIRECT_URI
    flow.fetch_token(code=code)
    credentials = flow.credentials
    user_info = requests.get(
        "https://www.googleapis.com/oauth2/v1/userinfo",
        headers={"Authorization": f"Bearer {credentials.token}"}
    ).json()
    email = user_info["email"]
    name = user_info["name"]
    # Simpan ke database (bisa ditambahkan nanti, misalnya SQLite)
    return redirect("https://backend-cb98.onrender.com")

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
    c.execute('''CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, credits INTEGER, login_streak INTEGER, last_login TEXT, last_guest_timestamp INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Fungsi Monetisasi
def check_credits(user_id, required_credits):
    if user_id in ADMIN_USERS:
        print(f"User {user_id} adalah admin, bypass pengecekan credit.")
        return True

    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    if not result:
        c.execute("INSERT INTO users (user_id, credits, login_streak, last_login, last_guest_timestamp) VALUES (?, ?, ?, ?, ?)", 
                  (user_id, 10, 0, datetime.now().strftime("%Y-%m-%d"), 0))
        conn.commit()
        conn.close()
        return required_credits <= 10
    if result[0] < required_credits:
        conn.close()
        return False
    c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (required_credits, user_id))
    conn.commit()
    conn.close()
    return True

def get_credits(user_id):
    if user_id in ADMIN_USERS:
        return "∞ (Admin)"
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 10

def get_credits_numeric(user_id):
    if user_id in ADMIN_USERS:
        return float('inf')
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 10

def top_up_credits(user_id, amount):
    if user_id in ADMIN_USERS:
        return "Kugy.ai: Kamu admin, credit kamu udah tak terbatas! 😎"
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO users (user_id, credits, login_streak, last_login, last_guest_timestamp) VALUES (?, COALESCE((SELECT credits FROM users WHERE user_id = ?) + ?, ?), COALESCE((SELECT login_streak FROM users WHERE user_id = ?), 0), COALESCE((SELECT last_login FROM users WHERE user_id = ?), ?), COALESCE((SELECT last_guest_timestamp FROM users WHERE user_id = ?), 0))",
              (user_id, user_id, amount, amount, user_id, user_id, datetime.now().strftime("%Y-%m-%d"), user_id))
    conn.commit()
    conn.close()
    return f"Kugy.ai: Credit {amount} ditambah! Total: {get_credits(user_id)} 💰"

def check_login_streak(user_id):
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT login_streak, last_login FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    today = datetime.now().strftime("%Y-%m-%d")
    if not result:
        c.execute("INSERT INTO users (user_id, credits, login_streak, last_login, last_guest_timestamp) VALUES (?, ?, ?, ?, ?)", 
                  (user_id, 10, 1, today, 0))
        conn.commit()
        conn.close()
        return "Selamat datang! Dapet 10 credit gratis nih~ 😸"
    streak, last_login = result
    last_date = datetime.strptime(last_login, "%Y-%m-%d")
    today_date = datetime.strptime(today, "%Y-%m-%d")
    if (today_date - last_date).days == 1:
        streak += 1
        if streak % 5 == 0:
            c.execute("UPDATE users SET credits = credits + 2, login_streak = ?, last_login = ? WHERE user_id = ?", (streak, today, user_id))
            conn.commit()
            conn.close()
            return f"Streak {streak} hari! Dapet bonus 2 credit! 🎉 Total: {get_credits(user_id)} 💰"
    elif (today_date - last_date).days > 1:
        streak = 1
    c.execute("UPDATE users SET login_streak = ?, last_login = ? WHERE user_id = ?", (streak, today, user_id))
    conn.commit()
    conn.close()
    return f"Streak login: {streak} hari. Ayo login tiap hari buat bonus! 😺"

def check_guest_limit(user_id):
    conn = sqlite3.connect("credits.db")
    c = conn.cursor()
    c.execute("SELECT last_guest_timestamp FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    current_time = datetime.now().timestamp()
    last_guest = result[0] if result else 0
    if current_time - last_guest < 24 * 3600:  # 24 jam cooldown
        conn.close()
        return False, int((24 * 3600 - (current_time - last_guest)) / 3600)
    c.execute("UPDATE users SET last_guest_timestamp = ? WHERE user_id = ?", (current_time, user_id))
    conn.commit()
    conn.close()
    return True, 0

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

missing_keys = [k for k, v in API_KEYS.items() if not v and k != "stability"]
if missing_keys:
    raise ValueError(f"Secrets {', '.join(missing_keys)} kosong!")

try:
    whisper_model = WhisperModel("tiny")
except Exception as e:
    raise RuntimeError(f"Gagal inisiasi Whisper: {str(e)}")

MAX_CACHE_SIZE = 50
image_cache = {}
session_file = "sessions.json"

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

def load_sessions():
    if not os.path.exists(session_file):
        with open(session_file, "w", encoding='utf-8') as f:
            json.dump({}, f)
        return {}
    try:
        with open(session_file, "r", encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_sessions(sessions_data):
    with open(session_file, "w", encoding='utf-8') as f:
        json.dump(sessions_data, f, indent=2, ensure_ascii=False)

sessions = load_sessions()

def assign_emoji(nickname):
    if "cat" in nickname.lower(): return "🐾"
    if "dog" in nickname.lower(): return "🐶"
    if "fox" in nickname.lower(): return "🦊"
    return "😺"

def get_expiry_warning(user_id, is_guest):
    if is_guest and user_id.startswith("guest_"):
        expiry = sessions[user_id]["expiry"]
        remaining = expiry - datetime.now().timestamp()
        if 0 < remaining <= 300:
            return f"⚠️ History kamu akan dihapus dalam {int(remaining/60)} menit!"
    return ""

def update_timer(user_id, is_guest):
    if is_guest and user_id.startswith("guest_"):
        expiry = sessions[user_id]["expiry"]
        remaining = max(0, int((expiry - datetime.now().timestamp()) / 60))
        return f"Mode Tamu: History akan dihapus dalam {remaining} menit"
    return ""

def show_last_message(user_input):
    if user_input in sessions and sessions[user_input].get("history"):
        last_msg = sessions[user_input]["history"][-1]["content"]
        return f"Pesan terakhir: {last_msg}"
    return "Belum ada obrolan sebelumnya."

def clean_expired_sessions():
    current_time = datetime.now().timestamp()
    expired_sessions = [key for key, value in sessions.items() if "expiry" in value and value["expiry"] < current_time]
    for key in expired_sessions:
        del sessions[key]
    save_sessions(sessions)

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
    headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Content-Type": "application/json"}
    payload = {"text_prompts": [{"text": prompt}], "cfg_scale": 7, "height": 1024, "width": 1024, "samples": 1, "steps": 30}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        print(f"Response status: {response.status_code}, Response text: {response.text}")
        if response.status_code == 200:
            resp_data = response.json()
            if "artifacts" in resp_data and resp_data["artifacts"]:
                base64_img = resp_data["artifacts"][0]["base64"]
                image = Image.open(BytesIO(base64.b64decode(base64_img)))
                if len(image_cache) >= MAX_CACHE_SIZE:
                    image_cache.pop(next(iter(image_cache)))
                image_cache[cache_key] = image
                return image, "Kugy.ai: Ini gambarnya buat ayang, cute banget kan? 🐱"
            return None, "Kugy.ai: Gagal dapetin gambar dari Stability... coba prompt lain ya!"
        return None, f"Kugy.ai: Error Stability AI (status {response.status_code}). Coba lagi nanti ya! Detail: {response.text}"
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return None, f"Kugy.ai: Error buat gambar: {str(e)}"

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_string

def save_image_for_download(image):
    try:
        if not image:
            print("Error: Gambar untuk download kosong!")
            return None

        print("Menyimpan gambar untuk download...")
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
        
        img1 = img1.resize((1024, 1024), Image.Resampling.LANCZOS)
        img2 = img2.resize((1024, 1024), Image.Resampling.LANCZOS)

        img1_base64 = image_to_base64(img1)
        headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Content-Type": "application/json"}
        intermediate_payload = {
            "init_image": img1_base64,
            "text_prompts": [{"text": prompt or "Blend two anime characters into a seamless style"}],
            "cfg_scale": 7,
            "steps": 50,
            "image_strength": 0.4,
            "style_preset": "anime",
        }
        print("Sending request to Stability AI for intermediate blending (Step 1)...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, json=intermediate_payload, timeout=60)
        print(f"Response status: {response.status_code}, Response text: {response.text}")

        if response.status_code != 200:
            print(f"Blending Step 1 failed: {response.text}")
            return None, f"Kugy.ai: Error Stability AI (Step 1, status {response.status_code}): {response.text}"

        resp_data = response.json()
        if "artifacts" not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in response for Step 1")
            return None, "Kugy.ai: Gagal dapetin hasil blending (Step 1)... coba lagi ya!"

        base64_img = resp_data["artifacts"][0]["base64"]
        intermediate_img = Image.open(BytesIO(base64.b64decode(base64_img)))

        intermediate_base64 = image_to_base64(intermediate_img)
        final_payload = {
            "init_image": intermediate_base64,
            "text_prompts": [{"text": prompt or "Blend two anime characters into a seamless style"}],
            "cfg_scale": 7,
            "steps": 50,
            "image_strength": 0.5,
            "style_preset": "anime",
        }
        print("Sending request to Stability AI for final blending (Step 2)...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, json=final_payload, timeout=60)
        print(f"Response status: {response.status_code}, Response text: {response.text}")

        if response.status_code != 200:
            print(f"Blending Step 2 failed: {response.text}")
            return None, f"Kugy.ai: Error Stability AI (Step 2, status {response.status_code}): {response.text}"

        resp_data = response.json()
        if "artifacts" not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in response for Step 2")
            return None, "Kugy.ai: Gagal dapetin hasil blending (Step 2)... coba lagi ya!"

        base64_img = resp_data["artifacts"][0]["base64"]
        final_img = Image.open(BytesIO(base64.b64decode(base64_img)))
        return final_img, "Kugy.ai: Yeay, gambar anime berhasil digabung dengan gaya AI! Keren banget, ayang! 🥰"

    except requests.exceptions.Timeout:
        print("Timeout error during blending")
        return None, "Kugy.ai: Timeout ke Stability AI, coba lagi nanti ya! 😿"
    except requests.exceptions.RequestException as e:
        print(f"Network error during blending: {str(e)}")
        return None, f"Kugy.ai: Error jaringan ke Stability AI: {str(e)} 😿"
    except Exception as e:
        print(f"Unexpected error during blending: {str(e)}")
        return None, f"Kugy.ai: Gagal blend gambar dengan AI: {str(e)} 😿"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def piapi_task(base_image, target_image, task_type, user_id):
    if not check_credits(user_id, 2):
        return "Kugy.ai: Credit kurang (butuh 2)! Top up yuk~ 💰"
    api_key = os.getenv("PIAPI_API_KEY")
    if not api_key:
        return "Kugy.ai: API Key Piapi gak ada, ayang! Set PIAPI_API_KEY dulu ya! 😿"
    
    max_res = 512
    if base_image.width > max_res or base_image.height > max_res:
        base_image = base_image.resize((min(base_image.width, max_res), min(base_image.height, max_res)), Image.Resampling.LANCZOS)
    if target_image.width > max_res or target_image.height > max_res:
        target_image = target_image.resize((min(target_image.width, max_res), min(target_image.height, max_res)), Image.Resampling.LANCZOS)

    if base_image.size != target_image.size:
        base_image = base_image.resize(target_image.size, Image.Resampling.LANCZOS)

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
    response = requests.post(
        PIAPI_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        job_id = result.get("data", {}).get("task_id")
        if job_id:
            poll_url = f"https://api.piapi.ai/api/v1/task/{job_id}"
            for attempt in range(10):
                poll_response = requests.get(poll_url, headers=headers, timeout=10)
                if poll_response.status_code == 200:
                    poll_result = poll_response.json()
                    if poll_result.get("data", {}).get("status") == "completed":
                        output = poll_result.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            image_url = output.get("image_url") or output.get("url")
                            image_base64 = output.get("image_base64")
                            if image_url:
                                image_response = requests.get(image_url, timeout=10)
                                if image_response.status_code == 200:
                                    return Image.open(BytesIO(image_response.content))
                                else:
                                    return f"Kugy.ai: Gagal ambil gambar dari URL (status {image_response.status_code})! 😿"
                            elif image_base64:
                                try:
                                    image_data = base64.b64decode(image_base64)
                                    return Image.open(BytesIO(image_data))
                                except Exception as e:
                                    return f"Kugy.ai: Gagal decode image_base64: {str(e)} 😿"
                            else:
                                return f"Kugy.ai: Output Piapi gak punya URL atau Base64, ayang! Output: {output} 😿"
                        elif isinstance(output, str):
                            image_url = output
                            image_response = requests.get(image_url, timeout=10)
                            if image_response.status_code == 200:
                                return Image.open(BytesIO(image_response.content))
                            else:
                                return f"Kugy.ai: Gagal ambil gambar dari URL (status {image_response.status_code})! 😿"
                        else:
                            return f"Kugy.ai: Format output Piapi gak sesuai, ayang! Output: {output} 😿"
                    elif poll_result.get("data", {}).get("status") == "failed":
                        return f"Kugy.ai: Task Piapi gagal, ayang! Pesan dari server: {poll_result.get('data', {}).get('message', 'Gak ada detail error')}. 😿"
                time.sleep(3)
            return "Kugy.ai: Timeout polling hasil Piapi, ayang! Coba lagi nanti! 😿"
        return "Kugy.ai: Gak dapet task ID dari Piapi, ayang! 😿"
    else:
        return f"Kugy.ai: Error Piapi (status {response.status_code}): {response.text}. Sabar ya, ayang! 😿"

def swap_couple(master_couple_img, face_pacar_img, face_kamu_img, task_type, user_id):
    if not check_credits(user_id, 2):
        return None, "Kugy.ai: Credit kurang (butuh 2)! Top up yuk~ 💰"
    if not master_couple_img or not face_pacar_img or not face_kamu_img:
        return None, "Kugy.ai: Harap unggah semua gambar terlebih dahulu, ayang! 😺"
    
    step1 = piapi_task(master_couple_img, face_pacar_img, task_type, user_id)
    if isinstance(step1, str):
        return None, f"Kugy.ai: Gagal di langkah 1: {step1}"
    
    final_img = piapi_task(step1, face_kamu_img, task_type, user_id)
    if isinstance(final_img, str):
        return None, f"Kugy.ai: Gagal di langkah 2: {final_img}"
    
    return final_img, "Kugy.ai: Yeay, foto couple-nya udah jadi! So sweet banget, ayang! 🥰"

def load_random_anime_face():
    try:
        dataset_path = "./anime_faces"
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"load_random_anime_face: Current working directory: {os.getcwd()}")
        print(f"load_random_anime_face: Mencari gambar di path absolut: {abs_dataset_path}")

        if not os.path.exists(abs_dataset_path):
            print(f"Folder {abs_dataset_path} tidak ditemukan!")
            return None, f"Kugy.ai: Folder {dataset_path} gak ditemukan! Pastiin foldernya ada ya, bro! 😿"

        image_files = []
        for root, dirs, files_in_dir in os.walk(abs_dataset_path):
             for file in files_in_dir:
                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                     image_files.append(os.path.join(root, file))

        print(f"load_random_anime_face: File gambar yang terdeteksi di {abs_dataset_path} dan subfolder: {len(image_files)}")
        if not image_files:
            all_items = []
            for root, dirs, files_in_dir in os.walk(abs_dataset_path):
                for item in dirs + files_in_dir:
                     all_items.append(os.path.join(root, item))
            print(f"load_random_anime_face: Semua item di folder (termasuk subfolder): {all_items}")
            return None, f"Kugy.ai: Gak ada gambar di folder {dataset_path} atau subfoldernya! Isi dulu ya, bro! 😿"

        random_image_path = random.choice(image_files)
        print(f"load_random_anime_face: Memuat gambar dari path: {random_image_path}")
        if not os.path.exists(random_image_path):
             print(f"Error: Path gambar terpilih {random_image_path} tidak ditemukan saat akan dibuka!")
             return None, f"Kugy.ai: Error internal, path gambar {random_image_path} tidak valid. 😿"

        img = Image.open(random_image_path).convert("RGB")
        return img, "Kugy.ai: Nih wajah anime random buat ayang! 😺"
    except Exception as e:
        print(f"Error saat load gambar: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Gagal load gambar: {str(e)} 😿"

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@rate_limit(max_per_minute=60)
def generate_anime_from_dataset(user_id):
    if not check_credits(user_id, 3):
        return None, "Kugy.ai: Credit kurang (butuh 3)! Top up yuk~ 💰"
    try:
        dataset_path = "./anime_faces"
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"generate_anime_from_dataset: Current working directory: {os.getcwd()}")
        print(f"generate_anime_from_dataset: Mengecek path absolut: {abs_dataset_path}")

        if not os.path.exists(abs_dataset_path):
            print(f"Folder {abs_dataset_path} tidak ditemukan!")
            return None, f"Kugy.ai: Folder {dataset_path} gak ditemukan! Pastiin foldernya ada ya, bro! 😿"

        image_files = []
        for root, dirs, files_in_dir in os.walk(abs_dataset_path):
             for file in files_in_dir:
                 if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                     image_files.append(os.path.join(root, file))

        print(f"generate_anime_from_dataset: File gambar yang terdeteksi (untuk pengecekan): {len(image_files)}")
        if not image_files:
             all_items = []
             for root, dirs, files_in_dir in os.walk(abs_dataset_path):
                 for item in dirs + files_in_dir:
                      all_items.append(os.path.join(root, item))
             print(f"generate_anime_from_dataset: Semua item di folder (termasuk subfolder): {all_items}")
             return None, f"Kugy.ai: Gak ada gambar di folder {dataset_path} atau subfoldernya untuk validasi! Isi dulu ya, bro! 😿"

        print("generate_anime_from_dataset: Folder dataset tidak kosong, melanjutkan untuk generate gambar baru via Stability AI...")
        prompt = f"Anime character with short {random.choice(['blue', 'red', 'black', 'blonde'])} hair, vibrant colors, detailed background"
        img, msg = generate_image(prompt, user_id)

        if img:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, format="PNG")
                print(f"generate_anime_from_dataset: Gambar baru disimpan sementara di {tmp.name}")
                return img, msg
        return None, msg
    except Exception as e:
        print(f"Error saat generate dari dataset: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Gagal generate dari dataset: {str(e)} 😿"

def find_reference_image(dataset_path, prompt):
    try:
        abs_dataset_path = os.path.abspath(dataset_path)
        print(f"find_reference_image: Mencari dataset di path absolut: {abs_dataset_path}")

        if not os.path.exists(abs_dataset_path):
            print(f"Dataset path not found: {abs_dataset_path}")
            return None, "default anime style"

        image_files = []
        for root, dirs, files in os.walk(abs_dataset_path):
            print(f"  Checking directory: {root}")
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
                    print(f"    Found image: {file}")

        if not image_files:
            print("No images found in dataset!")
            all_items = []
            for root, dirs, files in os.walk(abs_dataset_path):
                for item in dirs + files:
                    all_items.append(os.path.join(root, item))
            print(f"All items in dataset path (including subfolders): {all_items}")
            return None, "default anime style"

        print(f"Total images found: {len(image_files)}")

        if prompt:
            prompt_keywords = prompt.lower().split()
            print(f"Searching for keywords in filenames: {prompt_keywords}")
            for img_file in image_files:
                file_name = os.path.basename(img_file).lower()
                for keyword in prompt_keywords:
                    if keyword in file_name:
                        print(f"Match found: {img_file} with keyword '{keyword}'")
                        ref_img = Image.open(img_file).convert("RGB")
                        prompt_tag = file_name.split(".")[0].replace("_", " ").lower()
                        return ref_img, f"{prompt_tag}, detailed anime style"

        ref_img_path = random.choice(image_files)
        print(f"No keyword match, selecting random image: {ref_img_path}")
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
        print(f"User {user_id} lacks credits (needs 5).")
        return None, "Kugy.ai: Credit kurang (butuh 5)! Top up yuk~ 💰"

    try:
        if not isinstance(uploaded_img, Image.Image):
            print(f"Validation failed: uploaded_img type: {type(uploaded_img)}")
            return None, "Kugy.ai: Gambar input nggak valid, ayang! 😿"

        uploaded_img = uploaded_img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
        print(f"Resized image to: {uploaded_img.size}")

        buffer = BytesIO()
        uploaded_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        headers = {"Authorization": f"Bearer {API_KEYS['stability']}", "Accept": "application/json"}
        
        ghibli_prompt = (
            "Studio Ghibli style by Hayao Miyazaki, transform a human photo into an anime character with soft pastel colors, "
            "intricate hand-drawn details, lush natural background with cherry blossoms, flowing rivers, or serene forests, "
            "whimsical magical atmosphere with gentle warm lighting, expressive eyes, smooth skin texture, flowing hair, "
            "traditional Japanese clothing or natural attire, highly detailed, seamless transitions from photo to anime"
        )
        
        data = {
            "text_prompts[0][text]": ghibli_prompt,
            "cfg_scale": "10",
            "steps": "50",
            "image_strength": "0.4",
            "style_preset": "anime"
        }
        
        files = {
            'init_image': ('image.png', buffer, 'image/png')
        }
        
        print("Sending request to Stability AI for Ghibli style transfer (multipart)...")
        response = requests.post(STABILITY_IMAGE_TO_IMAGE_URL, headers=headers, files=files, data=data, timeout=60)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Ghibli style transfer failed: {response.text}")
            return None, f"Kugy.ai: Error Stability AI (status {response.status_code}): {response.text}"
            
        resp_data = response.json()
        if "artifacts" not in resp_data or not resp_data["artifacts"]:
            print("No artifacts in API response!")
            return None, "Kugy.ai: Gagal dapetin hasil dari Stability AI, coba lagi ya! 😿"
            
        generated_img_data = base64.b64decode(resp_data["artifacts"][0]["base64"])
        generated_img = Image.open(BytesIO(generated_img_data)).convert("RGB")
        print("Ghibli style transfer successful!")
        return generated_img, "Kugy.ai: Yeay, gambar dengan style Ghibli jadi! Keren banget, ayang! 🥰"
        
    except requests.exceptions.Timeout:
        print("Timeout error during Stability AI request")
        return None, "Kugy.ai: Timeout ke Stability AI, coba lagi nanti ya! 😿"
    except requests.exceptions.RequestException as e:
        print(f"Network error during Stability AI request: {str(e)}")
        return None, f"Kugy.ai: Error jaringan ke Stability AI: {str(e)} 😿"
    except Exception as e:
        print(f"Error during Ghibli style transfer: {str(e)}")
        traceback.print_exc()
        return None, f"Kugy.ai: Error: {str(e)} 😿"

def chat_with_ai(message, history, model_select, mode_dropdown, session_id, image=None, language="Indonesia", mood="Biasa"):
    if not check_credits(session_id, 1):
        return history, history + [{"role": "assistant", "content": "Kugy.ai: Credit kurang (butuh 1)! Top up yuk~ 💰"}]

    if session_id not in sessions:
        sessions[session_id] = {
            "history": [],
            "avatar": assign_emoji(session_id),
            "language": language,
            "mood": mood.split()[0],
            "expiry": datetime.now().timestamp() + 3600 if session_id.startswith("guest_") else None,
            "usage_count": 0 if session_id.startswith("guest_") else None
        }
    if session_id.startswith("guest_"):
        can_use, hours_left = check_guest_limit(session_id)
        if not can_use:
            return history, history + [{"role": "assistant", "content": f"Kugy.ai: Mode tamu cuma 1x/24 jam! Tunggu {hours_left} jam lagi atau top up, ayang! 💰"}]
        if sessions[session_id].get("usage_count", 0) >= 5:
            return history, history + [{"role": "assistant", "content": "Kugy.ai: Limit mode tamu (5 kali) tercapai! Top up buat akses penuh, ayang! 💰"}]
    
    history = sessions[session_id]["history"]
    history.append({"role": "user", "content": f"{sessions[session_id]['avatar']} {message}"})
    mood_value = mood.split()[0]
    user_name = session_id.split('@')[0] if not session_id.startswith("guest_") else "bro"

    if not history:
        greeting = f"{sessions[session_id]['avatar']} Yo, {user_name}! Keren banget balik lagi! 😎" if mode_dropdown == "Biasa" else f"{sessions[session_id]['avatar']} Halo, {user_name} sayang! Kangen nih! 🥰" if mode_dropdown == "Pacar" else f"{sessions[session_id]['avatar']} Yo, {user_name}! Siap kena roasting nih! 😂"
        history.append({"role": "assistant", "content": greeting})

    mood_messages = {
        "Senang": "Kamu Kugy.ai, pacar virtual ceria, manja, panggil 'sayang' atau 'ayang', kasih semangat biar happy, pake emoji lucu! 🥰",
        "Sedih": "Kamu Kugy.ai, pacar virtual penyayang, lembut, panggil 'sayang' atau 'ayang', hibur biar gak sedih, pake emoji hangat! 🤗",
        "Biasa": "Kamu Kugy.ai, pacar virtual santai, penyayang, panggil 'sayang' atau 'ayang', bicara hangat, lucu, romantis, pake emoji imut! 😺"
    } if mode_dropdown == "Pacar" else {
        "Senang": "Kamu Kugy.ai, asisten santai, bijak, gaul, kasih semangat biar tambah semangat, pake bahasa kekinian dan emoji! 🔥",
        "Sedih": "Kamu Kugy.ai, asisten santai, bijak, gaul, hibur biar gak sedih, pake bahasa kekinian dan emoji! 🥳",
        "Biasa": "Kamu Kugy.ai, asisten santai, bijak, gaul, pake bahasa kekinian, hindari kata afektif, pake emoji netral! 😎"
    } if mode_dropdown == "Biasa" else {
        "Senang": "Kamu Kugy.ai, roasting master, sindir santai, pake gaya gaul, kasih kritik kocak biar tambah semangat, pake emoji nakal! 😈",
        "Sedih": "Kamu Kugy.ai, roasting master, sindir lembut, pake gaya gaul, hibur dengan ejekan halus, pake emoji nakal! 😏",
        "Biasa": "Kamu Kugy.ai, roasting master, sindir tajam tapi asik, pake gaya gaul, kritik kocak, pake emoji nakal! 😂"
    }

    messages = [{"role": "system", "content": mood_messages.get(mood_value, mood_messages.get("Biasa", "Kamu Kugy.ai, asisten santai! 😎"))}] + history

    if image and model_select in ["OpenRouter", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"]:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analisis gambar ini, bro!"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]
        })

    reply = ""
    try:
        if model_select == "Mistral":
            from mistralai import Mistral
            client = Mistral(api_key=API_KEYS["mistral"])
            response = client.chat.complete(model="mistral-large-latest", messages=messages, temperature=0.7)
            reply = response.choices[0].message.content
        elif model_select == "OpenRouter":
            headers = {"Authorization": f"Bearer {API_KEYS['openrouter']}", "Content-Type": "application/json"}
            model_id = "anthropic/claude-3-haiku"  # Fallback aman kalau Grok error
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
                "model": model_id, "messages": messages, "temperature": 0.7
            }, timeout=60)
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        elif model_select == "Grok 3 Mini (OpenRouter)":
            headers = {"Authorization": f"Bearer {API_KEYS['openrouter']}", "Content-Type": "application/json"}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
                "model": "x-ai/grok-3-mini-beta", "messages": messages, "temperature": 0.7
            }, timeout=60)
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        elif model_select == "Gemini 2.0 Flash (OpenRouter)":
            headers = {"Authorization": f"Bearer {API_KEYS['openrouter']}", "Content-Type": "application/json"}
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json={
                "model": "google/gemini-2.0-flash-001", "messages": messages, "temperature": 0.7
            }, timeout=60)
            if response.status_code != 200:
                raise Exception(f"API Error: {response.status_code} - {response.text}")
            reply = response.json()["choices"][0]["message"]["content"]
        elif model_select == "DeepSeek":
            headers = {"Authorization": f"Bearer {API_KEYS['deepseek']}", "Content-Type": "application/json"}
            response = requests.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json={
                "model": "deepseek-chat", "messages": messages, "temperature": 0.7
            }, timeout=60)
            reply = response.json()["choices"][0]["message"]["content"]
        else:
            reply = "Model gak dikenal, bro!"
    except Exception as e:
        reply = f"Aduh, error nih: {str(e)}. Sabun dulu ya, bro! 😿"

    emoji_char = ""
    if mood_value == "Senang":
        emoji_char = "🎉"
    elif mood_value == "Sedih":
        emoji_char = "😔"
    else:
        emoji_char = "😎" if mode_dropdown == "Biasa" else "😂" if mode_dropdown == "Roasting" else "🥰"

    reply = f"{sessions[session_id]['avatar']} {reply} {emoji_char}"
    history.append({"role": "assistant", "content": reply})
    if session_id.startswith("guest_"):
        sessions[session_id]["usage_count"] += 1
    sessions[session_id]["history"] = history
    save_sessions(sessions)
    return history, history

def send_and_clear(message, history, model_select, mode_dropdown, session_id, image=None, language="Indonesia", mood="Biasa"):
    return "", *chat_with_ai(message, history, model_select, mode_dropdown, session_id, image, language, mood)

def respond(audio_path, history):
    if not audio_path:
        return history + [{"role": "assistant", "content": "Audio kosong, bro! Coba lagi ya!"}]
    try:
        segments, _ = whisper_model.transcribe(audio_path)
        message = " ".join(seg.text for seg in segments).strip()
        return history + [{"role": "user", "content": message or "Audio gak kedengeran, coba lagi ya!"}]
    except Exception as e:
        return history + [{"role": "assistant", "content": f"Error audio: {str(e)}"}]

# UI Gradio
demo = gr.Blocks(
    css="""
    #title {
        font-size: 30px;
        font-weight: bold;
        color: #D1E0D7;
        text-align: center;
        margin-bottom: 10px;
    }
    body {
        background-image: url('https://raw.githubusercontent.com/Minatoz997/angel_background.png/main/angel_background.png') !important;
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
        color: #ffffff !important;
        border: 2px solid #ffffff; /* Border putih biar kelihatan */
        background: rgba(0, 0, 0, 0.5); /* Background hitam semi-transparan */
        padding: 10px; /* Jarak dalam biar rapi */
        border-radius: 5px; /* Sudut rounded biar manis */
        display: inline-block; /* Biar kotaknya sesuai lebar teks */
    }
    .instruction-box {
        border: 1px solid #D7F2E6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .chatbot {
        font-size: 16px;
    }
    .gr-tabitem {
        background-color: #4A90E2 !important;
        border-radius: 5px;
        margin: 5px;
    }
    .gr-tabitem label {
        color: #FFFFFF !important;
        font-weight: bold;
    }
    .gr-tabitem:not(.selected) {
        opacity: 0.7;
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
        background: #FFD700;
        color: black;
        padding: 5px;
        border-radius: 5px;
        text-align: center;
    }
    """,
    theme=get_theme()
)

with demo:
    gr.HTML('<div id="title">kugy.ai — Asisten Imut Kamu 💙</div>')
    with gr.Tabs() as tabs:
        welcome_tab = gr.TabItem("Welcome", visible=True)
        chat_tab = gr.TabItem("Ngobrol", visible=False)
        image_tab = gr.TabItem("Gambar", visible=False)
        image_v2_tab = gr.TabItem("Gambar V2", visible=False)
        image_v3_tab = gr.TabItem("Gambar V3", visible=False)

        with welcome_tab:
            gr.Markdown("<div class='welcome-text'>### Selamat Datang di Kugy.ai!</div>")
            gr.Markdown("<div class='instruction-box'>Masukkan nickname/email (min 3 karakter) atau pilih Mode Tamu lalu klik Mulai Obrolan.</div>")
            user_input = gr.Textbox(label="Nickname / Email", placeholder="Contoh: kugyfan@email.com")
            avatars = ["😺", "🐶", "🦊", "🐼"]
            avatar_input = gr.Dropdown(choices=avatars, label="Pilih Avatar", value=random.choice(avatars))
            languages = ["Indonesia", "Inggris"]
            language_input = gr.Dropdown(choices=languages, label="Bahasa", value="Indonesia")
            moods = ["Senang 😊", "Sedih 🥺", "Biasa 😐"]
            mood_input = gr.Dropdown(choices=moods, label="Mood Hari Ini", value="Biasa 😐")
            with gr.Row():
                top_up_btn = gr.Button("Top Up Credit", variant="secondary")
                top_up_output = gr.Textbox(label="Instruksi Top Up", interactive=False)
            with gr.Row():
                guest_button = gr.Button("Mode Tamu", variant="secondary")
                register_button = gr.Button("Daftar dengan Email", variant="primary")
            welcome_message = gr.Textbox(label="Pesan", interactive=False)
            last_message = gr.Textbox(label="Pesan Terakhir", interactive=False)
            start_button = gr.Button("Mulai Obrolan")
            is_guest = gr.State(value=False)

            def guest_mode():
                user_id = f"guest_{int(datetime.now().timestamp())}"
                can_use, hours_left = check_guest_limit(user_id)
                if not can_use:
                    return (
                        "", True, user_id,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        f"Kugy.ai: Mode tamu cuma 1x/24 jam! Tunggu {hours_left} jam lagi atau top up, ayang! 💰",
                        show_last_message(user_id)
                    )
                expiry = datetime.now().timestamp() + 3600
                sessions[user_id] = {
                    "history": [],
                    "avatar": random.choice(avatars),
                    "expiry": expiry,
                    "language": "Indonesia",
                    "mood": "Biasa",
                    "usage_count": 0
                }
                save_sessions(sessions)
                streak_msg = check_login_streak(user_id)
                return (
                    "", True, user_id,
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    f"Mode Tamu aktif! {streak_msg}",
                    show_last_message(user_id)
                )

            def register_mode(email, avatar, language, mood):
                if len(email.strip()) < 3:
                    return (
                        email, False, email,
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        "Email/nickname minimal 3 karakter!",
                        show_last_message(email)
                    )
                user_id = email
                mood_value = mood.split()[0]
                sessions[user_id] = {
                    "history": [],
                    "avatar": assign_emoji(user_id),
                    "language": language,
                    "mood": mood_value
                }
                save_sessions(sessions)
                streak_msg = check_login_streak(user_id)
                return (
                    email, False, user_id,
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    f"Daftar berhasil! {streak_msg}",
                    show_last_message(user_id)
                )

            def validate_and_start(user_input_val, is_guest_val, avatar_val, language_val, mood_val):
                if not user_input_val and not is_guest_val:
                    return (
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        "Silakan masukkan nickname/email atau pilih mode tamu!",
                        show_last_message(user_input_val),
                        user_input_val, is_guest_val, avatar_val, language_val, mood_val
                    )
                if user_input_val and len(user_input_val) < 3 and not is_guest_val:
                    return (
                        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                        "Nickname/email minimal 3 karakter!",
                        show_last_message(user_input_val),
                        user_input_val, is_guest_val, avatar_val, language_val, mood_val
                    )
                user_id = user_input_val if user_input_val else f"guest_{int(time.time())}"
                expiry = time.time() + 3600 if is_guest_val else None
                mood_value = mood_val.split()[0]
                if user_id not in sessions:
                    sessions[user_id] = {
                        "history": [],
                        "avatar": avatar_val if is_guest_val else assign_emoji(user_id),
                        "language": language_val,
                        "mood": mood_value,
                        "expiry": expiry,
                        "usage_count": 0 if is_guest_val else None
                    }
                    save_sessions(sessions)
                return (
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    "",
                    show_last_message(user_id),
                    user_id, is_guest_val, avatar_val,
                    language_val, mood_val
                )

            guest_button.click(fn=guest_mode, inputs=[], outputs=[
                user_input, is_guest, user_input,
                welcome_tab, chat_tab, image_tab, image_v2_tab, image_v3_tab,
                welcome_message, last_message
            ])
            register_button.click(fn=register_mode, inputs=[
                user_input, avatar_input, language_input, mood_input
            ], outputs=[
                user_input, is_guest, user_input,
                welcome_tab, chat_tab, image_tab, image_v2_tab, image_v3_tab,
                welcome_message, last_message
            ])
            user_input.change(fn=show_last_message, inputs=user_input, outputs=last_message)
            start_button.click(
                fn=validate_and_start,
                inputs=[user_input, is_guest, avatar_input, language_input, mood_input],
                outputs=[
                    welcome_tab, chat_tab, image_tab, image_v2_tab, image_v3_tab,
                    welcome_message, last_message,
                    user_input, is_guest, avatar_input,
                    language_input, mood_input
                ]
            )
            top_up_btn.click(lambda: "Kirim pembayaran ke [QRIS/Norek], lalu hubungi admin untuk konfirmasi!", None, top_up_output)

        with chat_tab:
            credit_display = gr.Textbox(value="Credit Kamu: 0 💰", interactive=False, elem_classes=["credit-display"])
            user_id_state = gr.State("")
            session_dropdown = gr.Dropdown(list(sessions.keys()) or [None], value=None, label="Pilih Sesi Chat", allow_custom_value=True)
            chatbot = gr.Chatbot(type="messages", show_label=False, avatar_images=("https://i.ibb.co/yp4hMMV/kugy.png", None))
            state = gr.State([])
            timer_display = gr.Textbox(label="Waktu Tersisa (Mode Tamu)", interactive=False, visible=False)
            expiry_warning = gr.Textbox(label="Peringatan", interactive=False, visible=False)

            with gr.Row():
                btn_new = gr.Button("Buat Sesi Baru")
                btn_delete = gr.Button("Hapus Sesi")
                rename_input = gr.Textbox(label="Ganti Nama Sesi", placeholder="Nama baru...")
                rename_ok = gr.Button("OK", visible=True)
                rename_msg = gr.Textbox(label="", interactive=False, visible=False)

            with gr.Row():
                model_dropdown = gr.Dropdown(
                    ["Mistral", "OpenRouter", "DeepSeek", "Grok 3 Mini (OpenRouter)", "Gemini 2.0 Flash (OpenRouter)"],
                    value="Grok 3 Mini (OpenRouter)",
                    label="Pilih AI"
                )
                mode_dropdown = gr.Dropdown(["Biasa", "Pacar", "Roasting"], value="Pacar", label="Mode")

            with gr.Row():
                textbox = gr.Textbox(placeholder="Ketik di sini...", show_label=False)
                send_btn = gr.Button("Kirim")
                clear_btn = gr.Button("Reset")

            audio_input = gr.Audio(type="filepath", label="Kirim Suara")
            image_upload = gr.Image(type="pil", label="Upload Gambar")

            def update_credit_display(user_id):
                if user_id:
                    return f"Credit Kamu: {get_credits(user_id)} 💰"
                return "Credit Kamu: 0 💰"

            def load_session(selected, user_id):
                if selected and selected in sessions:
                    history = sessions[selected].get("history", [])
                    if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in history):
                        history = [{"role": "assistant", "content": "Sesi baru dibuat! Halo, bro! 😎"}] if not history else [
                            {"role": msg.get("role", "assistant"), "content": msg.get("content", "")} for msg in history
                        ]
                        sessions[selected]["history"] = history
                        save_sessions(sessions)
                    return history, history, update_credit_display(user_id)
                return [{"role": "assistant", "content": "Sesi kosong, buat pesan pertama ya! 😸"}], [{"role": "assistant", "content": "Sesi kosong, buat pesan pertama ya! 😸"}], update_credit_display(user_id)

            def on_new_session(user_id):
                new_id = f"session_{int(time.time())}"
                sessions[new_id] = {"history": [{"role": "assistant", "content": "Sesi baru dibuat! Halo, bro! 😎"}], "avatar": "😺", "language": "Indonesia", "mood": "Biasa"}
                save_sessions(sessions)
                return gr.update(choices=list(sessions.keys()), value=new_id), sessions[new_id]["history"], sessions[new_id]["history"], update_credit_display(user_id)

            def on_rename_session(session_id, new_name, user_id):
                if session_id in sessions and new_name and new_name not in sessions:
                    sessions[new_name] = sessions.pop(session_id)
                    save_sessions(sessions)
                    return gr.update(choices=list(sessions.keys()), value=new_name), sessions[new_name]["history"], sessions[new_name]["history"], "Sesi berhasil diganti nama!", list(sessions.keys()), update_credit_display(user_id)
                return session_dropdown, [], [], "Sesi gagal diganti nama!", list(sessions.keys()), update_credit_display(user_id)

            def on_delete_session(session_id, user_id):
                if session_id in sessions:
                    del sessions[session_id]
                    save_sessions(sessions)
                default_id = list(sessions.keys())[0] if sessions else None
                return gr.update(choices=list(sessions.keys()) or [None], value=default_id), \
                       sessions.get(default_id, {}).get("history", [{"role": "assistant", "content": "Sesi kosong, pilih atau buat sesi baru! 😺"}]) if default_id else [{"role": "assistant", "content": "Sesi kosong, pilih atau buat sesi baru! 😺"}], \
                       sessions.get(default_id, {}).get("history", [{"role": "assistant", "content": "Sesi kosong, pilih atau buat sesi baru! 😺"}]) if default_id else [{"role": "assistant", "content": "Sesi kosong, pilih atau buat sesi baru! 😺"}], update_credit_display(user_id)

            def update_chat(message, history, model_select, mode_dropdown, session_id, image, language, mood, is_guest, avatar, user_id):
                if not session_id or session_id not in sessions:
                    sessions[session_id] = {"history": [], "avatar": avatar, "language": language, "mood": mood.split()[0]}
                    save_sessions(sessions)
                    return "", [{"role": "assistant", "content": "Sesi baru dibuat! Halo, bro! 😎"}], [{"role": "assistant", "content": "Sesi baru dibuat! Halo, bro! 😎"}], update_credit_display(user_id)
                mood_value = mood.split()[0]
                try:
                    new_history, state_data = chat_with_ai(message, history, model_select, mode_dropdown, session_id, image, language, mood_value)
                    return "", new_history, state_data, update_credit_display(user_id)
                except Exception as e:
                    return "", history, history, update_credit_display(user_id)

            session_dropdown.change(load_session, [session_dropdown, user_id_state], [chatbot, state, credit_display])
            btn_new.click(on_new_session, [user_id_state], [session_dropdown, chatbot, state, credit_display])
            rename_ok.click(on_rename_session, [session_dropdown, rename_input, user_id_state], [session_dropdown, chatbot, state, rename_msg, session_dropdown, credit_display])
            btn_delete.click(on_delete_session, [session_dropdown, user_id_state], [session_dropdown, chatbot, state, credit_display])
            send_btn.click(update_chat, [textbox, state, model_dropdown, mode_dropdown, user_input, image_upload, language_input, mood_input, is_guest, avatar_input, user_id_state], [textbox, chatbot, state, credit_display])
            clear_btn.click(lambda: ("", [], []), None, [textbox, chatbot, state])
            audio_input.change(respond, [audio_input, state], chatbot)

        with image_tab:
            credit_display = gr.Textbox(value="Credit Kamu: 0 💰", interactive=False, elem_classes=["credit-display"])
            prompt_input = gr.Textbox(label="Prompt Gambar", placeholder="Contoh: 'Pantai malam'")
            generate_btn = gr.Button("Buat Gambar")
            image_output = gr.Image(label="Hasil Gambar")
            message_output = gr.Textbox(label="Pesan", interactive=False)
            download_file = gr.File(label="Download Gambar", visible=False)

            def generate_with_status(prompt, user_id):
                credits = get_credits_numeric(user_id)
                if user_id.startswith("guest_") and not check_credits(user_id, 3):
                    return None, "Kugy.ai: Mode tamu gak bisa buat gambar! Top up buat akses, ayang! 💰", f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)
                if not check_credits(user_id, 3):
                    return None, f"Kugy.ai: Credit kurang (butuh 3)! Top up yuk~ 💰", f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)
                
                image, msg = generate_image(prompt, user_id)
                
                if image:
                    image_path = save_image_for_download(image)
                    if image_path:
                        return image, msg, f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=image_path, visible=True, label="Download Gambar")
                
                return image, msg, f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)

            generate_btn.click(generate_with_status, [prompt_input, user_input], [image_output, message_output, credit_display, download_file])

        with image_v2_tab:
            credit_display = gr.Textbox(value="Credit Kamu: 0 💰", interactive=False, elem_classes=["credit-display"])
            master_couple = gr.Image(label="Foto Master Couple (Pasangan)", type="pil")
            face_pacar = gr.Image(label="Wajah Pacarmu", type="pil")
            face_kamu = gr.Image(label="Wajah Kamu", type="pil")
            task_type_input = gr.Dropdown(["face-swap", "hypothetical-new-model"], label="Pilih Tipe Task", value="face-swap")
            gen_btn = gr.Button("Generate Swap Couple")
            output = gr.Image(label="Hasil Face Swap")
            swap_message = gr.Textbox(label="Pesan", interactive=False)

            def swap_with_status(master_couple, face_pacar, face_kamu, task_type, user_id):
                if user_id.startswith("guest_") and not check_credits(user_id, 2):
                    return None, "Kugy.ai: Mode tamu gak bisa face swap! Top up buat akses, ayang! 💰", f"Credit Kamu: {get_credits(user_id)} 💰"
                if not check_credits(user_id, 2):
                    return None, f"Kugy.ai: Credit kurang (butuh 2)! Top up yuk~ 💰", f"Credit Kamu: {get_credits(user_id)} 💰"
                if not master_couple or not face_pacar or not face_kamu:
                    return None, "Kugy.ai: Harap unggah semua gambar terlebih dahulu, ayang! 😺", f"Credit Kamu: {get_credits(user_id)} 💰"
                
                image, msg = swap_couple(master_couple, face_pacar, face_kamu, task_type, user_id)
                return image, msg, f"Credit Kamu: {get_credits(user_id)} 💰"

            gen_btn.click(swap_with_status,
                          inputs=[master_couple, face_pacar, face_kamu, task_type_input, user_input],
                          outputs=[output, swap_message, credit_display])

        with image_v3_tab:
            credit_display = gr.Textbox(value="Credit Kamu: 0 💰", interactive=False, elem_classes=["credit-display"])
            
            with gr.Row():
                gr.Markdown("### Style Transfer Ghibli")
                
            user_image = gr.Image(label="Upload Gambar", type="pil")
            ghibli_btn = gr.Button("Terapkan Style Ghibli")
            ghibli_output = gr.Image(label="Hasil Style Ghibli")
            ghibli_msg = gr.Textbox(label="Pesan", interactive=False)
            ghibli_download = gr.File(label="Download Gambar Ghibli", visible=False)
            
            def apply_ghibli_with_status(uploaded_img, user_id):
                if user_id.startswith("guest_") and not check_credits(user_id, 5):
                    return None, "Kugy.ai: Mode tamu gak bisa Ghibli style! Top up buat akses, ayang! 💰", f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)
                if not check_credits(user_id, 5):
                    return None, f"Kugy.ai: Credit kurang (butuh 5)! Top up yuk~ 💰", f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)
                
                if not uploaded_img:
                    return None, "Kugy.ai: Upload gambar dulu, ayang! 😺", f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)
                
                image, msg = apply_ghibli_style(uploaded_img, user_id)
                
                if image:
                    image_path = save_image_for_download(image)
                    if image_path:
                        return image, msg, f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=image_path, visible=True, label="Download Gambar Ghibli")
                
                return image, msg, f"Credit Kamu: {get_credits(user_id)} 💰", gr.update(value=None, visible=False)
            
            ghibli_btn.click(apply_ghibli_with_status, 
                            inputs=[user_image, user_input],
                            outputs=[ghibli_output, ghibli_msg, credit_display, ghibli_download])

demo.launch(server_name="0.0.0.0", server_port=port)
