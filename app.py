import os
import json
import logging
import base64
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import uuid
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'authapp:'

# Initialize extensions
Session(app)

# CORS configuration for cross-origin requests
CORS(app, 
     origins=[
         "http://localhost:5000",
         "http://localhost:3000", 
         "https://*.vercel.app",
         "https://*.replit.dev",
         "*"  # Allow all origins for development
     ],
     supports_credentials=True,
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization', 'X-Requested-With'])

# API Configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
STABILITY_BASE_URL = "https://api.stability.ai"

# OpenRouter Models Configuration
OPENROUTER_MODELS = {
    "gpt-4": "openai/gpt-4",
    "claude-3": "anthropic/claude-3-sonnet",
    "gemini-pro": "google/gemini-pro"
}

# Database setup using PostgreSQL
@contextmanager
def get_db():
    """Database connection context manager"""
    try:
        conn = psycopg2.connect(
            os.environ['DATABASE_URL'],
            cursor_factory=RealDictCursor
        )
        yield conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def init_db():
    """Initialize the database with required tables"""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                # Users table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        provider TEXT NOT NULL,
                        is_guest BOOLEAN DEFAULT FALSE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # User sessions table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        ip_address TEXT,
                        user_agent TEXT,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # AI conversations table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS ai_conversations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        tokens_used INTEGER DEFAULT 0,
                        cost DECIMAL(10,6) DEFAULT 0.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # Generated images table
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS generated_images (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        image_url TEXT,
                        image_data TEXT,
                        model_name TEXT DEFAULT 'stability-ai',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")

# Initialize database on startup
init_db()

# Authentication routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'service': 'AuthApp Backend with AI APIs',
        'apis_available': {
            'openrouter': bool(OPENROUTER_API_KEY),
            'stability': bool(STABILITY_API_KEY)
        }
    })

@app.route('/user/profile', methods=['GET'])
def get_user_profile():
    """Get user profile information"""
    try:
        user_id = session.get('user_id')
        auth_header = request.headers.get('Authorization')
        
        if not user_id and not auth_header:
            return jsonify({'error': 'No authentication provided'}), 401
        
        if user_id:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT * FROM users WHERE id = %s', (user_id,))
                    user_row = cur.fetchone()
                    
                    if user_row:
                        # Get conversation count
                        cur.execute('SELECT COUNT(*) as count FROM ai_conversations WHERE user_id = %s', (user_id,))
                        conv_count = cur.fetchone()['count']
                        
                        # Get image count
                        cur.execute('SELECT COUNT(*) as count FROM generated_images WHERE user_id = %s', (user_id,))
                        img_count = cur.fetchone()['count']
                        
                        user_data = {
                            'id': user_row['id'],
                            'name': user_row['name'],
                            'email': user_row['email'],
                            'isGuest': user_row['is_guest'],
                            'provider': user_row['provider'],
                            'loginTime': user_row['created_at'].isoformat() if user_row['created_at'] else None,
                            'stats': {
                                'conversations': conv_count,
                                'images_generated': img_count
                            },
                            'sessionInfo': {
                                'id': session.get('session_id'),
                                'createdAt': datetime.utcnow().isoformat(),
                                'lastAccessed': datetime.utcnow().isoformat(),
                                'ipAddress': request.remote_addr,
                                'userAgent': request.headers.get('User-Agent', '')
                            }
                        }
                        return jsonify(user_data)
        
        # Fallback to mock data if no session
        user_data = {
            'id': str(uuid.uuid4()),
            'name': 'Test User',
            'email': 'test@example.com',
            'isGuest': False,
            'provider': 'google',
            'loginTime': datetime.utcnow().isoformat(),
            'stats': {'conversations': 0, 'images_generated': 0}
        }
        
        return jsonify(user_data)
        
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        return jsonify({'error': 'Failed to get user profile'}), 500

@app.route('/guest/session', methods=['POST'])
def create_guest_session():
    """Create a new guest session"""
    try:
        data = request.get_json()
        
        if not data or 'username' not in data:
            return jsonify({'error': 'Username is required'}), 400
        
        username = data['username'].strip()
        
        if len(username) < 2:
            return jsonify({'error': 'Username must be at least 2 characters'}), 400
        
        if len(username) > 20:
            return jsonify({'error': 'Username must be less than 20 characters'}), 400
        
        # Create guest user
        guest_id = f"guest_{uuid.uuid4()}"
        guest_email = f"{guest_id}@guest.local"
        
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO users (id, name, email, provider, is_guest)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (email) DO UPDATE SET
                    name = EXCLUDED.name,
                    updated_at = CURRENT_TIMESTAMP
                ''', (guest_id, username, guest_email, 'guest', True))
                
                # Create session record
                session_id = str(uuid.uuid4())
                cur.execute('''
                    INSERT INTO user_sessions (
                        id, user_id, created_at, last_accessed, 
                        expires_at, ip_address, user_agent
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    session_id, guest_id, datetime.utcnow(), datetime.utcnow(),
                    datetime.utcnow() + timedelta(days=30),
                    request.remote_addr, request.headers.get('User-Agent', '')
                ))
                
                conn.commit()
        
        # Store session info
        session['user_id'] = guest_id
        session['session_id'] = session_id
        session['is_guest'] = True
        
        return jsonify({
            'success': True,
            'user': {
                'id': guest_id,
                'name': username,
                'email': guest_email,
                'isGuest': True,
                'sessionId': session_id
            }
        })
        
    except Exception as e:
        logger.error(f"Error creating guest session: {str(e)}")
        return jsonify({'error': 'Failed to create guest session'}), 500

# OpenRouter API Integration
@app.route('/ai/chat', methods=['POST'])
def chat_with_ai():
    """Chat with AI using OpenRouter API"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not OPENROUTER_API_KEY:
            return jsonify({'error': 'OpenRouter API key not configured'}), 500
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        model = data.get('model', 'gpt-4')
        message = data['message']
        
        # Get model endpoint
        model_endpoint = OPENROUTER_MODELS.get(model, OPENROUTER_MODELS['gpt-4'])
        
        # Prepare request to OpenRouter
        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
            'X-Title': 'AuthApp AI Chat'
        }
        
        payload = {
            'model': model_endpoint,
            'messages': [
                {'role': 'user', 'content': message}
            ],
            'max_tokens': 1000,
            'temperature': 0.7
        }
        
        # Make request to OpenRouter
        response = requests.post(
            f'{OPENROUTER_BASE_URL}/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"OpenRouter API error: {response.text}")
            return jsonify({'error': 'AI service temporarily unavailable'}), 503
        
        result = response.json()
        ai_response = result['choices'][0]['message']['content']
        tokens_used = result.get('usage', {}).get('total_tokens', 0)
        
        # Save conversation to database
        conversation_id = str(uuid.uuid4())
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO ai_conversations (id, user_id, model_name, prompt, response, tokens_used)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (conversation_id, user_id, model, message, ai_response, tokens_used))
                conn.commit()
        
        return jsonify({
            'success': True,
            'response': ai_response,
            'model': model,
            'tokens_used': tokens_used,
            'conversation_id': conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}")
        return jsonify({'error': 'Failed to process AI request'}), 500

@app.route('/ai/multi-agent', methods=['POST'])
def multi_agent_chat():
    """Multi-agent chat using all 3 models"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not OPENROUTER_API_KEY:
            return jsonify({'error': 'OpenRouter API key not configured'}), 500
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        message = data['message']
        responses = {}
        
        # Query all three models
        for model_name, model_endpoint in OPENROUTER_MODELS.items():
            try:
                headers = {
                    'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                    'Content-Type': 'application/json',
                    'X-Title': f'AuthApp Multi-Agent - {model_name}'
                }
                
                payload = {
                    'model': model_endpoint,
                    'messages': [
                        {'role': 'user', 'content': message}
                    ],
                    'max_tokens': 800,
                    'temperature': 0.7
                }
                
                response = requests.post(
                    f'{OPENROUTER_BASE_URL}/chat/completions',
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result['choices'][0]['message']['content']
                    tokens_used = result.get('usage', {}).get('total_tokens', 0)
                    
                    responses[model_name] = {
                        'response': ai_response,
                        'tokens_used': tokens_used,
                        'success': True
                    }
                    
                    # Save each conversation
                    conversation_id = str(uuid.uuid4())
                    with get_db() as conn:
                        with conn.cursor() as cur:
                            cur.execute('''
                                INSERT INTO ai_conversations (id, user_id, model_name, prompt, response, tokens_used)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            ''', (conversation_id, user_id, model_name, message, ai_response, tokens_used))
                            conn.commit()
                else:
                    responses[model_name] = {
                        'error': f'API error: {response.status_code}',
                        'success': False
                    }
                    
            except Exception as e:
                logger.error(f"Error with {model_name}: {str(e)}")
                responses[model_name] = {
                    'error': str(e),
                    'success': False
                }
        
        return jsonify({
            'success': True,
            'responses': responses,
            'total_models': len(OPENROUTER_MODELS)
        })
        
    except Exception as e:
        logger.error(f"Error in multi-agent chat: {str(e)}")
        return jsonify({'error': 'Failed to process multi-agent request'}), 500

# Stability AI Integration
@app.route('/ai/generate-image', methods=['POST'])
def generate_image():
    """Generate image using Stability AI"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        if not STABILITY_API_KEY:
            return jsonify({'error': 'Stability AI API key not configured'}), 500
        
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Prompt is required'}), 400
        
        prompt = data['prompt']
        model = data.get('model', 'stable-diffusion-xl-1024-v1-0')
        
        # Prepare request to Stability AI
        headers = {
            'Authorization': f'Bearer {STABILITY_API_KEY}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        payload = {
            'text_prompts': [
                {
                    'text': prompt,
                    'weight': 1.0
                }
            ],
            'cfg_scale': 7,
            'height': 1024,
            'width': 1024,
            'samples': 1,
            'steps': 30
        }
        
        # Make request to Stability AI
        response = requests.post(
            f'{STABILITY_BASE_URL}/v1/generation/{model}/text-to-image',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Stability AI error: {response.text}")
            return jsonify({'error': 'Image generation service temporarily unavailable'}), 503
        
        result = response.json()
        
        # Get the generated image data
        image_data = result['artifacts'][0]['base64']
        
        # Save to database
        image_id = str(uuid.uuid4())
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO generated_images (id, user_id, prompt, image_data, model_name)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (image_id, user_id, prompt, image_data, model))
                conn.commit()
        
        return jsonify({
            'success': True,
            'image_id': image_id,
            'image_data': image_data,
            'prompt': prompt,
            'model': model
        })
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({'error': 'Failed to generate image'}), 500

@app.route('/ai/images/<image_id>', methods=['GET'])
def get_generated_image(image_id):
    """Get a generated image by ID"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT * FROM generated_images 
                    WHERE id = %s AND user_id = %s
                ''', (image_id, user_id))
                
                image_row = cur.fetchone()
                
                if not image_row:
                    return jsonify({'error': 'Image not found'}), 404
                
                return jsonify({
                    'id': image_row['id'],
                    'prompt': image_row['prompt'],
                    'image_data': image_row['image_data'],
                    'model_name': image_row['model_name'],
                    'created_at': image_row['created_at'].isoformat()
                })
        
    except Exception as e:
        logger.error(f"Error getting image: {str(e)}")
        return jsonify({'error': 'Failed to get image'}), 500

@app.route('/ai/conversations', methods=['GET'])
def get_conversations():
    """Get user's AI conversation history"""
    try:
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'error': 'Authentication required'}), 401
        
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        offset = (page - 1) * limit
        
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    SELECT * FROM ai_conversations 
                    WHERE user_id = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s OFFSET %s
                ''', (user_id, limit, offset))
                
                conversations = []
                for row in cur.fetchall():
                    conversations.append({
                        'id': row['id'],
                        'model_name': row['model_name'],
                        'prompt': row['prompt'],
                        'response': row['response'],
                        'tokens_used': row['tokens_used'],
                        'created_at': row['created_at'].isoformat()
                    })
                
                return jsonify({
                    'conversations': conversations,
                    'page': page,
                    'limit': limit
                })
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return jsonify({'error': 'Failed to get conversations'}), 500

@app.route('/ai/models', methods=['GET'])
def get_available_models():
    """Get list of available AI models"""
    return jsonify({
        'openrouter_models': list(OPENROUTER_MODELS.keys()),
        'stability_models': [
            'stable-diffusion-xl-1024-v1-0',
            'stable-diffusion-v1-6',
            'stable-diffusion-512-v2-1'
        ],
        'api_status': {
            'openrouter': bool(OPENROUTER_API_KEY),
            'stability': bool(STABILITY_API_KEY)
        }
    })

# Existing authentication routes
@app.route('/logout', methods=['POST', 'GET'])
def logout():
    """Logout user and cleanup session"""
    try:
        user_id = session.get('user_id')
        session_id = session.get('session_id')

        if session_id:
            with get_db() as conn:
                with conn.cursor() as cur:
                    cur.execute('DELETE FROM user_sessions WHERE id = %s', (session_id,))
                    conn.commit()

        session.clear()
        logger.info(f"User {user_id} logged out successfully")
        return jsonify({'success': True, 'message': 'Logged out successfully'})

    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return jsonify({'error': 'Logout failed'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def before_request():
    """Log all requests for debugging"""
    logger.info(f"{request.method} {request.path} - {request.remote_addr}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Frontend URL: {os.environ.get('FRONTEND_URL', 'http://localhost:5000')}")
    logger.info(f"OpenRouter API: {'✓' if OPENROUTER_API_KEY else '✗'}")
    logger.info(f"Stability AI: {'✓' if STABILITY_API_KEY else '✗'}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)