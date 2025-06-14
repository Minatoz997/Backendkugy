# Kugy AI Backend - Multi-Agent System

Backend API yang telah dioptimalkan untuk sistem multi-agent collaboration menggunakan 3 model AI dari OpenRouter.

## 🚀 Fitur Utama

### Multi-Agent System (KugyAgent)
- **Analyzer Agent**: Strategic Analysis & Task Decomposition
  - Models: deepseek/deepseek-coder, anthropic/claude-3.5-sonnet, meta-llama/llama-3.1-405b-instruct
  - Role: Menganalisis task dan memecah menjadi sub-komponen yang actionable

- **Researcher Agent**: Deep Research & Knowledge Synthesis  
  - Models: deepseek/deepseek-coder, meta-llama/llama-3.1-70b-instruct, anthropic/claude-3.5-sonnet
  - Role: Melakukan research mendalam dan menyediakan knowledge base

- **Synthesizer Agent**: Solution Integration & Final Output
  - Models: meta-llama/llama-3.1-405b-instruct, anthropic/claude-3.5-sonnet
  - Role: Mengintegrasikan hasil analisis dan research menjadi solusi final

### Optimasi Kolaborasi
- **Enhanced Task Detection**: Deteksi jenis task yang lebih akurat (coding, analysis, writing, general)
- **Iterative Collaboration**: Sistem feedback loop antar agent untuk hasil yang optimal
- **Quality Metrics**: Penilaian kualitas real-time untuk setiap iterasi
- **Convergence Detection**: Penghentian otomatis ketika solusi optimal tercapai

## 📊 Performance Metrics

- **Collaboration Score**: Skor efektivitas kolaborasi antar agent
- **Quality Assessment**: Penilaian completeness, accuracy, dan practicality
- **Processing Time**: Waktu pemrosesan yang dioptimalkan
- **Feedback Exchanges**: Tracking komunikasi antar agent

## 🛠️ API Endpoints

### Multi-Agent System
- `POST /api/multi-agent` - Process task dengan multi-agent collaboration
- `GET /multi-agent/status` - Status dan konfigurasi agent

### Other Features
- `POST /api/chat` - Chat completion
- `POST /api/generate-image` - Image generation
- `POST /virtusim/orders/create` - Virtual SIM orders
- `GET /api/user` - User authentication
- `GET /api/credits` - Credit management

## 🔧 Installation

```bash
pip install -r requirements.txt
python app.py
```

## 🌐 Environment Variables

```env
OPENROUTER_API_KEY=your_openrouter_api_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
FRONTEND_URL=https://your-frontend-url.com
DATABASE_URL=sqlite:///credits.db
```

## 📈 Usage Example

```python
import requests

# Multi-agent task processing
response = requests.post('http://localhost:5000/api/multi-agent', 
    json={
        'task': 'Create a Python web scraper for e-commerce data',
        'use_multi_agent': True
    },
    headers={'Content-Type': 'application/json'}
)

result = response.json()
print(f"Solution: {result['solution']}")
print(f"Collaboration Score: {result['collaboration_score']}")
print(f"Processing Time: {result['processing_time']}")
```

## 🎯 Credit Costs

- Chat: 1 credit
- Image Generation: 3 credits  
- Multi-Agent (KugyAgent): 5 credits
- Virtual SIM: 25 credits

## 🔄 Multi-Agent Workflow

1. **Task Analysis** (Analyzer Agent)
   - Deteksi jenis dan kompleksitas task
   - Decomposisi menjadi sub-tasks
   - Strategic planning

2. **Deep Research** (Researcher Agent)
   - Research mendalam untuk setiap sub-task
   - Pengumpulan referensi dan best practices
   - Knowledge synthesis

3. **Solution Integration** (Synthesizer Agent)
   - Integrasi hasil analisis dan research
   - Penyusunan solusi final yang komprehensif
   - Quality assurance dan optimization

## 📝 Response Format

```json
{
  "success": true,
  "task_id": "uuid",
  "task": "Original task description",
  "task_type": "coding|analysis|writing|general",
  "solution": "Final comprehensive solution",
  "format": "text|code|table|mixed",
  "explanation": "Detailed explanation",
  "implementation_steps": [...],
  "quality_metrics": {
    "completeness": "8.5",
    "accuracy": "9.0", 
    "practicality": "8.8"
  },
  "validation_checklist": [...],
  "next_steps": [...],
  "iterations": 3,
  "collaboration_score": 0.92,
  "models_used": {
    "analyzer": "anthropic/claude-3.5-sonnet",
    "researcher": "meta-llama/llama-3.1-70b-instruct", 
    "synthesizer": "meta-llama/llama-3.1-405b-instruct"
  },
  "collaboration_metrics": {
    "total_feedback_exchanges": 5,
    "quality_improvements": [7.2, 8.1, 8.9],
    "convergence_score": 0.92
  },
  "processing_time": "12.34 seconds"
}
```

## 🚀 Deployment

Backend dapat di-deploy ke:
- Render.com
- Heroku  
- Railway
- VPS/Cloud Server

Pastikan environment variables sudah dikonfigurasi dengan benar.

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details.