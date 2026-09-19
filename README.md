# FastAPI Vertex AI Gemini Bridge

A production-ready FastAPI service that provides a `/chat` endpoint bridging client applications with Google Cloud Vertex AI's Gemini models.

---

## Features

- **Vertex AI Gemini Integration**: Built on Google's modern `google-genai` SDK with native Vertex AI integration (`vertexai=True`).
- **Async Execution**: Asynchronous request handling via `client.aio.models.generate_content`.
- **Multi-turn Chat & History**: Full support for conversation history (`role: user | model | system`).
- **Streaming Response**: Real-time Server-Sent Events (SSE) streaming with `stream: true`.
- **Configurable**: Model selection, sampling temperature, max output tokens, and GCP credentials via environment variables or per-request parameters.
- **Interactive Documentation**: Built-in Swagger UI at `/docs` and ReDoc at `/redoc`.
- **Unit & Integration Tests**: Comprehensive test suite using `pytest` and `httpx`.

---

## Project Structure

```
hris/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application entrypoint, CORS, lifecycle, /health
│   ├── config.py                # Pydantic Settings (.env configuration)
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── chat.py              # Request / response schemas
│   ├── services/
│   │   ├── __init__.py
│   │   └── vertex_service.py    # Vertex AI Gemini client wrapper
│   └── routers/
│       ├── __init__.py
│       └── chat.py              # /chat router (JSON and SSE streaming)
├── tests/
│   ├── __init__.py
│   └── test_chat.py             # Pytest suite with mocks and edge cases
├── .env.example                 # Environment variables template
├── .env                         # Active configuration
├── .gitignore                   # Excludes .venv, cache, and sensitive files
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Cloud Project with the **Vertex AI API** enabled (`aiplatform.googleapis.com`).

### 2. Activate Virtual Environment & Install Dependencies

If not already activated:
```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Or run directly using the venv Python binary:
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Configure Google Cloud Credentials

Copy the template `.env.example` to `.env`:
```powershell
cp .env.example .env
```

Edit `.env` with your Google Cloud details:
```ini
# Google Cloud Project ID (Required)
GCP_PROJECT_ID=your-google-cloud-project-id

# Vertex AI Region (e.g., us-central1, asia-southeast1, europe-west1)
GCP_LOCATION=us-central1

# Default Gemini model
GEMINI_MODEL=gemini-1.5-flash

# Optional: Path to Service Account JSON key file
# If left empty, Google Application Default Credentials (ADC) will be used
GOOGLE_APPLICATION_CREDENTIALS=
```

#### Authentication Options:
1. **Google Cloud CLI (Recommended for Local Dev)**:
   ```bash
   gcloud auth application-default login
   ```
2. **Service Account Key (Recommended for Production / Containers)**:
   - Create a service account with the **Vertex AI User** (`roles/aiplatform.user`) role.
   - Download the JSON key file and set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` in `.env`.

---

## Running the Server

Start the Uvicorn development server:
```powershell
.\.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

Once running:
- **API Base URL**: `http://127.0.0.1:8000`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **Health Check**: `http://127.0.0.1:8000/health`

---

## API Usage (`POST /chat`)

### 1. Basic Single Prompt

**Request**:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain what an HRIS is in two sentences."
  }'
```

**Response**:
```json
{
  "response": "An HRIS (Human Resources Information System) is software used to store, manage, and process employee data and core HR operations in a centralized location. It streamlines functions like payroll, benefits administration, recruitment, and performance tracking to enhance organizational efficiency.",
  "model": "gemini-1.5-flash",
  "usage": {
    "prompt_tokens": 15,
    "candidates_tokens": 42,
    "total_tokens": 57
  },
  "finish_reason": "STOP"
}
```

---

### 2. Multi-turn Conversation with History & System Prompt

**Request**:
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What leave balance do they have?",
    "system_instruction": "You are an HR Assistant. Answer concisely.",
    "history": [
      {
        "role": "user",
        "content": "Employee John Doe requested annual leave yesterday."
      },
      {
        "role": "model",
        "content": "I have noted John Doe'\''s request for annual leave."
      }
    ],
    "temperature": 0.3
  }'
```

---

### 3. Real-time Streaming Response (Server-Sent Events)

Set `"stream": true` to receive token chunks as they arrive:
```bash
curl -N -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a 3-bullet onboarding checklist.",
    "stream": true
  }'
```

**Stream Output**:
```
data: {"text": "Here is an onboarding checklist:\n"}

data: {"text": "1. Set up workstation and software accounts.\n"}

data: {"text": "2. Complete required HR forms and benefits enrollment.\n"}

data: {"text": "3. Introduce the team and assign a peer buddy."}

data: [DONE]
```

---

### 4. Custom Model & Parameters Override

You can override model name and generation hyperparameters per request:
```json
{
  "message": "Write a formal employee termination policy summary.",
  "model": "gemini-1.5-pro",
  "temperature": 0.2,
  "max_output_tokens": 1024
}
```

---

## Running Tests

Execute the automated test suite using `pytest`:
```powershell
.\.venv\Scripts\pytest
```