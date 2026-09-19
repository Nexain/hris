import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers.chat import router as chat_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hris-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        f"Starting {settings.app_name} in {settings.app_env} mode. "
        f"Configured GCP Project: '{settings.gcp_project_id or '<not configured>'}', "
        f"Location: '{settings.gcp_location}', "
        f"Model: '{settings.gemini_model}'"
    )
    yield
    logger.info(f"Shutting down {settings.app_name}")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="FastAPI service acting as a bridge to communicate with Gemini in Google Cloud Vertex AI.",
    lifespan=lifespan,
)

# Enable CORS for cross-origin web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat_router)


@app.get("/", tags=["System"])
async def root():
    """Root endpoint welcoming users and directing to docs."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "gcp_location": settings.gcp_location,
        "gemini_model": settings.gemini_model,
        "gcp_project_configured": bool(
            settings.gcp_project_id
            and settings.gcp_project_id != "your-gcp-project-id"
        ),
    }

