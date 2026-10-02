import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.onboarding import router as onboarding_router
from app.api.tasks import router as tasks_router
from app.config import get_settings
from app.core.errors import AppException, app_exception_handler

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hris-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    curr_settings = get_settings()
    logger.info(
        f"Starting {curr_settings.app_name} in {curr_settings.app_env} mode. "
        f"GCP Project: '{curr_settings.gcp_project_id or '<local/unconfigured>'}', "
        f"Location: '{curr_settings.gcp_location}', "
        f"Gemini Model: '{curr_settings.gemini_model}'"
    )
    yield
    logger.info(f"Shutting down {curr_settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="DayOne AI Backend — AI Onboarding Copilot with Knowledge Base & Document Ingestion",
    lifespan=lifespan,
)

# Exception handlers
app.add_exception_handler(AppException, app_exception_handler)

# Enable CORS for cross-origin web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers under /api/v1 (spec base path)
app.include_router(chat_router, prefix="/api/v1")
app.include_router(documents_router, prefix="/api/v1")
app.include_router(onboarding_router, prefix="/api/v1")
app.include_router(tasks_router, prefix="/api/v1")




@app.get("/", tags=["System"])
async def root():
    """Root endpoint welcoming users and directing to docs."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs_url": "/docs",
        "health_check": "/health",
        "api_v1_base": "/api/v1",
    }


@app.get("/health", tags=["System"])
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "gcp_location": settings.gcp_location,
        "gemini_model": settings.gemini_model,
        "gcp_project_configured": bool(
            settings.gcp_project_id
            and settings.gcp_project_id != "your-gcp-project-id"
        ),
    }

