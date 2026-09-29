from unittest.mock import AsyncMock, MagicMock
import pytest
import httpx

from app.main import app
from app.services.vertex_service import get_vertex_service
from app.schemas.chat import ChatResponse, TokenUsage


@pytest.fixture
def mock_vertex_service():
    """Create a mock VertexGeminiService for testing without live GCP credentials."""
    mock_service = MagicMock()
    mock_service.generate_chat = AsyncMock()
    mock_service.generate_chat_stream = MagicMock()
    return mock_service


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root endpoint."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "Welcome" in data["message"]
        assert data["health_check"] == "/health"


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Test the health check endpoint."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "gcp_location" in data
        assert "gemini_model" in data


@pytest.mark.asyncio
async def test_chat_validation_empty_message():
    """Test validation when message is empty."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/chat", json={"message": ""})
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_missing_project_id_default():
    """Test that missing or placeholder GCP_PROJECT_ID returns a clear 500 error."""
    # Ensure default dependency without mocking
    app.dependency_overrides.clear()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={"message": "Hello, Gemini!"},
        )
        assert response.status_code == 500
        assert "GCP Project ID is not configured" in response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_success_with_mock(mock_vertex_service):
    """Test successful chat generation with mocked Vertex AI service."""
    mock_response = ChatResponse(
        response="Hello! I am Gemini on Vertex AI.",
        model="gemini-1.5-flash",
        usage=TokenUsage(prompt_tokens=10, candidates_tokens=15, total_tokens=25),
        finish_reason="STOP",
    )
    mock_vertex_service.generate_chat.return_value = mock_response

    app.dependency_overrides[get_vertex_service] = lambda: mock_vertex_service

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "message": "Hello!",
                "history": [
                    {"role": "user", "content": "Hi"},
                    {"role": "model", "content": "Greetings! How may I assist?"},
                ],
                "system_instruction": "You are a professional HR assistant.",
                "temperature": 0.5,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello! I am Gemini on Vertex AI."
        assert data["model"] == "gemini-1.5-flash"
        assert data["usage"]["total_tokens"] == 25
        assert data["finish_reason"] == "STOP"

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_streaming_with_mock(mock_vertex_service):
    """Test streaming chat with mocked Vertex AI service."""
    async def fake_stream(_request):
        chunks = ["Hello", " world", "!"]
        for chunk in chunks:
            yield chunk

    mock_vertex_service.generate_chat_stream = fake_stream
    app.dependency_overrides[get_vertex_service] = lambda: mock_vertex_service

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/chat",
            json={
                "message": "Tell me a story",
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body_text = response.text
        assert 'data: {"text": "Hello"}' in body_text
        assert 'data: {"text": " world"}' in body_text
        assert 'data: {"text": "!"}' in body_text
        assert "data: [DONE]" in body_text

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_vertex_service_build_helpers():
    """Test VertexGeminiService content and config builder methods."""
    from app.config import Settings
    from app.services.vertex_service import VertexGeminiService
    from app.schemas.chat import ChatMessage, ChatRequest

    settings = Settings(gcp_project_id="test-proj")
    service = VertexGeminiService(settings=settings)

    # Test _build_contents
    history = [
        ChatMessage(role="user", content="User 1"),
        ChatMessage(role="model", content="Model 1"),
        ChatMessage(role="system", content="System note"),
    ]
    contents = service._build_contents("Current prompt", history)
    assert len(contents) == 3  # 1 user + 1 model from history, + 1 current prompt
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "User 1"
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Model 1"
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "Current prompt"

    # Test _build_config with system instruction
    req = ChatRequest(
        message="Hi",
        history=history,
        system_instruction="Follow rules.",
        temperature=0.8,
        max_output_tokens=256,
    )
    config = service._build_config(req)
    assert config.temperature == 0.8
    assert config.max_output_tokens == 256
    assert "System note" in config.system_instruction
    assert "Follow rules." in config.system_instruction


@pytest.mark.asyncio
async def test_vertex_service_direct_generate():
    """Test VertexGeminiService.generate_chat directly with mocked client."""
    from app.config import Settings
    from app.services.vertex_service import VertexGeminiService
    from app.schemas.chat import ChatRequest

    mock_client = MagicMock()
    mock_gen_response = MagicMock()
    mock_gen_response.text = "Direct response"
    mock_gen_response.candidates = [MagicMock(finish_reason="STOP")]
    mock_gen_response.usage_metadata = MagicMock(
        prompt_token_count=12,
        candidates_token_count=8,
        total_token_count=20,
    )

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_gen_response)

    service = VertexGeminiService(
        settings=Settings(gcp_project_id="test-proj"),
        client=mock_client,
    )

    req = ChatRequest(message="Hello directly")
    res = await service.generate_chat(req)

    assert res.response == "Direct response"
    assert res.finish_reason == "STOP"
    assert res.usage.total_tokens == 20
