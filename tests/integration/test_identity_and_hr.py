"""Sprint 2: no anonymous fallback, is_hr on /profile, read-only HR employee endpoints."""

import httpx
import pytest

from app.core.config import Settings
from app.main import app
from app.repositories.firestore import FirestoreRepository, get_firestore_repository
from app.services.onboarding import OnboardingService, get_onboarding_service

HR = {"X-User-Id": "hr_admin_001"}
NAUFAL = {"X-User-Id": "user_001"}
AFIF = {"X-User-Id": "user_002"}


@pytest.fixture
def client_factory():
    firestore = FirestoreRepository(settings=Settings())
    app.dependency_overrides[get_firestore_repository] = lambda: firestore
    app.dependency_overrides[get_onboarding_service] = lambda: OnboardingService(firestore_repo=firestore)
    yield lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
    app.dependency_overrides.clear()


async def _create(client, headers, name, role="Backend Engineer", department="Engineering"):
    res = await client.post("/api/v1/profile", headers=headers, json={
        "name": name, "role": role, "department": department, "location": "Jakarta"})
    assert res.status_code == 201, res.text


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", [
    ("GET", "/api/v1/profile"),
    ("GET", "/api/v1/tasks"),
    ("GET", "/api/v1/onboarding/progress"),
    ("GET", "/api/v1/documents"),
    ("POST", "/api/v1/chat"),
    ("GET", "/api/v1/admin/employees"),
])
async def test_missing_user_id_is_401_not_user_001(client_factory, method, path):
    async with client_factory() as client:
        await _create(client, NAUFAL, "Naufal")
        for headers in ({}, {"X-User-Id": "  "}, {"X-Access-Level": "HR_ADMIN"}):
            res = await client.request(method, path, headers=headers, json={"message": "my tasks"})
            assert res.status_code == 401, (headers, res.text)
            assert res.json()["error"]["code"] == "UNAUTHENTICATED"


@pytest.mark.asyncio
async def test_profile_reports_is_hr(client_factory):
    async with client_factory() as client:
        await _create(client, NAUFAL, "Naufal")

        me = (await client.get("/api/v1/profile", headers=NAUFAL)).json()
        assert me["name"] == "Naufal" and me["is_hr"] is False and me["access_level"] == "EMPLOYEE"

        # A spoofed access level does not make an employee HR.
        me = (await client.get("/api/v1/profile", headers={**NAUFAL, "X-Access-Level": "HR_ADMIN"})).json()
        assert me["is_hr"] is False

        # HR has no stored profile but still gets one, flagged as HR.
        res = await client.get("/api/v1/profile", headers=HR)
        assert res.status_code == 200, res.text
        hr = res.json()
        assert hr["user_id"] == "hr_admin_001" and hr["is_hr"] is True and hr["access_level"] == "HR_ADMIN"

        # Employees without a profile still get 404.
        assert (await client.get("/api/v1/profile", headers=AFIF)).status_code == 404


@pytest.mark.asyncio
async def test_hr_lists_employees_with_progress(client_factory):
    async with client_factory() as client:
        await _create(client, NAUFAL, "Naufal")
        await _create(client, AFIF, "Afif", role="Product Designer", department="Design")
        tasks = (await client.get("/api/v1/tasks", headers=AFIF)).json()
        await client.post(f"/api/v1/tasks/{tasks[0]['id']}/complete", headers=AFIF)

        res = await client.get("/api/v1/admin/employees", headers=HR)
        assert res.status_code == 200, res.text
        employees = res.json()
        assert [e["name"] for e in employees] == ["Afif", "Naufal"]
        afif = employees[0]
        assert afif == {
            "user_id": "user_002", "name": "Afif", "role": "Product Designer", "department": "Design",
            "location": "Jakarta", "progress_percent": round(100 / 7, 2), "completed": 1, "total": 7,
        }
        assert employees[1]["completed"] == 0


@pytest.mark.asyncio
async def test_hr_reads_employee_tasks_and_progress_read_only(client_factory):
    async with client_factory() as client:
        await _create(client, AFIF, "Afif")
        own = (await client.get("/api/v1/tasks", headers=AFIF)).json()

        res = await client.get("/api/v1/admin/employees/user_002/tasks", headers=HR)
        assert res.status_code == 200
        assert [t["id"] for t in res.json()] == [t["id"] for t in own]

        progress = (await client.get("/api/v1/admin/employees/user_002/progress", headers=HR)).json()
        assert progress == {"total": 7, "completed": 0, "remaining": 7, "progress_percent": 0.0}

        assert (await client.get("/api/v1/admin/employees/nobody/tasks", headers=HR)).status_code == 404

        # HR cannot complete an employee's task.
        res = await client.post(f"/api/v1/tasks/{own[0]['id']}/complete", headers=HR)
        assert res.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "/api/v1/admin/employees",
    "/api/v1/admin/employees/user_001/tasks",
    "/api/v1/admin/employees/user_001/progress",
])
async def test_employees_cannot_use_hr_endpoints(client_factory, path):
    async with client_factory() as client:
        await _create(client, NAUFAL, "Naufal")
        for headers in (AFIF, {**AFIF, "X-Access-Level": "HR_ADMIN"}):
            res = await client.get(path, headers=headers)
            assert res.status_code == 403
            assert res.json()["error"]["code"] == "UNAUTHORIZED"
