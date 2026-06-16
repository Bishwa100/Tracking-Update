"""API routers and shared dependencies."""

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings as app_settings

_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Validate the X-API-Key header against the configured key."""
    if not api_key or api_key != app_settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return api_key


async def verify_admin_api_key(api_key: str = Security(_api_key_header)) -> str:
    """Validate against ADMIN_API_KEY (falls back to API_KEY when admin key not set)."""
    expected = app_settings.ADMIN_API_KEY or app_settings.API_KEY
    if not api_key or api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid or missing admin API key.")
    return api_key
