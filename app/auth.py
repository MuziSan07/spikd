import hashlib
from fastapi import Request
from fastapi.responses import RedirectResponse
from config import settings


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_credentials(username: str, password: str) -> bool:
    users = settings.USERS
    stored = users.get(username)
    if not stored:
        return False
    return stored == hash_password(password)


class require_login:
    """FastAPI dependency — redirects unauthenticated requests to /login."""
    def __init__(self, request: Request):
        self.request = request

    def __await__(self):
        return self._check().__await__()

    async def _check(self):
        user = self.request.session.get("user")
        if not user:
            raise _LoginRedirect()
        return user


class _LoginRedirect(Exception):
    pass


from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware


# Simpler dependency approach
async def require_login(request: Request) -> str:
    user = request.session.get("user")
    if not user:
        from fastapi import HTTPException
        # For API endpoints return 401, for page endpoints redirect
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
