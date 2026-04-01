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


class RedirectException(Exception):
    def __init__(self, url: str):
        self.url = url


async def require_login(request: Request) -> str:
    user = request.session.get("user")
    if not user:
        raise RedirectException(f"/login?next={request.url.path}")
    return user
