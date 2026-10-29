"""Authentication: username/password accounts and cookie sessions.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib, no new dependencies);
sessions are random tokens stored hashed in SQLite and carried in an
HttpOnly cookie. Every data endpoint resolves the caller through
`get_current_user`, so unauthenticated or forged requests never reach data.
"""

import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Response

from . import db

SESSION_COOKIE = "reporag_session"
SESSION_TTL_DAYS = 30
_PBKDF2_ITERATIONS = 200_000

# Registration is open by default; set ALLOW_SIGNUP=0 to freeze the user list
_ALLOW_SIGNUP = os.getenv("ALLOW_SIGNUP", "1") != "0"
# Cookie `secure` override; auto (default) uses secure for any non-localhost host
_COOKIE_SECURE_ENV = os.getenv("AUTH_COOKIE_SECURE", "")
# Cookie SameSite override. Default "lax" is right for same-origin apps; a
# frontend on another site (e.g. Vercel calling this backend) needs "none",
# otherwise browsers never store/send the session cookie cross-site.
_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax").strip().lower()
if _COOKIE_SAMESITE not in ("lax", "strict", "none"):
    _COOKIE_SAMESITE = "lax"

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,32}$")


class AuthError(ValueError):
    """Validation problem surfaced to the API layer as a 400."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
        ).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


def cookie_secure_for(request: Request) -> bool:
    """Secure cookies everywhere except plain-localhost development."""
    if _COOKIE_SECURE_ENV != "":
        return _COOKIE_SECURE_ENV != "0"
    host = (urlparse(f"//{request.headers.get('host', '')}", scheme="http").hostname
            or "").lower()
    return host not in ("localhost", "127.0.0.1", "::1")


def set_session_cookie(response: Response, token: str, request: Request) -> None:
    # SameSite=None is silently rejected by browsers unless the cookie is also
    # Secure, so force it on for that case.
    secure = cookie_secure_for(request) or _COOKIE_SAMESITE == "none"
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_TTL_DAYS * 24 * 3600,
        httponly=True,
        samesite=_COOKIE_SAMESITE,
        secure=secure,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=SESSION_COOKIE, path="/")


def register(username: str, password: str) -> dict:
    """Create an account and start a session; returns the user row."""
    if not _ALLOW_SIGNUP:
        raise AuthError("registration is disabled on this deployment")
    username = username.strip()
    if not _USERNAME_RE.match(username):
        raise AuthError("username must be 3-32 chars: letters, digits, . _ -")
    if len(password) < 8:
        raise AuthError("password must be at least 8 characters")
    if db.get_user_by_username(username) is not None:
        raise AuthError("username is already taken")
    user = db.create_user_row(username, _hash_password(password))
    return user


def login(username: str, password: str) -> dict:
    user = db.get_user_by_username(username.strip())
    # same work either way so response time does not leak account existence
    stored = user["pw_hash"] if user else _hash_password(password)
    if not user or not _verify_password(password, stored):
        raise AuthError("invalid username or password")
    return {"id": user["id"], "username": user["username"]}


def issue_session(response: Response, request: Request, user_id: int) -> None:
    token = secrets.token_urlsafe(32)
    expires = _now() + timedelta(days=SESSION_TTL_DAYS)
    db.insert_session(
        hashlib.sha256(token.encode()).hexdigest(), user_id, expires.isoformat(timespec="seconds")
    )
    set_session_cookie(response, token, request)


def _token_hash(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE)
    return hashlib.sha256(token.encode()).hexdigest() if token else None


def current_user(request: Request) -> dict | None:
    """Resolved user for the request's session cookie, or None."""
    token_hash = _token_hash(request)
    return db.get_session_user(token_hash) if token_hash else None


def logout(request: Request) -> None:
    token_hash = _token_hash(request)
    if token_hash:
        db.delete_session(token_hash)


def get_current_user(request: Request) -> dict:
    """FastAPI dependency: 401 unless the request carries a live session."""
    user = current_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="sign in to use the API")
    return user
