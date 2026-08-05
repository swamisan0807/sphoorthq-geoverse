"""Self-service signup/login. Any user can create their own account - no
invite or admin approval step, matching the ask. Sessions are JWT bearer
tokens (Authorization: Bearer <token>), verified by get_current_user, which
other routers depend on to require login."""

import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.auth import mailer, store, tokens

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger("geoverse.auth")

# usernames double as email addresses in this platform (that's what the
# forgot-password flow emails a reset link to), so the pattern has to admit
# '@' - not just the bare handle characters a plain "username" implies.
_USERNAME_PATTERN = r"^[a-zA-Z0-9_.+-]+(@[a-zA-Z0-9_.-]+)?$"


def _web_base_url(request: Request) -> str:
    """Where the web UI's reset-password page lives, for the link mailed
    out. No hardcoded localhost default: src/api/main.py serves the built
    UI itself (same origin as the API), so request.base_url is already
    correct for wherever this happens to be deployed - dev machine, a
    real domain, whatever. WEB_BASE_URL only needs setting for the split
    deployment case (UI hosted separately from the API)."""
    return os.environ.get("WEB_BASE_URL") or str(request.base_url).rstrip("/")


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=254, pattern=_USERNAME_PATTERN)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class AuthResponse(BaseModel):
    token: str
    username: str


class UserInfo(BaseModel):
    username: str
    created_at: float


class ForgotPasswordRequest(BaseModel):
    username: str


class ForgotPasswordResponse(BaseModel):
    detail: str
    # only populated when SMTP isn't configured (see src/auth/mailer.py) -
    # lets the flow be tested end-to-end without real email credentials.
    # Never set when an email was actually sent.
    dev_reset_link: str | None = None


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


def get_current_user(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    username = tokens.decode_token(token)
    if username is None:
        raise HTTPException(401, "invalid or expired session token")
    return username


@router.post("/signup", response_model=AuthResponse)
def signup(req: SignupRequest):
    try:
        store.create_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(409, str(e))
    token = tokens.issue_token(req.username)
    return AuthResponse(token=token, username=req.username)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    user = store.verify_user(req.username, req.password)
    if user is None:
        raise HTTPException(401, "invalid username or password")
    token = tokens.issue_token(req.username)
    return AuthResponse(token=token, username=req.username)


@router.get("/me", response_model=UserInfo)
def me(username: str = Depends(get_current_user)):
    user = store.get_user(username)
    if user is None:
        raise HTTPException(404, "user not found")
    return UserInfo(username=user["username"], created_at=user["created_at"])


@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(req: ForgotPasswordRequest, request: Request):
    """Always returns the same generic message regardless of whether the
    account exists - doesn't leak which usernames are registered. If it
    does exist, a real reset link (src/auth/tokens.issue_reset_token,
    15 min TTL) is emailed via src/auth/mailer. Without SMTP_HOST/
    SMTP_USER/SMTP_PASSWORD configured, mailer falls back to writing the
    message to datasets/metadata/outbox/ and this response includes the
    link directly so the flow is still testable locally."""
    generic_detail = "if an account with that username exists, a password reset link has been sent"
    user = store.get_user(req.username)
    if user is None:
        logger.info("forgot-password requested for unknown username %r", req.username)
        return ForgotPasswordResponse(detail=generic_detail)

    reset_token = tokens.issue_reset_token(req.username, user["token_version"])
    reset_link = f"{_web_base_url(request)}/reset-password?token={reset_token}"
    result = mailer.send_password_reset_email(req.username, reset_link)
    return ForgotPasswordResponse(detail=generic_detail, dev_reset_link=None if result.sent else reset_link)


@router.post("/reset-password", response_model=AuthResponse)
def reset_password(req: ResetPasswordRequest):
    decoded = tokens.decode_reset_token(req.token)
    if decoded is None:
        raise HTTPException(400, "invalid or expired reset link - request a new one")
    username, token_version = decoded
    user = store.get_user(username)
    if user is None or user["token_version"] != token_version:
        # version mismatch = this link was already used, or superseded by
        # a newer forgot-password request - either way it's dead
        raise HTTPException(400, "invalid or expired reset link - request a new one")
    store.set_password(username, req.new_password)
    logger.info("password reset for %s", username)
    token = tokens.issue_token(username)
    return AuthResponse(token=token, username=username)
