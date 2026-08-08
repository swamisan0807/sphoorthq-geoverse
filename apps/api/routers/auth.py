"""Self-service signup/login, plus Auth0 as a second, ready-to-deploy way in
(utils/auth/auth0.py) - neither replaces the other. Any user can create
their own local account - no invite or admin approval step, matching the
ask. Sessions are JWT bearer tokens (Authorization: Bearer <token>),
verified by get_current_user, which other routers depend on to require
login - Auth0 login ends at the exact same kind of token, so nothing else
in the app needs to know or care which way a given session started."""

import logging
import os
import secrets

import requests
from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from utils.auth import auth0, mailer, store, tokens

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger("geoverse.auth")

_AUTH0_STATE_COOKIE = "auth0_state"

# usernames double as email addresses in this platform (that's what the
# forgot-password flow emails a reset link to), so the pattern has to admit
# '@' - not just the bare handle characters a plain "username" implies.
_USERNAME_PATTERN = r"^[a-zA-Z0-9_.+-]+(@[a-zA-Z0-9_.-]+)?$"


def _web_base_url(request: Request) -> str:
    """Where the web UI's reset-password page lives, for the link mailed
    out. No hardcoded localhost default: apps/api/main.py serves the built
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
    # only populated when SMTP isn't configured (see utils/auth/mailer.py) -
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
    does exist, a real reset link (utils/auth/tokens.issue_reset_token,
    15 min TTL) is emailed via utils/auth/mailer. Without SMTP_HOST/
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


def _auth0_redirect_uri(request: Request) -> str:
    """Must exactly match a URL registered in the Auth0 application's
    "Allowed Callback URLs" - this backend route itself (it needs the client
    secret, see auth0_callback below), always on the API's own real origin
    (request.base_url) - deliberately NOT _web_base_url(request), which
    WEB_BASE_URL can override to point at the frontend's origin instead (the
    split dev setup, UI on :5173, API on :8000). Auth0 has to redirect back
    to wherever this backend route actually lives, regardless of where
    WEB_BASE_URL says the frontend is - conflating the two here was a real
    bug (a set WEB_BASE_URL sent Auth0 to the frontend's origin, which isn't
    registered as a callback URL and doesn't have this route anyway)."""
    return f"{str(request.base_url).rstrip('/')}/api/auth/auth0/callback"


@router.get("/auth0/login")
def auth0_login(request: Request):
    """Starts the Auth0 Authorization Code flow - redirects the browser to
    Auth0's Universal Login page. state is a random, single-use value bound
    to a short-lived cookie (not embedded in any predictable way) so the
    callback below can reject a forged/replayed redirect (CSRF on the login
    flow itself) - standard OAuth2 state-parameter practice."""
    if not auth0.is_configured():
        raise HTTPException(
            501, "Auth0 isn't configured on this server (AUTH0_DOMAIN/AUTH0_CLIENT_ID/AUTH0_CLIENT_SECRET)"
        )
    state = secrets.token_urlsafe(24)
    redirect = RedirectResponse(auth0.build_authorize_url(_auth0_redirect_uri(request), state))
    redirect.set_cookie(_AUTH0_STATE_COOKIE, state, max_age=600, httponly=True, samesite="lax")
    return redirect


@router.get("/auth0/callback")
def auth0_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
    auth0_state: str | None = Cookie(default=None, alias=_AUTH0_STATE_COOKIE),
):
    """Exchanges the authorization code for tokens (server-side, using the
    client secret - never exposed to the browser), fetches the real Auth0
    profile, provisions or finds the matching local user, and redirects to
    the frontend with this app's own session token - from that point on
    it's indistinguishable from a password login to every other route.

    Note: the state cookie is deleted on the RedirectResponse this function
    actually returns, not via an injected Response param - FastAPI only
    applies headers/cookies set on an injected Response when the route
    returns None and lets FastAPI build the response itself; here we
    explicitly return a RedirectResponse, so it has to carry its own
    delete_cookie() call (a real gotcha, not a style choice)."""
    if error is not None:
        raise HTTPException(400, f"Auth0 login failed: {error} - {error_description or ''}".strip(" -"))
    if not code or not state or not auth0_state or state != auth0_state:
        raise HTTPException(400, "invalid or expired Auth0 login attempt - please try logging in again")

    try:
        tokens_resp = auth0.exchange_code(code, _auth0_redirect_uri(request))
        profile = auth0.fetch_userinfo(tokens_resp["access_token"])
    except requests.RequestException as e:
        logger.warning("Auth0 code exchange/userinfo failed: %s", e)
        raise HTTPException(502, "couldn't complete Auth0 login - the Auth0 request itself failed") from e

    username = profile.get("email") or profile.get("sub")
    if not username:
        raise HTTPException(502, "Auth0 didn't return an email or subject id for this account")

    user = store.get_user(username)
    if user is None:
        store.create_user(username, secrets.token_urlsafe(32), auth_provider="auth0")
    elif user["auth_provider"] != "auth0":
        # Someone already registered this exact username/email with a password
        # (utils/auth/store.create_user, auth_provider="local") - refuse to let
        # an Auth0 login silently take over that account. Closes the obvious
        # pre-registration hijack: signing up locally with a victim's email
        # first, hoping they later log in via Auth0 and land on your account.
        raise HTTPException(
            409,
            f"'{username}' already has a password-based account on this platform - log in with that "
            "password instead of Auth0, or use a different Auth0 account.",
        )

    session_token = tokens.issue_token(username)
    dest = f"{_web_base_url(request)}/auth0/callback?token={session_token}&username={username}"
    redirect = RedirectResponse(dest)
    redirect.delete_cookie(_AUTH0_STATE_COOKIE)
    return redirect
