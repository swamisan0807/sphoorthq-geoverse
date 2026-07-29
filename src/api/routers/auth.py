"""Self-service signup/login. Any user can create their own account - no
invite or admin approval step, matching the ask. Sessions are JWT bearer
tokens (Authorization: Bearer <token>), verified by get_current_user, which
other routers depend on to require login."""

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from src.auth import store, tokens

router = APIRouter(prefix="/api/auth", tags=["auth"])


class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_.-]+$")
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
