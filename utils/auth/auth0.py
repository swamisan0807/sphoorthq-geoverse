"""Auth0 login (Authorization Code flow, confidential client) as an
additional way in, alongside the existing self-service username/password
flow in utils/auth/store.py - neither replaces the other.

Why this flow specifically: the app has a client secret (a "Regular Web
Application" in Auth0's terms), and the whole platform is already "one
process on one port" (apps/api/main.py serves the built React UI itself) -
so a server-side authorization-code exchange fits the existing architecture
better than a browser-side SPA SDK would, and lets the callback route just
mint this app's own normal session JWT (utils/auth/tokens.issue_token) the
same way password login does. Every other router's Depends(get_current_user)
needs zero changes: Auth0 is just another way to end up with that JWT.

Credentials: AUTH0_DOMAIN / AUTH0_CLIENT_ID / AUTH0_CLIENT_SECRET env vars,
nothing hardcoded (see config/platform.yaml). Without them set, is_configured()
is False and the /api/auth/auth0/* routes return a clear 501 instead of a
confusing failure deep in an HTTP call.
"""

import os

import requests

_TIMEOUT_S = 10


def _domain() -> str | None:
    return os.environ.get("AUTH0_DOMAIN")


def is_configured() -> bool:
    return bool(_domain() and os.environ.get("AUTH0_CLIENT_ID") and os.environ.get("AUTH0_CLIENT_SECRET"))


def build_authorize_url(redirect_uri: str, state: str) -> str:
    """The URL to send the browser to for Auth0's Universal Login page."""
    params = {
        "client_id": os.environ["AUTH0_CLIENT_ID"],
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": "openid profile email",
        "state": state,
    }
    query = "&".join(f"{k}={requests.utils.quote(v, safe='')}" for k, v in params.items())
    return f"https://{_domain()}/authorize?{query}"


def exchange_code(code: str, redirect_uri: str) -> dict:
    """Server-side code-for-tokens exchange - the client secret never
    touches the browser, only this backend call."""
    resp = requests.post(
        f"https://{_domain()}/oauth/token",
        json={
            "grant_type": "authorization_code",
            "client_id": os.environ["AUTH0_CLIENT_ID"],
            "client_secret": os.environ["AUTH0_CLIENT_SECRET"],
            "code": code,
            "redirect_uri": redirect_uri,
        },
        timeout=_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_userinfo(access_token: str) -> dict:
    """Real profile fields (sub, email, email_verified, name, ...) for the
    just-authenticated Auth0 user - https://{domain}/userinfo, standard OIDC."""
    resp = requests.get(
        f"https://{_domain()}/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()
