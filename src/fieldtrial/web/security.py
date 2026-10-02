"""Request checks for the console and API (docs/PLAN.md section 14).

- **Local mode** (127.0.0.1): only requests whose ``Host`` is a loopback name are served,
  which blocks DNS-rebinding attacks from web pages.
- **LAN mode**: every request needs the session token, given once as ``?token=`` (it is
  then kept in an HttpOnly cookie and removed from the address bar) or as
  ``Authorization: Bearer <token>``.
- **CSRF**: every console write carries ``X-CSRF-Token`` matching the ``fieldtrial_csrf``
  cookie. API writes carry an ``X-Fieldtrial-Client`` header instead; browsers cannot add
  custom headers cross-site without a CORS preflight, which this server never allows.
- Every response gets a strict Content-Security-Policy: scripts, styles and connections
  only from this server, no inline scripts, no ``eval``.
"""

import secrets
from collections.abc import Iterable
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlencode

from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

TOKEN_COOKIE = "fieldtrial_token"
CSRF_COOKIE = "fieldtrial_csrf"
CSRF_HEADER = "x-csrf-token"
CLIENT_HEADER = "x-fieldtrial-client"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "media-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "content-security-policy": CSP,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
}


def _headers(scope: Scope) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}


def _cookies(headers: dict[str, str]) -> dict[str, str]:
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(headers.get("cookie", ""))
    except Exception:  # malformed cookie header: treat as none
        return {}
    return {k: v.value for k, v in jar.items()}


def _hostname(host: str) -> str:
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _equal(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and secrets.compare_digest(str(a), str(b))


class SecurityMiddleware:
    """Host, token and CSRF checks, plus security headers on every response."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        lan_token: str | None = None,
        allowed_hosts: Iterable[str] | None = None,
    ) -> None:
        self.app = app
        self.lan_token = lan_token
        self.allowed_hosts = None if lan_token else frozenset(allowed_hosts or LOOPBACK_HOSTS)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check the request, then add security headers to the response."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = _headers(scope)
        cookies = _cookies(headers)
        method = scope["method"]
        path: str = scope["path"]

        problem = self._check(scope, headers, cookies, method, path)
        if isinstance(problem, Response):
            for name, value in SECURITY_HEADERS.items():
                problem.headers.setdefault(name, value)
            await problem(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    out.setdefault(name, value)
                if CSRF_COOKIE not in cookies:
                    out.append("set-cookie", self._csrf_cookie(scope))
            await send(message)

        await self.app(scope, receive, send_wrapper)

    def _check(
        self,
        scope: Scope,
        headers: dict[str, str],
        cookies: dict[str, str],
        method: str,
        path: str,
    ) -> Response | None:
        if self.allowed_hosts is not None:
            host = _hostname(headers.get("host", ""))
            if host not in self.allowed_hosts:
                return PlainTextResponse("Forbidden host. Open the console via 127.0.0.1.", 403)
        if self.lan_token is not None:
            query = dict(parse_qsl(scope.get("query_string", b"").decode("latin-1")))
            bearer = headers.get("authorization", "")
            given = bearer[7:] if bearer.lower().startswith("bearer ") else None
            if method in SAFE_METHODS and _equal(query.get("token"), self.lan_token):
                # Keep the token out of the address bar, history and logs.
                rest = urlencode([(k, v) for k, v in query.items() if k != "token"])
                response = RedirectResponse(path + (f"?{rest}" if rest else ""), 303)
                response.set_cookie(
                    TOKEN_COOKIE, self.lan_token, httponly=True, samesite="strict", path="/"
                )
                return response
            if not (
                _equal(given, self.lan_token) or _equal(cookies.get(TOKEN_COOKIE), self.lan_token)
            ):
                return PlainTextResponse(
                    "This console needs its access token: scan the QR code or open the URL "
                    "that `fieldtrial serve --lan` printed.",
                    401,
                )
        if method not in SAFE_METHODS:
            if path.startswith("/api/"):
                if CLIENT_HEADER not in headers and not _equal(
                    headers.get(CSRF_HEADER), cookies.get(CSRF_COOKIE)
                ):
                    return PlainTextResponse(
                        f"API writes need an {CLIENT_HEADER} header (fieldtrial.client sends it).",
                        403,
                    )
            elif not _equal(headers.get(CSRF_HEADER), cookies.get(CSRF_COOKIE)):
                return PlainTextResponse("Missing or wrong CSRF token. Reload the page.", 403)
        return None

    @staticmethod
    def _csrf_cookie(scope: Scope) -> str:
        token = scope.setdefault("state", {}).get("csrf_token") or secrets.token_urlsafe(32)
        return f"{CSRF_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"


def csrf_token(scope: Scope, cookies: dict[str, str]) -> str:
    """The CSRF token for this browser: its cookie, or a new one set on this response."""
    existing = cookies.get(CSRF_COOKIE)
    if existing:
        return existing
    state = scope.setdefault("state", {})
    if "csrf_token" not in state:
        state["csrf_token"] = secrets.token_urlsafe(32)
    token: str = state["csrf_token"]
    return token
