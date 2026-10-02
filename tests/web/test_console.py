"""The operator console: every page and partial, the trial flow, and no external requests."""

import hashlib
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fieldtrial.web.app import STATIC, create_app

STUDY = "/studies/alpha"


class Console:
    """Drives the console like a browser with HTMX would."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        client.get("/")  # receive the CSRF cookie
        self.headers = {
            "X-CSRF-Token": client.cookies.get("fieldtrial_csrf") or "",
            "HX-Request": "true",
        }

    def post(self, url: str, data: dict[str, object] | None = None) -> str:
        r = self.client.post(url, data=data or {}, headers=self.headers, follow_redirects=False)
        assert r.status_code == 200, r.text
        return r.headers.get("hx-redirect") or r.text

    def session(self, study: str = STUDY) -> str:
        return self.post(
            f"{study}/sessions",
            {
                "operator": "ana",
                "rig": "rig-1",
                "check_0": "on",
                "check_1": "on",
                "check_2": "on",
                "idempotency_key": "s-1",
            },
        )


def field(html: str, name: str) -> str:
    match = re.search(rf'name="{name}" value="([^"]*)"', html)
    assert match, f"no {name} in page"
    return match.group(1)


def trial_id(html: str) -> str:
    match = re.search(r"/trials/([0-9a-f-]{36})/", html)
    assert match
    return match.group(1)


def run_trial(console: Console, url: str, stage: str = "3", termination: str = "success") -> str:
    page = console.post(f"{url}/start", {"slot_id": field(console.client.get(url).text, "slot_id")})
    tid = trial_id(page)
    page = console.post(f"{url}/trials/{tid}/stop", {"version": field(page, "version")})
    console.post(
        f"{url}/trials/{tid}/complete",
        {"stage": stage, "termination": termination, "version": field(page, "version")},
    )
    return tid


# --- pages -------------------------------------------------------------------------------------


def test_studies_page(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    for slug in ("alpha", "open", "draft"):
        assert f'href="/studies/{slug}"' in r.text


def test_single_study_redirects(studies: Path) -> None:
    with TestClient(create_app(studies / "alpha"), base_url="http://127.0.0.1") as c:
        r = c.get("/", follow_redirects=False)
        assert r.headers["location"] == "/studies/alpha"


def test_empty_folder(tmp_path: Path) -> None:
    with TestClient(create_app(tmp_path), base_url="http://127.0.0.1") as c:
        assert "No studies here yet" in c.get("/").text


def test_draft_and_unknown_studies(client: TestClient, studies: Path) -> None:
    draft = client.get("/studies/draft")
    assert "This study is a draft" in draft.text
    (studies / "draft" / "study.yaml").write_text("broken: [")
    assert "has problems" in client.get("/studies/draft").text
    missing = client.get("/studies/nope")
    assert missing.status_code == 404
    assert "no study" in missing.text
    assert "<html" in missing.text


def test_study_page_blinded(client: TestClient) -> None:
    page = client.get(STUDY).text
    assert "blinded" in page
    assert "baseline" not in page
    assert "q50" not in page
    assert "Gripper pads clean" in page
    assert "0</strong> of 8 trials done" in page


def test_study_page_open(client: TestClient) -> None:
    page = client.get("/studies/open").text
    assert "baseline" in page
    assert "queue 30" in page


def test_rig_checklist_is_required(client: TestClient) -> None:
    console = Console(client)
    r = client.post(
        f"{STUDY}/sessions",
        data={"operator": "ana", "rig": "r", "check_0": "on"},
        headers=console.headers,
    )
    assert r.status_code == 400
    assert r.headers["hx-retarget"] == "#flash"
    assert "tick every rig check" in r.text


# --- the trial console -----------------------------------------------------------------------


def test_trial_flow_through_the_console(client: TestClient) -> None:
    console = Console(client)
    url = console.session()
    assert url.startswith(f"{STUDY}/sessions/")

    page = client.get(url).text
    assert 'data-action="start_stop"' in page
    assert "Trial 1/8" in page
    assert "Reset the scene" in page
    assert "baseline" not in page  # blinded

    started = console.post(
        f"{url}/start", {"slot_id": field(page, "slot_id"), "idempotency_key": "k"}
    )
    again = console.post(
        f"{url}/start", {"slot_id": field(page, "slot_id"), "idempotency_key": "k"}
    )
    assert trial_id(started) == trial_id(again)
    assert 'class="timer"' in started
    assert 'data-timeout="45.0"' in started

    tid = trial_id(started)
    stopped = console.post(f"{url}/trials/{tid}/stop", {"version": field(started, "version")})
    assert "Label the outcome" in stopped
    assert 'name="stage" value="-1"' in stopped
    assert 'data-success-index="3"' in stopped

    done = console.post(
        f"{url}/trials/{tid}/complete",
        {
            "stage": "1",
            "termination": "stuck",
            "failure_tags": ["dropped", "collision"],
            "notes": "slipped at handover",
            "version": field(stopped, "version"),
        },
    )
    assert "Undo trial 1" in done
    assert "Trial 2/8" in done

    undone = console.post(f"{url}/trials/{tid}/undo")
    assert "Label the outcome" in undone  # stopped time kept, label cleared
    relabelled = console.post(
        f"{url}/trials/{tid}/complete",
        {"stage": "3", "termination": "success", "version": field(undone, "version")},
    )
    assert "Trial 2/8" in relabelled

    second = console.post(f"{url}/start", {"slot_id": field(relabelled, "slot_id")})
    invalid = console.post(
        f"{url}/trials/{trial_id(second)}/invalidate",
        {"reason": "gripper fault", "version": field(second, "version")},
    )
    assert "Undo trial" in invalid

    stale = client.post(
        f"{url}/trials/{tid}/complete",
        data={"stage": "3", "termination": "success", "version": "1"},
        headers=console.headers,
    )
    assert stale.status_code in (400, 409)
    assert stale.headers["hx-retarget"] == "#flash"

    missing = client.post(
        f"{url}/trials/{tid}/complete", data={"stage": "3"}, headers=console.headers
    )
    assert missing.status_code == 422
    assert "Please fill in: termination, version." in missing.text

    assert console.post(f"{url}/end") == STUDY
    ended = client.post(f"{url}/start", data={"slot_id": "x"}, headers=console.headers)
    assert ended.status_code == 400


def test_mirror_and_panel(client: TestClient) -> None:
    console = Console(client)
    url = console.session()
    mirror = client.get(f"{url}/mirror").text
    assert 'class="mirror"' in mirror
    assert "Waiting to start" in mirror
    assert 'data-action="start_stop"' not in mirror
    assert "?mirror=true" in mirror
    console.post(f"{url}/start", {"slot_id": field(client.get(url).text, "slot_id")})
    live = client.get(f"{url}/panel", params={"mirror": True}).text
    assert 'class="timer"' in live
    assert "<form" not in live
    assert "<html" not in client.get(f"{url}/panel").text


def test_finished_study_points_to_report(client: TestClient) -> None:
    console = Console(client)
    url = console.session()
    for _ in range(8):
        run_trial(console, url)
    page = client.get(url).text
    assert "All scheduled trials are done" in page
    assert f'href="{STUDY}/report"' in page
    assert "Start a session" not in client.get(STUDY).text


def test_sim_arms_suggest_an_outcome(tmp_path: Path) -> None:
    from tests.services.conftest import write_small_study

    from fieldtrial.services.study import lock_study

    folder = write_small_study(tmp_path / "sim")
    path = folder / "study.yaml"
    path.write_text(
        path.read_text()
        .replace("runner: manual", "runner: sim")
        .replace("policy: {path: ", "policy: {sim_success_rate: 1.0, path: ")
    )
    lock_study(folder)
    with TestClient(create_app(folder), base_url="http://127.0.0.1") as c:
        console = Console(c)
        url = console.session("/studies/sim")
        page = console.post(f"{url}/start", {"slot_id": field(c.get(url).text, "slot_id")})
        page = console.post(
            f"{url}/trials/{trial_id(page)}/stop", {"version": field(page, "version")}
        )
        assert "simulated runner suggests" in page
        assert re.search(r'value="3" data-key="4"\s+checked', page)
        assert re.search(r'value="success" required\s+checked', page)


# --- history and edits -------------------------------------------------------------------------


def test_history_and_edit(client: TestClient) -> None:
    console = Console(client)
    url = console.session()
    tid = run_trial(console, url)
    history = client.get(f"{STUDY}/history").text
    assert f'id="trial-{tid}"' in history
    assert "✓" in history
    form = client.get(f"{STUDY}/trials/{tid}/edit").text
    assert "Reason for the correction" in form
    row = console.post(
        f"{STUDY}/trials/{tid}/edit",
        {
            "actor": "bo",
            "reason": "video shows a tilt",
            "stage": "2",
            "termination": "stuck",
            "notes": "tilted",
            "version": field(form, "version"),
        },
    )
    assert "✗" in row
    assert "inserted" in row
    assert client.get(f"{STUDY}/trials/{tid}/row").text.count("<tr") == 1
    assert "No trials yet" in client.get("/studies/open/history").text


# --- report ------------------------------------------------------------------------------------


def test_report_unblinding(client: TestClient) -> None:
    console = Console(client)
    url = console.session()
    run_trial(console, url)
    blinded = client.get(f"{STUDY}/report").text
    assert "Results stay hidden" in blinded
    assert "7 trials are still pending" in blinded
    refused = client.post(f"{STUDY}/unblind", data={}, headers=console.headers)
    assert refused.status_code == 400
    assert console.post(f"{STUDY}/unblind", {"confirm": "yes"}) == f"{STUDY}/report"
    report = client.get(f"{STUDY}/report").text
    assert "early unblinding" in report
    assert "Success rate per arm" in report
    md = client.get(f"{STUDY}/report.md")
    assert md.headers["content-type"].startswith("text/markdown")
    assert "attachment" in md.headers["content-disposition"]
    assert md.text.startswith("# ")
    results = client.get(f"{STUDY}/results.json")
    assert results.json()["schema_version"] == 1


# --- security ----------------------------------------------------------------------------------


def test_console_writes_need_csrf(client: TestClient) -> None:
    client.get("/")
    r = client.post(f"{STUDY}/sessions", data={"operator": "a", "rig": "r"})
    assert r.status_code == 403
    r = client.post(
        f"{STUDY}/sessions", data={"operator": "a", "rig": "r"}, headers={"X-CSRF-Token": "wrong"}
    )
    assert r.status_code == 403


def test_lan_console_flow(studies: Path) -> None:
    with TestClient(create_app(studies, lan_token="tok"), base_url="http://10.0.0.5:8765") as c:
        assert c.get("/").status_code == 401
        r = c.get("/?token=tok")  # follows the redirect with the cookie set
        assert r.status_code == 200
        assert "LAN" in r.text
        console = Console(c)
        assert console.session().startswith(f"{STUDY}/sessions/")


# --- no external requests ----------------------------------------------------------------------


class _Links(HTMLParser):
    ATTRS = frozenset(
        {"href", "src", "action", "hx-get", "hx-post", "data-events", "poster", "srcset"}
    )

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.inline_scripts = 0
        self.inline_styles = 0
        self._in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        self.urls += [v for k, v in attrs if k in self.ATTRS and v]
        if tag == "script":
            self._in_script = "src" not in values
        if tag == "style" or "style" in values or any(k.startswith("on") for k in values):
            self.inline_styles += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script and data.strip():
            self.inline_scripts += 1


def _pages(client: TestClient) -> list[str]:
    """Every page and partial, each fetched while it shows the state named beside it."""
    console = Console(client)
    url = console.session()
    tid = run_trial(console, url)
    pages = ["/", STUDY, "/studies/draft", "/studies/open", url, f"{url}/mirror", f"{url}/panel"]
    pages += [f"{STUDY}/history", f"{STUDY}/trials/{tid}/edit", f"{STUDY}/report", "/studies/nope"]
    html = [client.get(page).text for page in pages]
    started = console.post(f"{url}/start", {"slot_id": field(client.get(url).text, "slot_id")})
    html.append(client.get(f"{url}/panel").text)  # running: timer and Stop
    assert 'class="timer"' in html[-1]
    console.post(f"{url}/trials/{trial_id(started)}/stop", {"version": field(started, "version")})
    html.append(client.get(f"{url}/panel").text)  # stopped: the label form
    assert "Label the outcome" in html[-1]
    console.post(f"{STUDY}/unblind", {"confirm": "yes"})
    html.append(client.get(f"{STUDY}/report").text)  # unblinded results
    assert "Success rate per arm" in html[-1]
    return html


def test_pages_make_no_external_requests(client: TestClient) -> None:
    for html in _pages(client):
        parser = _Links()
        parser.feed(html)
        assert parser.inline_scripts == 0
        assert parser.inline_styles == 0
        for url in parser.urls:
            assert not re.match(r"^[a-z][a-z0-9+.-]*:|^//", url, re.IGNORECASE), url
            if url.startswith("/static/"):
                assert client.get(url).status_code == 200, url
        assert "http://" not in html.replace('xmlns="http://www.w3.org', "")
        assert "https://" not in html


def test_static_assets_have_no_external_urls() -> None:
    for path in STATIC.rglob("*"):
        if path.suffix in (".js", ".css", ".html"):
            text = path.read_text(encoding="utf-8")
            assert not re.search(r"https?://", text), path


def test_vendored_htmx_matches_its_recorded_hash() -> None:
    readme = (STATIC / "vendor" / "README.md").read_text(encoding="utf-8")
    for match in re.finditer(r"`([\w.-]+\.js)` \| [^|]+\|[^|]+\|[^|]+\| `([0-9a-f]{64})`", readme):
        name, digest = match.groups()
        assert hashlib.sha256((STATIC / "vendor" / name).read_bytes()).hexdigest() == digest
    assert "htmx-2.0.11.min.js" in readme


@pytest.mark.parametrize("path", ["/static/app.css", "/static/app.js"])
def test_static_files_are_served(client: TestClient, path: str) -> None:
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["x-content-type-options"] == "nosniff"


def test_redirects_for_htmx_and_plain_forms(client: TestClient) -> None:
    console = Console(client)
    data = {"operator": "a", "rig": "r", "check_0": "on", "check_1": "on", "check_2": "on"}
    htmx = client.post(f"{STUDY}/sessions", data=data, headers=console.headers)
    assert htmx.status_code == 200  # an XHR would silently follow a 303
    assert htmx.headers["hx-redirect"].startswith(f"{STUDY}/sessions/")
    plain = client.post(
        f"{STUDY}/sessions",
        data=data,
        headers={"X-CSRF-Token": console.headers["X-CSRF-Token"]},
        follow_redirects=False,
    )
    assert plain.status_code == 303
    assert plain.headers["location"].startswith(f"{STUDY}/sessions/")
