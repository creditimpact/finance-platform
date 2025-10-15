import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server


pytestmark = pytest.mark.slow


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, *, timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            request = Request(url)
            request.add_header("Accept", "application/json, text/html")
            with urlopen(request, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (URLError, HTTPError, ConnectionError):
            time.sleep(0.5)
            continue
        except OSError:
            time.sleep(0.5)
            continue
    raise TimeoutError(f"Timed out waiting for {url}")


def _fetch_json(url: str) -> tuple[int, dict]:
    request = Request(url)
    request.add_header("Accept", "application/json")
    with urlopen(request, timeout=10) as response:
        payload = json.load(response)
        return response.status, payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def _temporary_env(updates: dict[str, str]) -> Iterator[None]:
    original: dict[str, str | None] = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _run_api_server(runs_root: Path, port: int) -> Iterator[str]:
    env_updates = {
        "RUNS_ROOT": str(runs_root),
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "test-key"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "CELERY_BROKER_URL": os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        "SECRET_KEY": os.environ.get("SECRET_KEY", "dev-secret"),
    }
    with _temporary_env(env_updates):
        api_app = importlib.import_module("backend.api.app")
        api_app = importlib.reload(api_app)
        app = api_app.create_app()
        server = make_server("127.0.0.1", port, app)
        ctx = app.app_context()
        ctx.push()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            _wait_for_http(f"http://127.0.0.1:{port}/")
            yield f"http://127.0.0.1:{port}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            ctx.pop()


@contextmanager
def _run_ui_server(frontend_dir: Path, port: int, env: dict[str, str]) -> Iterator[str]:
    command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(frontend_dir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/")
        yield f"http://127.0.0.1:{port}"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.fixture(scope="session")
def ensure_playwright() -> None:
    browsers_dir_env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if browsers_dir_env:
        browsers_dir = Path(browsers_dir_env)
    else:
        browsers_dir = Path.home() / ".cache" / "ms-playwright"

    chromium_already_installed = any(browsers_dir.glob("chromium-*"))
    if chromium_already_installed:
        return

    try:
        subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover - environment specific
        pytest.skip(f"Playwright Chromium install failed: {exc.stderr.decode(errors='ignore')[:200]}")


@pytest.fixture(scope="session")
def ensure_frontend_dependencies() -> Path:
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    if os.environ.get("CI"):
        return frontend_dir
    node_modules = frontend_dir / "node_modules"
    has_modules = node_modules.is_dir() and any(node_modules.iterdir())
    if not has_modules:
        subprocess.run(
            ["npm", "ci", "--no-progress"],
            cwd=str(frontend_dir),
            check=True,
        )
    return frontend_dir


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is required for the frontend dev server")
def test_frontend_review_pipeline_end_to_end(tmp_path: Path, ensure_playwright: None, ensure_frontend_dependencies: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "S-e2e-frontend"
    run_dir = runs_root / sid
    packs_dir = run_dir / "frontend" / "review" / "packs"
    responses_dir = run_dir / "frontend" / "review" / "responses"
    responses_dir.mkdir(parents=True, exist_ok=True)

    accounts = [
        ("idx-001", "Alice Example", "incorrect_information"),
        ("idx-002", "Brian Sample", "identity_theft"),
        ("idx-003", "Carol Client", "balance_inaccurate"),
    ]

    manifest_entries: list[dict] = []

    for index, (account_id, holder_name, primary_issue) in enumerate(accounts, start=1):
        suffix = f"{1200 + index * 100}"
        pack_payload = {
            "account_id": account_id,
            "holder_name": holder_name,
            "primary_issue": primary_issue,
            "questions": [
                {"id": "ownership", "prompt": "Do you own this account?"},
                {"id": "recognize", "prompt": "Do you recognize this account?"},
            ],
            "display": {
                "holder_name": holder_name,
                "primary_issue": primary_issue,
                "account_number": {
                    "per_bureau": {"EX": f"****{index}111", "EQ": f"****{index}222", "TU": f"****{index}333"}
                },
                "account_type": {
                    "per_bureau": {"EX": "Credit Card", "EQ": "Credit Card", "TU": "Credit Card"}
                },
                "status": {
                    "per_bureau": {"EX": "Open", "EQ": "Closed", "TU": "Open"}
                },
                "balance_owed": {
                    "per_bureau": {"EX": f"${suffix}", "EQ": f"${suffix}", "TU": f"${suffix}"}
                },
                "date_opened": {
                    "per_bureau": {"EX": "2020-01-15", "EQ": "2020-01-16", "TU": "2020-01-17"}
                },
                "closed_date": {
                    "per_bureau": {"EX": None, "EQ": "2021-05-01", "TU": None}
                },
            },
        }
        pack_path = packs_dir / f"{account_id}.json"
        _write_json(pack_path, pack_payload)

        relative_path = f"frontend/review/packs/{account_id}.json"
        manifest_entries.append(
            {
                "account_id": account_id,
                "holder_name": holder_name,
                "primary_issue": primary_issue,
                "display": pack_payload["display"],
                "path": relative_path,
                "pack_path": relative_path,
                "bytes": pack_path.stat().st_size,
                "has_questions": True,
            }
        )

    manifest_payload = {
        "sid": sid,
        "stage": "review",
        "schema_version": "1.0",
        "packs": manifest_entries,
        "counts": {"packs": len(manifest_entries), "responses": 0},
        "responses_dir": "frontend/review/responses",
        "packs_count": len(manifest_entries),
        "questions": [
            "ownership",
            "recognize",
            "identity_theft",
            "explanation",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(run_dir / "frontend" / "review" / "index.json", manifest_payload)

    api_port = _find_free_port()
    with _run_api_server(runs_root, api_port) as api_base:
        status, manifest_response = _fetch_json(
            f"{api_base}/api/runs/{sid}/frontend/review/index"
        )
        assert status == 200
        assert manifest_response["sid"] == sid
        assert len(manifest_response.get("packs", [])) == len(accounts)

        detail_status, detail_response = _fetch_json(
            f"{api_base}/api/runs/{sid}/frontend/review/accounts/{accounts[0][0]}"
        )
        assert detail_status == 200
        assert detail_response["holder_name"] == accounts[0][1]
        assert detail_response["display"]["balance_owed"]["per_bureau"]["EX"].startswith("$")

        frontend_dir = ensure_frontend_dependencies
        ui_port = _find_free_port()
        ui_env = os.environ.copy()
        ui_env.update(
            {
                "VITE_API_URL": api_base,
                "VITE_ENABLE_FRONTEND_REVIEW_MOCK": "0",
                "BROWSER": "none",
                "CI": os.environ.get("CI", ""),
            }
        )
        with _run_ui_server(frontend_dir, ui_port, ui_env) as ui_base:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(f"{ui_base}/runs/{sid}/accounts", wait_until="networkidle")

                for account_id, holder_name, primary_issue in accounts:
                    page.wait_for_selector(f"text={holder_name}")
                    page.wait_for_selector(f"text={account_id}")
                    expected_issue = primary_issue.replace("_", " ")
                    page.wait_for_selector(f"text=Primary issue: {expected_issue}")

                page.get_by_text(accounts[0][1], exact=True).click()
                page.wait_for_selector("text=Questions")
                page.wait_for_selector("text=Submit answers")
                page.wait_for_selector("text=Account number:")
                browser.close()


