from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import pytest
from playwright.sync_api import sync_playwright

from tests.e2e.test_frontend_review_end_to_end import (
    _find_free_port,
    _run_ui_server,
    ensure_frontend_dependencies,
    ensure_playwright,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("npm") is None, reason="npm is required for the frontend dev server"),
]


def _build_display(holder_name: str, suffix: str) -> dict:
    return {
        "display_version": "2024-01-01",
        "holder_name": holder_name,
        "primary_issue": "incorrect_information",
        "account_number": {
            "per_bureau": {
                "experian": f"****{suffix}",
                "equifax": f"****{suffix}",
                "transunion": f"****{suffix}",
            }
        },
        "account_type": {
            "per_bureau": {
                "experian": "Credit Card",
                "equifax": "Credit Card",
                "transunion": "Credit Card",
            }
        },
        "status": {
            "per_bureau": {
                "experian": "Open",
                "equifax": "Closed",
                "transunion": "Open",
            }
        },
        "balance_owed": {
            "per_bureau": {
                "experian": "$200",
                "equifax": "$0",
                "transunion": "$200",
            }
        },
        "date_opened": {
            "per_bureau": {
                "experian": "2021-01-02",
                "equifax": "2021-01-03",
                "transunion": "2021-01-01",
            }
        },
        "closed_date": {
            "per_bureau": {
                "experian": None,
                "equifax": "2023-05-01",
                "transunion": None,
            }
        },
    }


def _fulfill_json(route, payload: dict) -> None:
    route.fulfill(
        status=200,
        headers={"Content-Type": "application/json"},
        body=json.dumps(payload),
    )


def test_frontend_review_waiting_flow(ensure_frontend_dependencies: Path, ensure_playwright: None):
    sid = "S-UI-WAIT"
    accounts = [
        ("idx-001", "Alice Example", "001"),
        ("idx-002", "Bob Sample", "002"),
    ]

    pack_payloads = {
        account_id: {
            "account_id": account_id,
            "holder_name": holder_name,
            "primary_issue": "incorrect_information",
            "questions": [
                {"id": "ownership", "prompt": "Do you own this account?", "required": True},
                {"id": "recognize", "prompt": "Do you recognize this account?", "required": True},
                {"id": "identity_theft", "prompt": "Is this identity theft?", "required": True},
                {"id": "explanation", "prompt": "Anything else we should know?", "required": False},
            ],
            "display": _build_display(holder_name, suffix),
        }
        for account_id, holder_name, suffix in accounts
    }

    waiting_stage = {
        "sid": sid,
        "stage": "review",
        "schema_version": "1.0",
        "packs_count": 0,
        "counts": {"packs": 0, "responses": 0},
        "packs": [],
        "packs_index": [],
        "index_path": "frontend/review/index.json",
        "index_rel": "review/index.json",
        "packs_dir": "frontend/review/packs",
        "packs_dir_rel": "review/packs",
        "responses_dir": "frontend/review/responses",
        "responses_dir_rel": "review/responses",
    }

    ready_stage = {
        "sid": sid,
        "stage": "review",
        "schema_version": "1.0",
        "packs_count": len(accounts),
        "counts": {"packs": len(accounts), "responses": 0},
        "packs": [
            {
                "account_id": account_id,
                "holder_name": holder_name,
                "primary_issue": "incorrect_information",
                "pack_path": f"frontend/review/packs/{account_id}.json",
                "pack_path_rel": f"packs/{account_id}.json",
                "path": f"frontend/review/packs/{account_id}.json",
                "file": f"frontend/review/packs/{account_id}.json",
                "has_questions": True,
            }
            for account_id, holder_name, _ in accounts
        ],
        "packs_index": [
            {"account": account_id, "file": f"packs/{account_id}.json"}
            for account_id, _, _ in accounts
        ],
        "index_path": "frontend/review/index.json",
        "index_rel": "review/index.json",
        "packs_dir": "frontend/review/packs",
        "packs_dir_rel": "review/packs",
        "responses_dir": "frontend/review/responses",
        "responses_dir_rel": "review/responses",
    }

    pack_listing = {
        "items": [
            {"account_id": account_id, "file": f"frontend/review/packs/{account_id}.json"}
            for account_id, _, _ in accounts
        ]
    }

    manifest_ready = {"value": False}
    saved_requests: list[dict] = []

    ui_port = _find_free_port()
    api_origin = f"http://127.0.0.1:{ui_port}"
    env = {
        "VITE_API_URL": api_origin,
        "VITE_ENABLE_FRONTEND_REVIEW_MOCK": "0",
        "BROWSER": "none",
        "CI": os.environ.get("CI", ""),
    }

    with _run_ui_server(ensure_frontend_dependencies, ui_port, env) as ui_base:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page()

            page.add_init_script(
                """
                (() => {
                  const instances = [];
                  class TestEventSource {
                    constructor(url) {
                      this.url = url;
                      this.readyState = 1;
                      this.withCredentials = false;
                      this.onerror = null;
                      this.onmessage = null;
                      this.onopen = null;
                      this._listeners = {};
                      instances.push(this);
                      if (typeof this.onopen === 'function') {
                        try { this.onopen(new Event('open')); } catch (err) { console.error(err); }
                      }
                    }
                    addEventListener(type, listener) {
                      if (!this._listeners[type]) {
                        this._listeners[type] = [];
                      }
                      this._listeners[type].push(listener);
                    }
                    removeEventListener(type, listener) {
                      if (!this._listeners[type]) {
                        return;
                      }
                      this._listeners[type] = this._listeners[type].filter((fn) => fn !== listener);
                    }
                    emit(type, payload) {
                      const listeners = this._listeners[type] ?? [];
                      const data = typeof payload === 'string' ? payload : JSON.stringify(payload ?? {});
                      const event = { data };
                      for (const listener of [...listeners]) {
                        try { listener(event); } catch (err) { console.error(err); }
                      }
                      if (type === 'message' && typeof this.onmessage === 'function') {
                        try { this.onmessage(event); } catch (err) { console.error(err); }
                      }
                    }
                    close() {
                      this.readyState = 2;
                    }
                  }
                  window.__testEventSources = instances;
                  window.__triggerTestEventSource = (index, type, payload) => {
                    const target = instances[index];
                    if (target) {
                      target.emit(type, payload);
                    }
                  };
                  window.EventSource = TestEventSource;
                })();
                """
            )

            def handle_root_index(route, request):
                payload = ready_stage if manifest_ready["value"] else waiting_stage
                _fulfill_json(route, {"sid": sid, "review": payload})

            def handle_stage_index(route, request):
                payload = ready_stage if manifest_ready["value"] else waiting_stage
                _fulfill_json(route, payload)

            def handle_frontend_index_api(route, request):
                count = len(accounts) if manifest_ready["value"] else 0
                _fulfill_json(route, {"packs_count": count})

            def handle_frontend_review_index_api(route, request):
                if manifest_ready["value"]:
                    payload = {"frontend": {"review": ready_stage}}
                    _fulfill_json(route, payload)
                else:
                    payload = {
                        "status": "building",
                        "queued": True,
                        "frontend": {"review": waiting_stage},
                    }
                    _fulfill_json(route, payload, status=202)

            def handle_pack_listing(route, request):
                if manifest_ready["value"]:
                    _fulfill_json(route, pack_listing)
                else:
                    _fulfill_json(route, {"items": []})

            def handle_pack_asset(route, request):
                parsed = urlparse(request.url)
                account_id = Path(parsed.path).name.replace(".json", "")
                payload = pack_payloads.get(account_id)
                if payload is None:
                    route.fulfill(status=404)
                else:
                    _fulfill_json(route, payload)

            def handle_response_post(route, request):
                parsed = urlparse(request.url)
                account_id = Path(parsed.path).name
                try:
                    payload = request.post_data_json() or {}
                except Exception:
                    payload = {}
                saved_requests.append({"account_id": account_id, "payload": payload})
                response_payload = {
                    "sid": sid,
                    "account_id": account_id,
                    "answers": payload.get("answers", {}),
                    "saved_at": "2024-01-01T00:00:00Z",
                }
                _fulfill_json(route, response_payload)

            page.route(f"{api_origin}/runs/{sid}/frontend/index.json", handle_root_index)
            page.route(f"{api_origin}/runs/{sid}/frontend/review/index.json", handle_stage_index)
            page.route(f"{api_origin}/api/runs/{sid}/frontend/index", handle_frontend_index_api)
            page.route(
                f"{api_origin}/api/runs/{sid}/frontend/review/index",
                handle_frontend_review_index_api,
            )
            page.route(f"{api_origin}/api/runs/{sid}/frontend/review/packs", handle_pack_listing)
            page.route(f"{api_origin}/runs/{sid}/frontend/review/packs/*.json", handle_pack_asset)
            page.route(f"{api_origin}/api/runs/{sid}/frontend/review/response/*", handle_response_post)

            try:
                page.goto(f"{ui_base}/runs/{sid}/review", wait_until="networkidle")
                page.wait_for_selector("text=Waiting for review packs…")

                manifest_ready["value"] = True
                page.evaluate(
                    "count => window.__triggerTestEventSource?.(0, 'packs_ready', { packs_count: count })",
                    len(accounts),
                )

                page.wait_for_selector("text=Account idx-001")
                page.wait_for_selector("text=Account idx-002")
                page.wait_for_selector("select#account-question-ownership")

                page.select_option("select#account-question-ownership", "yes")
                page.select_option("select#account-question-recognize", "no")
                page.select_option("select#account-question-identity-theft", "no")
                page.fill("textarea#account-question-explanation", "Automated review response.")
                page.get_by_role("button", name="Save answers").first.click()

                page.wait_for_selector("span:has-text('Saved')")
            finally:
                browser.close()

    assert saved_requests, "Expected at least one submission to be recorded"
    first_saved = saved_requests[0]
    assert first_saved["account_id"] == "idx-001"
    assert first_saved["payload"].get("answers", {}).get("ownership") == "yes"
