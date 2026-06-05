from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import expect, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SERVER_EXE = ROOT / "build" / "Release" / "persona_gateway_e2e_server.exe"
SERVER_CONFIG = ROOT / "tools" / "persona_gateway_e2e_server.json"
PERSONA_REGISTRY = ROOT / "data" / "persona_gateway_e2e" / "persona_registry.json"
BASE_URL = "http://127.0.0.1:18080"


def wait_for_server(request, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = request.get(f"{BASE_URL}/", timeout=2500)
            if response.status == 200:
                return
            last_error = RuntimeError(f"GET / returned {response.status}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"E2E server did not become ready: {last_error}")


def start_server_if_needed(request):
    try:
        response = request.get(f"{BASE_URL}/", timeout=1500)
        if response.status == 200:
            return None
    except Exception:  # noqa: BLE001
        pass

    if not SERVER_EXE.exists():
        raise RuntimeError(f"missing server executable: {SERVER_EXE}")

    log_path = ROOT / "logs" / "persona_gateway_e2e" / "playwright-server.stdout.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        [str(SERVER_EXE), str(SERVER_CONFIG), "--no-stdin-stop"],
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    wait_for_server(request, timeout_s=20)
    return process


def api_e2e(request) -> dict:
    with PERSONA_REGISTRY.open("r", encoding="utf-8") as f:
        registry = json.load(f)
    lidazhi = next((p for p in registry.get("personas", []) if p.get("personaId") == "lidazhi"), None)
    if not lidazhi:
        raise RuntimeError(f"persona registry missing lidazhi: {PERSONA_REGISTRY}")

    user_uuid = f"playwright-user-{int(time.time())}"
    register = request.post(
        f"{BASE_URL}/api/auth/register",
        data={
            "userUuid": user_uuid,
            "tenantId": "default",
            "subject": user_uuid,
            "ttlSeconds": 3600,
        },
    )
    if register.status != 200:
        raise RuntimeError(f"register failed: {register.status} {register.text()}")
    register_body = register.json()
    token = register_body["data"]["token"]

    headers = {"Authorization": f"Bearer {token}"}
    me = request.get(f"{BASE_URL}/api/auth/me", headers=headers)
    if me.status != 200:
        raise RuntimeError(f"auth/me failed: {me.status} {me.text()}")
    me_body = me.json()
    assert me_body["data"]["authenticated"] is True
    assert me_body["data"]["userUuid"] == user_uuid

    session_id = f"pw-session-{int(time.time())}"
    create = request.post(
        f"{BASE_URL}/api/session/create",
        headers=headers,
        data={
            "traceId": "pw-session-create",
            "sessionId": session_id,
            "personaId": "lidazhi",
            "personality": lidazhi["personality"],
        },
    )
    if create.status != 200:
        raise RuntimeError(f"session/create failed: {create.status} {create.text()}")

    chat = request.post(
        f"{BASE_URL}/api/chat/message",
        headers=headers,
        data={
            "traceId": "pw-chat-message",
            "sessionId": session_id,
            "personaId": "lidazhi",
            "mode": "chat",
            "message": "李大志，老师刚讲完一元二次方程配方法，你现在听懂了吗？",
            "stream": False,
        },
        timeout=45000,
    )
    if chat.status != 200:
        raise RuntimeError(f"chat/message failed: {chat.status} {chat.text()}")
    chat_body = chat.json()
    content = chat_body.get("data", {}).get("reply", {}).get("content", "")
    if not content:
        raise RuntimeError(f"chat response has no content: {json.dumps(chat_body, ensure_ascii=False)}")

    metrics = request.get(f"{BASE_URL}/api/session/{session_id}/metrics", headers=headers)
    if metrics.status != 200:
        raise RuntimeError(f"session metrics failed: {metrics.status} {metrics.text()}")

    return {
        "userUuid": user_uuid,
        "sessionId": session_id,
        "chatContentPreview": content[:120],
        "pipelineLatency": chat_body.get("data", {}).get("pipelineLatency", {}),
        "memory": chat_body.get("data", {}).get("memory", {}),
    }


def ui_e2e(page) -> None:
    page.goto(BASE_URL, wait_until="domcontentloaded")
    expect(page.locator("#app")).to_be_visible(timeout=10000)

    register_button = page.get_by_role("button", name="注册")
    if register_button.count() > 0:
        inputs = page.locator("input")
        if inputs.count() >= 1:
            inputs.nth(0).fill(f"ui-user-{int(time.time())}")
        register_button.first.click()
        page.wait_for_timeout(1000)

    expect(page.locator("body")).not_to_contain_text("request header matched a security filter")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-server", action="store_true", help="leave a server started by this script running")
    args = parser.parse_args()

    server = None
    with sync_playwright() as playwright:
        request = playwright.request.new_context(ignore_https_errors=True)
        try:
            server = start_server_if_needed(request)
            api_result = api_e2e(request)

            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            ui_e2e(page)
            browser.close()

            print(json.dumps({"ok": True, "api": api_result}, ensure_ascii=False, indent=2))
            return 0
        except PlaywrightError as exc:
            print(f"[playwright-e2e] FAIL: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[playwright-e2e] FAIL: {exc}", file=sys.stderr)
            return 1
        finally:
            request.dispose()
            if server and not args.keep_server:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
