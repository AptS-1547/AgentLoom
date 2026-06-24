from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SERVER_EXE = ROOT / "build" / "x64-Release-Tests-v145" / "Release" / "webrtc_signaling_smoke_server.exe"
BASE_URL = "http://127.0.0.1:18180"
WS_URL = "ws://127.0.0.1:18180/ws/vision/signaling"
FRAME_DIR = ROOT / "logs" / "webrtc_frames"


def wait_for_server(request, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = request.get(f"{BASE_URL}/health", timeout=1500)
            if response.status == 200:
                return
            last_error = RuntimeError(f"GET /health returned {response.status}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"WebRTC smoke server did not become ready: {last_error}")


def start_server_if_needed(request, decoder: str, frame_dir: Path, sampler: str, adaptive_sampler: bool):
    try:
        response = request.get(f"{BASE_URL}/health", timeout=1000)
        if response.status == 200:
            return None
    except Exception:  # noqa: BLE001
        pass

    if not SERVER_EXE.exists():
        raise RuntimeError(f"missing server executable: {SERVER_EXE}")

    log_path = ROOT / "logs" / "webrtc_signaling_smoke_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    command = [str(SERVER_EXE), "--port", "18180", "--decoder", decoder, "--frame-dir", str(frame_dir), "--sampler", sampler]
    if adaptive_sampler:
        command.append("--adaptive-sampler")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    wait_for_server(request, timeout_s=20)
    return process


def run_browser_smoke(page, duration_s: float, sessions: int) -> dict:
    page.goto(BASE_URL, wait_until="domcontentloaded")
    return page.evaluate(
        """async ({ wsUrl, durationMs, sessions }) => {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240, frameRate: 10 },
                audio: false,
            });

            async function runSession(index) {
                const sessionId = `playwright-webrtc-smoke-${index}`;
                const pc = new RTCPeerConnection({ iceServers: [] });
                const channel = pc.createDataChannel(`smoke-${index}`);
                const localStream = new MediaStream();
                for (const track of stream.getVideoTracks()) {
                    localStream.addTrack(track.clone());
                }
                for (const track of localStream.getVideoTracks()) {
                    pc.addTrack(track, localStream);
                }

                const events = [];
                const pendingRemoteIce = [];
                const ws = new WebSocket(wsUrl);

                const waitOpen = new Promise((resolve, reject) => {
                    ws.onopen = resolve;
                    ws.onerror = () => reject(new Error(`websocket open failed for ${sessionId}`));
                    setTimeout(() => reject(new Error(`websocket open timeout for ${sessionId}`)), 8000);
                });
                await waitOpen;

                const waitAnswer = new Promise((resolve, reject) => {
                    ws.onmessage = (event) => {
                        const message = JSON.parse(event.data);
                        events.push(message.type);
                        if (message.type === "answer") {
                            resolve(message);
                        } else if (message.type === "ice") {
                            const candidate = {
                                sdpMLineIndex: message.payload.sdp_mline_index,
                                candidate: message.payload.candidate,
                            };
                            if (pc.remoteDescription) {
                                pc.addIceCandidate(candidate).catch((error) => {
                                    events.push(`ice-error:${error.message}`);
                                });
                            } else {
                                pendingRemoteIce.push(candidate);
                            }
                        } else if (message.type === "error") {
                            reject(new Error(`${message.payload.code}: ${message.payload.message}`));
                        }
                    };
                    setTimeout(() => reject(new Error(`answer timeout for ${sessionId}`)), 12000);
                });

                pc.onicecandidate = (event) => {
                    if (!event.candidate) {
                        return;
                    }
                    ws.send(JSON.stringify({
                        type: "ice",
                        session_id: sessionId,
                        payload: {
                            sdp_mline_index: event.candidate.sdpMLineIndex ?? 0,
                            candidate: event.candidate.candidate,
                        },
                    }));
                };

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);
                ws.send(JSON.stringify({
                    type: "offer",
                    session_id: sessionId,
                    payload: { sdp: offer.sdp },
                }));

                const answerMessage = await waitAnswer;
                await pc.setRemoteDescription({ type: "answer", sdp: answerMessage.payload.sdp });
                for (const candidate of pendingRemoteIce) {
                    await pc.addIceCandidate(candidate);
                }

                return { sessionId, pc, channel, ws, localStream, events, answerLength: answerMessage.payload.sdp.length };
            }

            const runs = await Promise.all(Array.from({ length: sessions }, (_, index) => runSession(index)));
            await new Promise((resolve) => setTimeout(resolve, durationMs));

            const result = runs.map((run) => ({
                sessionId: run.sessionId,
                events: run.events,
                signalingState: run.pc.signalingState,
                iceConnectionState: run.pc.iceConnectionState,
                localDescriptionType: run.pc.localDescription?.type,
                remoteDescriptionType: run.pc.remoteDescription?.type,
                answerLength: run.answerLength,
                dataChannelReadyState: run.channel.readyState,
            }));

            for (const run of runs) {
                run.ws.send(JSON.stringify({ type: "close", session_id: run.sessionId }));
                run.ws.close();
                for (const track of run.localStream.getTracks()) {
                    track.stop();
                }
                run.pc.close();
            }
            for (const track of stream.getTracks()) {
                track.stop();
            }
            return result;
        }""",
        {"wsUrl": WS_URL, "durationMs": int(duration_s * 1000), "sessions": sessions},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", default="software", choices=["auto", "software", "nvidia", "vaapi"])
    parser.add_argument("--frame-dir", default=str(FRAME_DIR))
    parser.add_argument("--sampler", default="all", choices=["all", "opencv"])
    parser.add_argument("--adaptive-sampler", action="store_true")
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--real-camera", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    args = parser.parse_args()
    frame_dir = Path(args.frame_dir)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    server = None
    with sync_playwright() as playwright:
        request = playwright.request.new_context()
        browser = None
        try:
            server = start_server_if_needed(request, args.decoder, frame_dir, args.sampler, args.adaptive_sampler)
            launch_args = [
                "--use-fake-ui-for-media-stream",
                "--allow-file-access-from-files",
            ]
            if not args.real_camera:
                launch_args.append("--use-fake-device-for-media-stream")
            browser = playwright.chromium.launch(
                headless=False if (args.real_camera or args.headed) else True,
                args=launch_args,
            )
            context = browser.new_context(permissions=["camera"])
            page = context.new_page()
            result = run_browser_smoke(page, args.duration, args.sessions)
            for session in result:
                if session["remoteDescriptionType"] != "answer":
                    raise RuntimeError(f"remote answer was not applied: {session}")

            deadline = time.time() + 10
            frames = []
            while time.time() < deadline:
                frames = sorted(frame_dir.glob("frame_*.ppm"))
                if not frames:
                    frames = sorted(frame_dir.glob("*_frame_*.ppm"))
                if frames:
                    break
                time.sleep(0.25)
            min_frames = args.min_frames
            if min_frames is None:
                min_frames = 0 if args.sampler == "opencv" else 1
            if len(frames) < min_frames:
                raise RuntimeError(f"no decoded frames were written: {result}")

            per_session = {}
            for session in result:
                per_session[session["sessionId"]] = len(list(frame_dir.glob(f"{session['sessionId']}_frame_*.ppm")))

            print(json.dumps({
                "ok": True,
                "webrtc": result,
                "frames": {
                    "count": len(frames),
                    "perSession": per_session,
                    "first": str(frames[0]) if frames else None,
                    "firstBytes": frames[0].stat().st_size if frames else 0,
                },
            }, ensure_ascii=False, indent=2))
            return 0
        except PlaywrightError as exc:
            print(f"[webrtc-smoke] FAIL: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[webrtc-smoke] FAIL: {exc}", file=sys.stderr)
            return 1
        finally:
            if browser:
                browser.close()
            request.dispose()
            if server and not args.keep_server:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == "__main__":
    raise SystemExit(main())
