from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def terminate_process(process: subprocess.Popen[object]) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def wait_for_health(health_probe: Path, target: str, timeout_seconds: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_seconds
    last_output = ""

    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                sys.executable,
                str(health_probe),
                "--target",
                target,
                "--timeout",
                "1.0",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        last_output = "\n".join(part for part in (stdout, stderr) if part)

        if result.returncode == 0:
            return True, last_output

        time.sleep(1.0)

    return False, last_output


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the inference server and run a smoke test.")
    parser.add_argument("--server", required=True, help="Path to bert_inference_server executable")
    parser.add_argument("--client", required=True, help="Path to bert_inference_client executable")
    parser.add_argument("--model", required=True, help="Path to the smoke-test ONNX model")
    parser.add_argument(
        "--health-probe",
        required=True,
        help="Path to tools/grpc_health_probe.py",
    )
    parser.add_argument("--target", default="127.0.0.1:50051", help="Server target host:port")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=45.0,
        help="Seconds to wait for the server to become healthy",
    )
    args = parser.parse_args()

    server_path = Path(args.server).resolve()
    client_path = Path(args.client).resolve()
    model_path = Path(args.model).resolve()
    health_probe_path = Path(args.health_probe).resolve()

    if ":" not in args.target:
        raise SystemExit(f"Invalid target: {args.target}")
    port = args.target.rsplit(":", 1)[1]

    with tempfile.TemporaryDirectory(prefix="bert-smoke-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        server_stdout_path = temp_dir_path / "server_stdout.log"
        server_stderr_path = temp_dir_path / "server_stderr.log"

        with server_stdout_path.open("w", encoding="utf-8") as stdout_file, \
                server_stderr_path.open("w", encoding="utf-8") as stderr_file:
            server_process = subprocess.Popen(
                [
                    str(server_path),
                    str(model_path),
                    port,
                    "--stats-log-interval-seconds",
                    "0",
                    "--slow-request-ms",
                    "0",
                ],
                cwd=str(server_path.parent),
                stdout=stdout_file,
                stderr=stderr_file,
            )

        try:
            healthy, health_output = wait_for_health(
                health_probe=health_probe_path,
                target=args.target,
                timeout_seconds=args.startup_timeout,
            )
            if not healthy:
                print("Health check did not become SERVING", file=sys.stderr)
                if health_output:
                    print(health_output, file=sys.stderr)
                print(read_text(server_stdout_path), file=sys.stderr)
                print(read_text(server_stderr_path), file=sys.stderr)
                return 1

            print(health_output)

            client_result = subprocess.run(
                [str(client_path), args.target],
                cwd=str(client_path.parent),
                check=False,
                capture_output=True,
                text=True,
            )
            if client_result.stdout:
                print(client_result.stdout)
            if client_result.stderr:
                print(client_result.stderr, file=sys.stderr)

            if client_result.returncode != 0:
                print("Smoke client failed", file=sys.stderr)
                print(read_text(server_stdout_path), file=sys.stderr)
                print(read_text(server_stderr_path), file=sys.stderr)
                return client_result.returncode

            return 0
        finally:
            terminate_process(server_process)


if __name__ == "__main__":
    raise SystemExit(main())
