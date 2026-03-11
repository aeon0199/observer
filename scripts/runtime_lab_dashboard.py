from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "runtime_lab_dashboard"
RUNS_DIR = REPO_ROOT / "runs"
MAX_LOG_LINES = 400
REGISTRY_CANDIDATES = [
    ("runtime_lab (root)", REPO_ROOT / "models.json"),
    ("v1.5 observability", REPO_ROOT / "v1.5" / "models.json"),
    ("v2 intervention", REPO_ROOT / "intervention_engine_v1.5_v2" / "models.json"),
    ("v1 hysteresis", REPO_ROOT / "baseline_hysteresis_v1" / "models.json"),
]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _tail_text(path: Path, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _load_models(registry_path: Path) -> Dict[str, Any]:
    payload = _read_json(registry_path) or {}
    models = payload.get("models", {}) or {}
    return {
        "default_model": payload.get("default_model"),
        "models": [
            {
                "key": str(key),
                "hf_id": str(value.get("hf_id", "")),
                "device": value.get("device"),
                "torch_dtype": value.get("torch_dtype"),
            }
            for key, value in models.items()
        ],
    }


def _registry_presets() -> list[Dict[str, str]]:
    presets: list[Dict[str, str]] = []
    for label, path in REGISTRY_CANDIDATES:
        if path.exists():
            presets.append(
                {
                    "label": label,
                    "path": str(path.relative_to(REPO_ROOT)),
                }
            )
    return presets


def _resolve_registry_path(registry_param: str) -> Path:
    candidate = Path(registry_param)
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _infer_mode(run_dir: Path) -> str:
    name = run_dir.name
    if name.startswith("observe_run_"):
        return "observe"
    if name.startswith("stress_run_"):
        return "stress"
    if name.startswith("control_run_"):
        return "control"
    if name.startswith("hysteresis_run_"):
        return "hysteresis"
    return "unknown"


def _artifact_manifest(run_dir: Path) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for child in sorted(run_dir.iterdir()):
        if child.is_file():
            manifest[child.name] = str(child.relative_to(REPO_ROOT))
    return manifest


def _run_summary(run_dir: Path) -> Dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    results = _read_json(run_dir / "results.json")
    config = _read_json(run_dir / "config.json")
    mode = _infer_mode(run_dir)

    headline: Dict[str, Any] = {}
    if mode == "stress" and results:
        metrics = results.get("metrics", {}) or {}
        headline = {
            "primary_metric": metrics.get("primary_metric"),
            "regime": metrics.get("regime"),
            "token_match_rate": metrics.get("token_match_rate"),
            "recovery_ratio": metrics.get("recovery_ratio"),
            "device": ((results.get("runtime", {}) or {}).get("device")),
        }
    elif mode == "control" and summary:
        headline = {
            "avg_raw_div_mean": summary.get("avg_raw_div_mean"),
            "avg_score_mean": summary.get("avg_score_mean"),
            "status_counts": summary.get("status_counts"),
            "device": ((summary.get("runtime", {}) or {}).get("device")),
            "degraded_steps": ((summary.get("diagnostics_health", {}) or {}).get("degraded_steps")),
            "dashboard_path": (summary.get("artifacts", {}) or {}).get("dashboard_path"),
        }
    elif mode == "observe" and summary:
        headline = {
            "tokens": summary.get("tokens"),
            "avg_divergence": summary.get("avg_divergence"),
            "device": ((summary.get("runtime", {}) or {}).get("device")),
            "degraded_steps": ((summary.get("diagnostics_health", {}) or {}).get("degraded_steps")),
        }
    elif mode == "hysteresis" and summary:
        headline = {
            "regime": (summary.get("metrics", {}) or {}).get("regime"),
            "recovery": (summary.get("metrics", {}) or {}).get("recovery"),
            "distribution_shift": summary.get("distribution_shift"),
            "device": ((summary.get("runtime", {}) or {}).get("device")),
        }

    return {
        "id": run_dir.name,
        "mode": mode,
        "path": str(run_dir),
        "updated_at": int(run_dir.stat().st_mtime),
        "headline": headline,
        "summary": summary,
        "results": results,
        "config": config,
        "artifacts": _artifact_manifest(run_dir),
        "output_tail": _tail_text(run_dir / "output.txt") or _tail_text(run_dir / "output_base.txt"),
    }


class JobManager:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.runs_dir = repo_root / "runs"
        self._lock = threading.Lock()
        self._job: Optional[Dict[str, Any]] = None

    def _snapshot_unlocked(self) -> Dict[str, Any]:
        if not self._job:
            return {"status": "idle"}
        return dict(self._job)

    def _reader(self, proc: subprocess.Popen[str], job_id: str) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with self._lock:
                if not self._job or self._job.get("id") != job_id:
                    break
                self._job["logs"].append(line.rstrip("\n"))
                self._job["logs"] = self._job["logs"][-MAX_LOG_LINES:]
        proc.wait()
        with self._lock:
            if self._job and self._job.get("id") == job_id:
                self._job["returncode"] = proc.returncode
                self._job["status"] = "completed" if proc.returncode == 0 else "failed"
                self._job["finished_at"] = time.time()
                self._job["run"] = self._find_latest_run(self._job.get("mode"), self._job.get("started_at", 0.0))
                self._job["pid"] = None

    def _find_latest_run(self, mode: str, started_at: float) -> Optional[Dict[str, Any]]:
        if not self.runs_dir.exists():
            return None
        prefix_map = {
            "observe": "observe_run_",
            "stress": "stress_run_",
            "control": "control_run_",
            "hysteresis": "hysteresis_run_",
        }
        prefix = prefix_map.get(mode, "")
        candidates = [
            path for path in self.runs_dir.iterdir()
            if path.is_dir() and path.name.startswith(prefix) and path.stat().st_mtime >= started_at - 2.0
        ]
        if not candidates:
            return None
        latest = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        return {"id": latest.name, "path": str(latest)}

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._job and self._job.get("status") == "running":
                raise RuntimeError("A run is already in progress.")

            mode = str(payload.get("mode", "observe"))
            registry_path = str(payload.get("registry_path") or "models.json")
            prompt = str(payload.get("prompt") or "Explain how airplanes fly in a clear, accurate way.")
            max_tokens = int(payload.get("max_tokens") or 64)
            backend = str(payload.get("backend") or "hf")
            seed = int(payload.get("seed") or 42)
            custom_model = str(payload.get("custom_model_key") or "").strip()
            model = custom_model or str(payload.get("model") or "")

            cmd = ["python", "-m", "runtime_lab.cli.main", mode]
            if model:
                cmd += ["--model", model]
            cmd += ["--prompt", prompt]
            cmd += ["--max-tokens", str(max_tokens)]
            cmd += ["--seed", str(seed)]
            cmd += ["--backend", backend]
            cmd += ["--registry-path", registry_path]
            cmd += ["--runs-dir", str(self.runs_dir)]

            if payload.get("nnsight_remote"):
                cmd.append("--nnsight-remote")
            if payload.get("nnsight_device"):
                cmd += ["--nnsight-device", str(payload["nnsight_device"])]

            if mode == "stress":
                cmd += ["--layer", str(int(payload.get("layer") or -1))]
                cmd += ["--type", str(payload.get("intervention_type") or "additive")]
                cmd += ["--magnitude", str(float(payload.get("magnitude") or 2.0))]
                cmd += ["--start", str(int(payload.get("start") or 5))]
                cmd += ["--duration", str(int(payload.get("duration") or 10))]
            elif mode == "control":
                cmd += ["--measure-layer", str(int(payload.get("measure_layer") or -1))]
                cmd += ["--act-layer", str(int(payload.get("act_layer") or -1))]
                cmd += ["--type", str(payload.get("intervention_type") or "scaling")]
                if payload.get("shadow"):
                    cmd.append("--shadow")
            elif mode == "hysteresis":
                pass

            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
            env["RUNS_DIR"] = str(self.runs_dir)

            proc = subprocess.Popen(
                cmd,
                cwd=str(self.repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            job_id = str(uuid.uuid4())[:8]
            self._job = {
                "id": job_id,
                "pid": proc.pid,
                "mode": mode,
                "command": cmd,
                "status": "running",
                "started_at": time.time(),
                "finished_at": None,
                "returncode": None,
                "logs": [],
                "run": None,
            }
            thread = threading.Thread(target=self._reader, args=(proc, job_id), daemon=True)
            thread.start()
            return self._snapshot_unlocked()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._job or self._job.get("status") != "running" or not self._job.get("pid"):
                raise RuntimeError("No running job to stop.")
            pid = int(self._job["pid"])
            os.kill(pid, signal.SIGTERM)
            self._job["status"] = "stopping"
            return self._snapshot_unlocked()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()


JOB_MANAGER = JobManager(REPO_ROOT)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_payload(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            registry_param = parse_qs(parsed.query).get("registry", ["models.json"])[0]
            registry_path = _resolve_registry_path(registry_param)
            self._send_json(
                {
                    "repo_root": str(REPO_ROOT),
                    "runs_dir": str(RUNS_DIR),
                    "registry_path": str(registry_path),
                    "registry_presets": _registry_presets(),
                    "models": _load_models(registry_path),
                    "active_job": JOB_MANAGER.snapshot(),
                }
            )
            return

        if parsed.path == "/api/status":
            self._send_json({"job": JOB_MANAGER.snapshot()})
            return

        if parsed.path == "/api/runs":
            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            runs = [_run_summary(path) for path in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if path.is_dir()]
            self._send_json({"runs": runs[:30]})
            return

        if parsed.path.startswith("/api/run/"):
            run_id = parsed.path.split("/api/run/", 1)[1]
            run_dir = RUNS_DIR / run_id
            if not run_dir.exists():
                self._send_json({"error": "Run not found."}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json({"run": _run_summary(run_dir)})
            return

        if parsed.path == "/api/artifact":
            params = parse_qs(parsed.query)
            rel = params.get("path", [""])[0]
            target = (REPO_ROOT / rel).resolve()
            if not str(target).startswith(str(REPO_ROOT.resolve())) or not target.exists():
                self._send_json({"error": "Artifact not found."}, status=HTTPStatus.NOT_FOUND)
                return
            content_type = "text/plain; charset=utf-8"
            if target.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif target.suffix == ".json":
                content_type = "application/json; charset=utf-8"
            elif target.suffix == ".jsonl":
                content_type = "text/plain; charset=utf-8"
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            payload = self._read_payload()
            try:
                job = JOB_MANAGER.start(payload)
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.CREATED)
            return

        if parsed.path == "/api/stop":
            try:
                job = JOB_MANAGER.stop()
            except Exception as e:
                self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job})
            return

        self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Runtime Lab dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-open-browser", action="store_true")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, int(args.port)), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"[runtime-lab-dashboard] Serving at {url}")
    if not args.no_open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[runtime-lab-dashboard] Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
