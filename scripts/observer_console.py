"""Observer Console — a live dashboard for Runtime Lab experiments.

Single-file server built on the stdlib. Streams token-level telemetry
over Server-Sent Events as runs execute.

    python scripts/observer_console.py [--port 8899]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "observer_console"
RUNS_DIR = REPO_ROOT / "runs"
REGISTRY_PATH = REPO_ROOT / "models.json"

MAX_LOGS = 500
EVENT_TAIL_POLL_S = 0.1
RUN_DISCOVERY_POLL_S = 0.2


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_mode(run_name: str) -> str:
    for mode in ("observe", "stress", "control", "hysteresis"):
        if run_name.startswith(f"{mode}_run_"):
            return mode
    if run_name.startswith("sweep_"):
        return "sweep"
    return "unknown"


def load_registry() -> Dict[str, Any]:
    payload = read_json(REGISTRY_PATH) or {}
    models = payload.get("models", {}) or {}
    return {
        "default_model": payload.get("default_model"),
        "models": [
            {"key": k, "hf_id": v.get("hf_id", "")}
            for k, v in models.items()
        ],
    }


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    mode = infer_mode(run_dir.name)

    if mode == "sweep":
        sweep = read_json(run_dir / "sweep.json") or {}
        desc = sweep.get("describe") or {}
        aggregate = sweep.get("aggregate") or {}
        headline: Dict[str, Any] = {
            "sweep_mode": sweep.get("mode"),
            "n_seeds": sweep.get("n_seeds"),
            "n_ok": sweep.get("n_ok"),
            "model": desc.get("model"),
        }
        # Surface mean/std of the key per-mode metric.
        def _pick(key: str) -> None:
            a = aggregate.get(key)
            if a:
                headline[key] = a.get("mean")
                headline[f"{key}_std"] = a.get("stdev")

        _pick("avg_divergence")
        _pick("metrics.recovery")
        _pick("metrics.drift")
        return {
            "id": run_dir.name,
            "mode": "sweep",
            "sweep_mode": sweep.get("mode"),
            "updated_at": int(run_dir.stat().st_mtime),
            "prompt": desc.get("prompt") or "",
            "headline": headline,
        }

    summary = read_json(run_dir / "summary.json") or {}
    config = read_json(run_dir / "config.json") or {}

    headline: Dict[str, Any] = {
        "model": summary.get("model_id") or config.get("config", {}).get("model"),
        "device": (summary.get("runtime", {}) or {}).get("device"),
        "dtype": (summary.get("runtime", {}) or {}).get("resolved_dtype"),
    }
    advisory = summary.get("advisory") or {}
    flags = advisory.get("flags") or []
    if flags:
        headline["advisory_flag"] = flags[0]
        headline["advisory_summary"] = advisory.get("summary_line")

    if mode == "observe":
        headline.update({
            "tokens": summary.get("tokens"),
            "avg_divergence": summary.get("avg_divergence"),
        })
    elif mode == "hysteresis":
        metrics = summary.get("metrics", {}) or {}
        headline.update({
            "regime": metrics.get("regime"),
            "drift": metrics.get("drift"),
            "hysteresis": metrics.get("hysteresis"),
            "recovery": metrics.get("recovery"),
        })
    elif mode == "stress":
        results = read_json(run_dir / "results.json") or {}
        metrics = results.get("metrics", {}) or {}
        headline.update({
            "regime": metrics.get("regime"),
            "recovery_ratio": metrics.get("recovery_ratio"),
            "token_match_rate": metrics.get("token_match_rate"),
        })
    elif mode == "control":
        headline.update({
            "avg_raw_div_mean": summary.get("avg_raw_div_mean"),
            "avg_score_mean": summary.get("avg_score_mean"),
            "status_counts": summary.get("status_counts"),
        })

    prompt = config.get("config", {}).get("prompt") or ""
    if not prompt:
        # Stress stores prompt in results.json → config → prompt.
        # Hysteresis stores under config.json → config → prompt (covered above).
        results_any = read_json(run_dir / "results.json") or {}
        prompt = (results_any.get("config") or {}).get("prompt") or prompt

    return {
        "id": run_dir.name,
        "mode": mode,
        "updated_at": int(run_dir.stat().st_mtime),
        "prompt": prompt,
        "headline": headline,
    }


def list_runs(limit: int = 60) -> List[Dict[str, Any]]:
    if not RUNS_DIR.exists():
        return []
    dirs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [summarize_run(p) for p in dirs[:limit]]


def run_detail(run_dir: Path) -> Dict[str, Any]:
    mode = infer_mode(run_dir.name)

    if mode == "sweep":
        sweep = read_json(run_dir / "sweep.json") or {}
        return {
            "id": run_dir.name,
            "mode": "sweep",
            "sweep": sweep,
        }

    detail: Dict[str, Any] = {
        "id": run_dir.name,
        "mode": mode,
        "summary": read_json(run_dir / "summary.json"),
        "config": read_json(run_dir / "config.json"),
        "results": read_json(run_dir / "results.json"),
    }

    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        events: List[Dict[str, Any]] = []
        try:
            with events_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        detail["events"] = events

    if mode == "hysteresis":
        detail["frames"] = {
            "base": read_json(run_dir / "frame_base.json"),
            "perturb": read_json(run_dir / "frame_perturb.json"),
            "reask": read_json(run_dir / "frame_reask.json"),
        }
        detail["outputs"] = {
            "base": _read_text(run_dir / "output_base.txt"),
            "perturb": _read_text(run_dir / "output_perturb.txt"),
            "reask": _read_text(run_dir / "output_reask.txt"),
        }
    else:
        detail["output"] = _read_text(run_dir / "output.txt")

    return detail


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# SSE bus
# ---------------------------------------------------------------------------


class EventBus:
    """Fan out messages from the job to every active SSE client."""

    def __init__(self) -> None:
        self._clients: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2048)
        with self._lock:
            self._clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, event: str, data: Any) -> None:
        payload = {"event": event, "data": data}
        with self._lock:
            dead: List[queue.Queue] = []
            for q in self._clients:
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                if q in self._clients:
                    self._clients.remove(q)


BUS = EventBus()


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._job: Optional[Dict[str, Any]] = None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if not self._job:
                return {"status": "idle"}
            return {k: v for k, v in self._job.items() if k not in ("proc",)}

    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if self._job and self._job["status"] in ("running", "starting"):
                raise RuntimeError("A job is already running.")

        mode = str(payload.get("mode") or "observe")
        if mode not in ("observe", "stress", "control", "hysteresis"):
            raise ValueError(f"Unknown mode: {mode}")

        prompt = str(payload.get("prompt") or "")
        max_tokens = int(payload.get("max_tokens") or 64)
        seed = int(payload.get("seed") or 42)
        model = str(payload.get("model") or "").strip()

        cmd: List[str] = ["python", "-m", "runtime_lab.cli.main", mode]
        if model:
            cmd += ["--model", model]
        if prompt:
            cmd += ["--prompt", prompt]
        cmd += ["--max-tokens", str(max_tokens), "--seed", str(seed)]

        probe_layers = str(payload.get("probe_layers") or "auto")
        if mode in ("observe", "stress", "control"):
            cmd += ["--probe-layers", probe_layers]

        seeds_spec = payload.get("seeds")
        if seeds_spec:
            cmd += ["--seeds", str(seeds_spec)]

        # Sampling — always pass through if given so sampling can be opted into.
        if payload.get("temperature") is not None:
            cmd += ["--temperature", str(float(payload["temperature"]))]
        if payload.get("top_p") is not None:
            cmd += ["--top-p", str(float(payload["top_p"]))]
        if payload.get("top_k") is not None:
            cmd += ["--top-k", str(int(payload["top_k"]))]

        # Helper: only pass a CLI flag if the caller gave a value; otherwise let
        # the CLI's own default (which may be a semantic string like "mid") apply.
        def _pass(key: str, flag: str, cast=lambda x: str(x)) -> None:
            if key in payload and payload[key] is not None:
                cmd.extend([flag, cast(payload[key])])

        if mode == "stress":
            _pass("layer", "--layer")  # pass through as string; CLI handles semantic
            _pass("intervention_type", "--type")
            _pass("magnitude", "--magnitude", lambda v: str(float(v)))
            if payload.get("absolute_magnitude"):
                cmd.append("--absolute-magnitude")
            _pass("start", "--start", lambda v: str(int(v)))
            _pass("duration", "--duration", lambda v: str(int(v)))
        elif mode == "hysteresis":
            pm = str(payload.get("perturbation_mode") or "prompt")
            cmd += ["--perturbation-mode", pm]
            if pm == "noise":
                _pass("noise_layer", "--noise-layer")  # semantic-friendly
                _pass("noise_magnitude", "--noise-magnitude", lambda v: str(float(v)))
                _pass("noise_start", "--noise-start", lambda v: str(int(v)))
                _pass("noise_duration", "--noise-duration", lambda v: str(int(v)))
        elif mode == "control":
            _pass("measure_layer", "--measure-layer", lambda v: str(int(v)))
            _pass("act_layer", "--act-layer", lambda v: str(int(v)))
            _pass("intervention_type", "--type")
            if payload.get("shadow"):
                cmd.append("--shadow")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

        started_at = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        job_id = uuid.uuid4().hex[:8]
        job = {
            "id": job_id,
            "mode": mode,
            "model": model,
            "prompt": prompt,
            "command": cmd,
            "status": "starting",
            "started_at": started_at,
            "finished_at": None,
            "pid": proc.pid,
            "run_id": None,
            "run_dir": None,
            "exit_code": None,
            "proc": proc,
            "logs": [],
            "event_count": 0,
        }
        with self._lock:
            self._job = job

        BUS.publish("job_started", {k: v for k, v in job.items() if k != "proc"})

        threading.Thread(target=self._watch_stdout, args=(job_id,), daemon=True).start()
        threading.Thread(target=self._watch_run_dir, args=(job_id,), daemon=True).start()
        return self.snapshot()

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            job = self._job
            if not job or job["status"] not in ("running", "starting"):
                raise RuntimeError("No job to stop.")
            pid = job.get("pid")
        if pid:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        with self._lock:
            if self._job:
                self._job["status"] = "stopping"
        return self.snapshot()

    # -------- internals

    def _watch_stdout(self, job_id: str) -> None:
        with self._lock:
            job = self._job
        if not job or job["id"] != job_id:
            return
        proc: subprocess.Popen = job["proc"]
        assert proc.stdout is not None

        for line in proc.stdout:
            line = line.rstrip("\n")
            if not line:
                continue
            with self._lock:
                if self._job and self._job["id"] == job_id:
                    self._job["logs"].append(line)
                    self._job["logs"] = self._job["logs"][-MAX_LOGS:]
                    if self._job["status"] == "starting":
                        self._job["status"] = "running"
            BUS.publish("log", {"line": line})

        proc.wait()
        exit_code = proc.returncode
        with self._lock:
            if self._job and self._job["id"] == job_id:
                self._job["exit_code"] = exit_code
                self._job["finished_at"] = time.time()
                self._job["status"] = "completed" if exit_code == 0 else "failed"
                run_dir = self._job.get("run_dir")

        final_summary: Optional[Dict[str, Any]] = None
        if run_dir:
            final_summary = read_json(Path(run_dir) / "summary.json")

        BUS.publish("job_finished", {
            "exit_code": exit_code,
            "status": "completed" if exit_code == 0 else "failed",
            "run_id": Path(run_dir).name if run_dir else None,
            "summary": final_summary,
        })

    def _watch_run_dir(self, job_id: str) -> None:
        """Discover run directory, then tail events.jsonl (if present)."""
        with self._lock:
            job = self._job
        if not job or job["id"] != job_id:
            return

        mode = job["mode"]
        started_at = job["started_at"]
        prefix = f"{mode}_run_"

        run_dir: Optional[Path] = None
        while True:
            with self._lock:
                current = self._job
                if not current or current["id"] != job_id:
                    return
                if current["status"] in ("completed", "failed", "stopping"):
                    # Still try to find the dir briefly before giving up.
                    pass

            if RUNS_DIR.exists():
                candidates = [
                    p for p in RUNS_DIR.iterdir()
                    if p.is_dir() and p.name.startswith(prefix)
                    and p.stat().st_mtime >= started_at - 1.0
                ]
                if candidates:
                    run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
                    with self._lock:
                        if self._job and self._job["id"] == job_id:
                            self._job["run_dir"] = str(run_dir)
                            self._job["run_id"] = run_dir.name
                    BUS.publish("run_detected", {"run_id": run_dir.name, "run_dir": str(run_dir)})
                    break

            with self._lock:
                if self._job and self._job["id"] == job_id and self._job["status"] in ("completed", "failed"):
                    return
            time.sleep(RUN_DISCOVERY_POLL_S)

        if run_dir is None:
            return

        # Stream events.jsonl if it exists (observe/stress/control).
        events_path = run_dir / "events.jsonl"
        offset = 0
        buffer = ""
        while True:
            with self._lock:
                current = self._job
                if not current or current["id"] != job_id:
                    return
                job_status = current["status"]

            if events_path.exists():
                try:
                    with events_path.open("r", encoding="utf-8") as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                except Exception:
                    chunk = ""
                if chunk:
                    buffer += chunk
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        with self._lock:
                            if self._job and self._job["id"] == job_id:
                                self._job["event_count"] += 1
                        BUS.publish("event", ev)

            if job_status in ("completed", "failed"):
                # Drain one more pass, then exit.
                time.sleep(0.15)
                if events_path.exists():
                    try:
                        with events_path.open("r", encoding="utf-8") as f:
                            f.seek(offset)
                            chunk = f.read()
                    except Exception:
                        chunk = ""
                    if chunk:
                        buffer += chunk
                        for line in buffer.split("\n"):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                ev = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            BUS.publish("event", ev)
                # For hysteresis, publish the frames when they appear.
                if self._job and self._job["mode"] == "hysteresis":
                    for phase in ("base", "perturb", "reask"):
                        fp = run_dir / f"frame_{phase}.json"
                        if fp.exists():
                            data = read_json(fp)
                            if data is not None:
                                BUS.publish("frame", {"phase": phase, "frame": data})
                return

            time.sleep(EVENT_TAIL_POLL_S)


JOBS = JobManager()


# ---------------------------------------------------------------------------
# Capabilities (machine-readable description of what this rig can do)
# ---------------------------------------------------------------------------

_MODES_DOC: Dict[str, Dict[str, Any]] = {
    "observe": {
        "description": "Passive observability — generate tokens and log full telemetry (divergence, spectral, SVD, layer-stiffness). No interventions.",
        "useful_for": [
            "Baseline trajectories on a prompt × model combination",
            "Checking whether divergence scales with length before drawing conclusions",
            "Gathering reference distributions for comparisons later",
        ],
        "params": {
            "prompt": "(str) what the model generates from",
            "model": "(str) key in models.json",
            "max_tokens": "(int) length of generation",
            "seed": "(int) deterministic seed",
            "temperature": "(float) 0=greedy (default); >0 enables sampling, essential for perturbation effects to flip tokens",
            "seeds": "(str) multi-seed sweep spec like '0-4' or '1,3,7'; produces sweep_observe_<ts>/",
            "probe_layers": "'auto' (mid-stack + final) or comma-separated ints",
        },
    },
    "stress": {
        "description": "Branchpoint A/B comparison — run baseline vs intervention from the same prompt-pass seed cache. Compares token match, logit-KL, recovery.",
        "useful_for": [
            "Measuring sensitivity of a layer to a specific perturbation",
            "Checking if an intervention has *any* effect even when tokens don't flip (use logit_kl_mean_during)",
        ],
        "params": {
            "layer": "(str|int) 'mid' (default, resolves to n//2-1), 'early', 'late', or explicit int",
            "type": "additive | projection | scaling | sae",
            "magnitude": "(float) default 0.15, interpreted as fraction of hidden-state norm unless --absolute-magnitude",
            "start": "(int) token index where intervention begins",
            "duration": "(int) token count during which intervention is active",
        },
        "advisories_emitted": [
            "no-op (token_match=1.0) — recommends mid-stack + larger magnitude + sampling",
            "runaway — suggests checking magnitude / layer choice",
        ],
    },
    "hysteresis": {
        "description": "BASE → PERTURB → REASK protocol. Measures whether the model's state returns to baseline after a perturbation is removed.",
        "useful_for": [
            "perturbation_mode=prompt: prompt-contamination persistence (legacy — not true internal hysteresis)",
            "perturbation_mode=noise: real internal-dynamics hysteresis via hidden-state injection at a chosen layer",
        ],
        "params": {
            "perturbation_mode": "prompt | noise",
            "noise_layer": "'mid' (default) or int — recommended to stay mid-stack so effect cascades",
            "noise_magnitude": "float, default 0.3, fraction of hidden norm",
            "noise_start": "(int) step where noise begins",
            "noise_duration": "(int) how many steps noise is active",
        },
    },
    "control": {
        "description": "Closed-loop controller — continuously monitors divergence, applies scaling intervention when threshold crossed. Shadow mode measures without intervening.",
        "useful_for": [
            "Testing whether proportional control actually stabilizes trajectories (run in active mode!)",
            "Calibrating thresholds via shadow mode before going active",
        ],
        "params": {
            "shadow": "(bool) measure only; set to false to let controller intervene",
            "measure_layer": "(int) layer whose divergence drives the controller",
            "act_layer": "(int) layer where scaling is applied",
            "threshold_warn / threshold_crit": "divergence thresholds for WARN / CRIT status",
            "scale_warn / scale_crit": "scaling factor applied at each status level",
        },
    },
}


def _capabilities_payload() -> Dict[str, Any]:
    registry = load_registry()
    return {
        "version": "0.2.0-llm-first",
        "description": (
            "Observer — closed-loop stability instrumentation for LLM inference. "
            "This capabilities endpoint is for LLM-driven experiment planning: "
            "fetch it, pick a mode, construct a payload per `params`, POST to /api/launch, "
            "then read /api/runs/<id> to get the summary including `advisory` — a structured "
            "block explaining what happened and what to try next."
        ),
        "endpoints": {
            "POST /api/launch": "Start a run. Body: {mode, model, prompt, ...mode-specific params}. Returns {job: {...}}.",
            "GET /api/status": "Poll for current job status.",
            "POST /api/stop": "Cancel active job.",
            "GET /api/runs": "List recent runs (summary metadata).",
            "GET /api/runs/{id}": "Full detail for one run, including advisory.",
            "GET /api/compare?ids=a,b,c": "Get divergence/entropy/hidden series for multiple runs for overlay comparison.",
            "GET /api/stream": "SSE stream of job events (logs, events, job_finished).",
            "GET /api/capabilities": "This payload.",
        },
        "models": registry.get("models", []),
        "default_model": registry.get("default_model"),
        "modes": _MODES_DOC,
        "advisories": {
            "description": (
                "Every completed run's summary.json contains `advisory`, a structured block with "
                "{observations, likely_causes, next_actions, flags, confidence, summary_line}. "
                "The next_actions list is designed to be acted on directly: copy the params dict "
                "into your next POST /api/launch body."
            ),
            "known_flags": ["no-op", "nominal", "quiet", "volatile", "degenerate", "persistent-effect",
                            "noise-absorbed", "shadow", "active", "good-recovery", "prompt-contamination-test"],
        },
        "recipes": _RECIPES,
    }


_RECIPES: Dict[str, Dict[str, Any]] = {
    "smoke-test": {
        "description": "Smallest possible end-to-end validation — 8 tokens, default model.",
        "mode": "observe",
        "payload": {"max_tokens": 8, "prompt": "The sky is"},
    },
    "hysteresis-noise-validate": {
        "description": "Real hidden-state hysteresis with defaults that actually show effect (mid-stack, magnitude 0.3, sampling on).",
        "mode": "hysteresis",
        "payload": {
            "perturbation_mode": "noise",
            "noise_layer": "mid",
            "noise_magnitude": 0.3,
            "noise_duration": 8,
            "temperature": 0.8,
            "max_tokens": 64,
        },
    },
    "stress-logit-kl": {
        "description": "Stress run that measures logit-KL even if tokens don't flip. Use to confirm an intervention IS affecting the model even when match_rate=1.",
        "mode": "stress",
        "payload": {
            "layer": "mid",
            "type": "additive",
            "magnitude": 0.3,
            "start": 5,
            "duration": 10,
            "max_tokens": 48,
        },
    },
    "control-ab": {
        "description": "Run the same prompt in shadow and active control back-to-back. Compare metrics to verify the controller actually helps.",
        "mode": "control",
        "chain": [
            {"payload": {"shadow": True, "max_tokens": 64}, "tag": "shadow"},
            {"payload": {"shadow": False, "max_tokens": 64}, "tag": "active"},
        ],
    },
    "seed-variance": {
        "description": "Run N seeds to check that a measured effect is not a single-seed fluke.",
        "mode": "observe",
        "payload": {"seeds": "0-4", "max_tokens": 48, "temperature": 0.7},
    },
}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    # ---- helpers

    def _write(self, status: int, body: bytes, content_type: str, extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._write(status, data, "application/json; charset=utf-8")

    def _read_body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    # ---- static

    def _serve_static(self, rel: str) -> bool:
        path = (STATIC_DIR / rel).resolve()
        try:
            path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return False
        if not path.exists() or not path.is_file():
            return False
        mime = STATIC_MIME.get(path.suffix, "application/octet-stream")
        body = path.read_bytes()
        self._write(200, body, mime)
        return True

    # ---- routes

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            if self._serve_static("index.html"):
                return
            self._json({"error": "index.html missing"}, 500)
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            if self._serve_static(rel):
                return
            self._write(404, b"not found", "text/plain")
            return

        if path == "/api/models":
            self._json(load_registry())
            return

        if path == "/api/runs":
            self._json({"runs": list_runs()})
            return

        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            run_dir = RUNS_DIR / run_id
            if not run_dir.exists() or not run_dir.is_dir():
                self._json({"error": "run not found"}, 404)
                return
            self._json(run_detail(run_dir))
            return

        if path == "/api/compare":
            ids = parse_qs(parsed.query).get("ids", [""])[0]
            run_ids = [i for i in ids.split(",") if i]
            out = []
            for rid in run_ids[:10]:
                rd = RUNS_DIR / rid
                if not rd.exists():
                    continue
                mode = infer_mode(rd.name)
                summary = read_json(rd / "summary.json") or {}
                series: List[Dict[str, Any]] = []
                events_path = rd / "events.jsonl"
                if events_path.exists():
                    try:
                        with events_path.open("r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    ev = json.loads(line)
                                    series.append({
                                        "t": ev.get("t"),
                                        "divergence": (ev.get("diagnostics") or {}).get("divergence"),
                                        "entropy": ((ev.get("diagnostics") or {}).get("spectral") or {}).get("spectral_entropy"),
                                        "hidden": ev.get("hidden_post_norm"),
                                    })
                                except Exception:
                                    pass
                    except Exception:
                        pass
                out.append({
                    "id": rid,
                    "mode": mode,
                    "series": series,
                    "headline": summarize_run(rd).get("headline", {}),
                })
            self._json({"runs": out})
            return

        if path == "/api/status":
            self._json({"job": JOBS.snapshot()})
            return

        if path == "/api/capabilities":
            self._json(_capabilities_payload())
            return

        if path == "/api/stream":
            self._stream()
            return

        self._write(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/launch":
            try:
                JOBS.start(self._read_body())
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            self._json({"job": JOBS.snapshot()}, 201)
            return
        if path == "/api/stop":
            try:
                JOBS.stop()
            except Exception as e:
                self._json({"error": str(e)}, 400)
                return
            self._json({"job": JOBS.snapshot()})
            return
        self._write(404, b"not found", "text/plain")

    # ---- SSE

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        q = BUS.subscribe()
        # Send snapshot of current job on connect.
        try:
            self._send_sse("hello", {"job": JOBS.snapshot()})
        except Exception:
            BUS.unsubscribe(q)
            return

        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                except queue.Empty:
                    # heartbeat
                    try:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                    except Exception:
                        break
                    continue
                try:
                    self._send_sse(msg["event"], msg["data"])
                except Exception:
                    break
        finally:
            BUS.unsubscribe(q)

    def _send_sse(self, event: str, data: Any) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")
        self.wfile.write(payload)
        self.wfile.flush()


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[observer-console] serving at {url}")
    print("[observer-console] press Ctrl-C to stop")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[observer-console] shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
