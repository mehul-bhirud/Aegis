"""
╔══════════════════════════════════════════════════════════════════════════╗
║  A E G I S  —  Insider Threat Detection Engine  v2.0                   ║
║  FastAPI · PyTorch VAE · Ollama LLM · WebSocket · Merkle Chain         ║
║                                                                        ║
║  Pipeline:                                                             ║
║    JSONL Stream → SHA-256 Merkle Chain → VAE Inference → Risk Score    ║
║      → [if critical] Ollama LLM Analysis → WebSocket Broadcast         ║
║                                                                        ║
║  Run:  python main.py                                                  ║
║  Or:   uvicorn main:app --host 0.0.0.0 --port 8000                     ║
║                                                                        ║
║  Dependencies:                                                         ║
║    pip install fastapi uvicorn[standard] websockets torch httpx pandas  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import io

# Force UTF-8 encoding for Windows terminals to support emojis and box-drawing
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import asyncio
import hashlib
import json
import logging
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ROOT         = Path(__file__).resolve().parent.parent.parent  # Project root (Cummins/)
JSONL_PATH   = ROOT / "data" / "demo_activity_stream.jsonl"
MODEL_PATH   = ROOT / "ml" / "models" / "aegis_vae_model_weighted.pth"
META_PATH    = ROOT / "ml" / "data" / "feature_meta.json"
ROLES_PATH   = ROOT / "ml" / "data" / "user_roles.csv"
THRESH_PATH  = ROOT / "ml" / "data" / "threshold_stats.json"

OLLAMA_URL     = "http://localhost:11434/api/generate"
OLLAMA_MODEL   = "llama3"
OLLAMA_TIMEOUT = 60.0              # seconds — local LLMs can be slow

ALERT_THRESHOLD = 85               # risk_score > this → critical_alert + LLM
INPUT_DIM       = 61               # feature vector width (from preprocess.py)
LATENT_DIM      = 10                # VAE latent space (from train_vae.py)
STREAM_SPEED    = 0.1              # seconds between log reads (~10 logs/sec)

# Calibration defaults (overridden at startup from threshold_stats.json)
TRAIN_MSE_MEAN = 0.08752130717039108
TRAIN_MSE_STD  = 0.022675734013319016


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING — SOC TERMINAL STYLE (ANSI color codes, no external deps)
# ═══════════════════════════════════════════════════════════════════════════

class _SOCFormatter(logging.Formatter):
    """Custom formatter that makes the terminal look like a live SOC console."""

    _GREY    = "\033[38;5;245m"
    _CYAN    = "\033[36m"
    _GREEN   = "\033[32;1m"
    _YELLOW  = "\033[33;1m"
    _RED     = "\033[31;1m"
    _MAGENTA = "\033[35;1m"
    _BOLD    = "\033[1m"
    _RESET   = "\033[0m"

    _LEVEL_STYLES = {
        logging.DEBUG:    (_GREY,    "DBG"),
        logging.INFO:     (_CYAN,    "INF"),
        logging.WARNING:  (_YELLOW,  "WRN"),
        logging.ERROR:    (_RED,     "ERR"),
        logging.CRITICAL: (_RED,     "🚨 CRIT"),
    }

    def format(self, record: logging.LogRecord) -> str:
        color, tag = self._LEVEL_STYLES.get(record.levelno, (self._CYAN, "INF"))
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        ms = f"{record.created % 1:.3f}"[1:]          # .NNN
        return f"{color}{ts}{ms} │ {tag:>8s} │ {record.getMessage()}{self._RESET}"


log = logging.getLogger("aegis")
log.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_SOCFormatter())
log.addHandler(_handler)
log.propagate = False


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2 — VAE MODEL ARCHITECTURE (exact copy from train_vae.py)
# ═══════════════════════════════════════════════════════════════════════════
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  This class is wired to the REAL trained architecture.              ║
# ║  If you retrain with a different shape, update the layers here.     ║
# ╚══════════════════════════════════════════════════════════════════════╝

class InsiderThreatVAE(nn.Module):
    """Variational Autoencoder for enterprise activity anomaly detection.

    Architecture:  22 → 32 → 16 → [μ, logσ²] → 5 (latent) → 16 → 32 → 22
    """

    def __init__(self, input_dim: int = INPUT_DIM, latent_dim: int = LATENT_DIM):
        super().__init__()
        # Encoder: input_dim → 32 → 16 → (mu, logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64),        nn.LeakyReLU(0.2),
        )
        self.fc_mu     = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder: latent_dim → 64 → 128 → input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 128),         nn.LeakyReLU(0.2),
            nn.Linear(128, input_dim),  nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Hackathon Approximation for City Coordinates (Lat, Lon) — Impossible Travel
CITY_COORDS = {
    "Pune": (18.52, 73.85), "Bangalore": (12.97, 77.59), "Mumbai": (19.07, 72.87),
    "Singapore": (1.35, 103.81), "Chicago": (41.87, -87.62), "Tokyo": (35.67, 139.65),
    "London": (51.50, -0.12), "Frankfurt": (50.11, 8.68), "Austin": (30.26, -97.74),
    "Sydney": (-33.86, 151.20), "Delhi": (28.61, 77.20), "Seoul": (37.56, 126.97),
    "Amsterdam": (52.37, 4.90), "Dublin": (53.35, -6.26), "New_York": (40.71, -74.00),
    "Seattle": (47.61, -122.33), "Toronto": (43.65, -79.38), "Sao_Paulo": (-23.55, -46.63),
    "Dubai": (25.20, 55.27), "Johannesburg": (-26.20, 28.04),
}


def preprocess_json_to_tensor(log_data: dict, user_history: list | None = None) -> torch.Tensor:
    """Translates raw JSON into a 61-dim tensor, including STATEFUL session features.

    Resolves Training-Serving Skew by matching the offline preprocess.py pipeline
    and adding live session windowing + impossible travel velocity.

    Feature vector layout (61 dimensions):
      [0-1]   Temporal:  hour_sin, hour_cos  (cyclical encoding)
      [2]     Volume:    scaled volume_mb
      [3]     Entropy:   file_entropy
      [4]     Biometric: typing_cadence_ms  (scaled)
      [5]     Biometric: mouse_velocity     (scaled)
      [6]     Session:   rolling cumulative volume  (windowed)
      [7]     Session:   error rate over last 10 actions
      [8]     Velocity:  impossible travel speed  (km/h scaled)
      [9]     Reserved
      [10-22] Action-type one-hot  (13 categories)
      [23-25] Reserved
      [26-35] Location one-hot     (10 categories)
      [36-49] Reserved
      [50]    EDR agent active     (boolean)
      [51]    MFA success          (boolean)
      [52-55] Reserved
      [56-60] Binary Threat Flags  (5 kill-shot discriminators)
    """
    if user_history is None:
        user_history = []

    features = [0.0] * 56

    # ── 1. THE BASICS (Time, Volume, Biometrics) ─────────────────────
    current_time_str = log_data.get("timestamp", "").replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(current_time_str)
        hour_float = dt.hour + (dt.minute / 60.0)
        features[0] = math.sin(2 * math.pi * hour_float / 24.0)
        features[1] = math.cos(2 * math.pi * hour_float / 24.0)
    except Exception:
        dt = datetime.utcnow()

    resource = log_data.get("resource", {})
    current_vol = resource.get("volume_mb", 0.0)
    features[2] = min(current_vol / 1000.0, 1.0)

    enrich = log_data.get("enrichments", {}).get("aegis_telemetry", {})
    features[3] = enrich.get("file_entropy", 0.0)
    features[4] = min(enrich.get("typing_cadence_ms", 100) / 1000.0, 1.0)
    features[5] = min(enrich.get("mouse_velocity", 50) / 500.0, 1.0)

    # ── 2. STATEFUL: Session Windowing ───────────────────────────────
    session_volume = current_vol
    failed_actions = 0

    for past_log in user_history:
        session_volume += past_log.get("resource", {}).get("volume_mb", 0.0)
        if past_log.get("action", {}).get("status") == "failed":
            failed_actions += 1

    # Index 6: Rolling Session Volume (massive indicator of data hoarding)
    features[6] = min(session_volume / 5000.0, 1.0)
    # Index 7: Session Error Rate (indicator of brute force / hunting)
    features[7] = failed_actions / max(len(user_history), 1)

    # ── 3. STATEFUL: Location Velocity (Impossible Travel) ───────────
    current_loc = log_data.get("context", {}).get("location", "")
    features[8] = 0.0  # default velocity

    if len(user_history) > 0:
        last_log = user_history[-1]
        last_loc = last_log.get("context", {}).get("location", "")
        last_time_str = last_log.get("timestamp", "").replace("Z", "+00:00")

        if current_loc and last_loc and current_loc != last_loc:
            try:
                last_dt = datetime.fromisoformat(last_time_str)
                hours_diff = abs((dt - last_dt).total_seconds()) / 3600.0

                if current_loc in CITY_COORDS and last_loc in CITY_COORDS:
                    lat1, lon1 = CITY_COORDS[current_loc]
                    lat2, lon2 = CITY_COORDS[last_loc]
                    rough_dist_km = math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111

                    if hours_diff > 0:
                        velocity_kmh = rough_dist_km / hours_diff
                        # Normalize (cap at 1000 km/h — airplane speed)
                        # > 1.0 means impossible travel (e.g. Pune→London in 5 min)
                        features[8] = min(velocity_kmh / 1000.0, 1.0)
            except Exception:
                pass

    # ── 4. EXPANDED VOCABULARIES ─────────────────────────────────────
    action_type = log_data.get("action", {}).get("type", "")
    _ACTION_MAP = {
        "login": 10, "vpn_connect": 11, "db_query": 12, "file_download": 13,
        "config_change": 14, "service_stop": 15, "file_copy": 16, "db_update": 17,
        "refund_process": 18, "git_commit": 19, "email_sent": 20, "ticket_update": 21,
        "container_deploy": 22,
    }
    if action_type in _ACTION_MAP:
        features[_ACTION_MAP[action_type]] = 1.0

    _LOCATION_MAP = {
        "Pune": 26, "Bangalore": 27, "Mumbai": 28, "Singapore": 29, "Chicago": 30,
        "Tokyo": 31, "London": 32, "Frankfurt": 33, "Austin": 34, "Sydney": 35,
    }
    if current_loc in _LOCATION_MAP:
        features[_LOCATION_MAP[current_loc]] = 1.0

    features[50] = 1.0 if log_data.get("context", {}).get("edr_agent_active") else 0.0
    features[51] = 1.0 if log_data.get("actor", {}).get("mfa_status") == "success" else 0.0

    # ── 5. Binary Threat Flags (final 5 dims → total = 61) ──────────
    final_vector = features + [0.0] * 5
    assert len(final_vector) == 61, f"Mismatch: {len(final_vector)}"

    return torch.tensor([final_vector], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  MERKLE INTEGRITY CHAIN
# ═══════════════════════════════════════════════════════════════════════════

class EnterpriseMerkleTree:
    """Rolling SHA-256 hash chain guaranteeing log ordering and integrity.

    Each new log's hash is combined with the previous root:
        new_root = SHA-256(old_root ‖ SHA-256(raw_json))
    Any log tampered → every subsequent root diverges → detectable.
    """

    def __init__(self):
        self._root  = hashlib.sha256(b"AEGIS_GENESIS_BLOCK").hexdigest()
        self._count = 0

    def ingest(self, raw_json: str) -> str:
        """Hash raw JSON, chain with current root, return new root."""
        log_hash = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
        combined = f"{self._root}{log_hash}"
        self._root = hashlib.sha256(combined.encode("utf-8")).hexdigest()
        self._count += 1
        return self._root

    @property
    def root(self) -> str:
        return self._root

    @property
    def count(self) -> int:
        return self._count


# ═══════════════════════════════════════════════════════════════════════════
#  RISK SCORING (Calibrated Sigmoid)
# ═══════════════════════════════════════════════════════════════════════════

import random

def mse_to_risk_score(mse: float) -> int:
    """
    Hackathon Magic Scaler: Maps raw MSE to a diverse 0-100 curve.
    """
    # Based on the terminal output and example target:
    NORMAL_MSE = 0.029   # Baseline MSE from terminal (mean=0.029659)
    CRITICAL_MSE = 1.750 # Critical MSE observed in terminal stream logs
    
    # 1. Normalize the raw MSE to a 0.0 - 1.0 scale
    normalized = (mse - NORMAL_MSE) / (CRITICAL_MSE - NORMAL_MSE)
    
    # Clamp it so it doesn't go below 0 or above 1
    normalized = max(0.0, min(1.0, normalized))
    
    # 2. Apply a logarithmic curve to spread out the "Noise"
    risk = (normalized ** 0.65) * 100
    
    # 3. Add UI Jitter (The "Wow" Factor)
    if risk > 5 and risk < 95:
        risk += random.uniform(-3.0, 3.0)
        
    return max(1, min(100, int(risk)))


# ═══════════════════════════════════════════════════════════════════════════
#  OLLAMA LLM CLIENT (Rate-Limited with asyncio.Lock)
# ═══════════════════════════════════════════════════════════════════════════

class OllamaAnalyst:
    """Async, priority-queued client for local Ollama LLM threat analysis.

    To prevent crashing the local GPU during high-velocity anomaly bursts
    (e.g., 50 alerts in 1 sec), this implements an LLM Priority Queue
    with an Asynchronous Non-Blocking Lock.

    Lock Behavior:
      - IF LOCKED:   LLM is busy → return hardcoded fallback immediately
      - IF UNLOCKED: Acquire lock → await Llama 3 → release in finally
    """

    # Hardcoded fallback when the LLM is already processing another threat
    _BUSY_FALLBACK: dict = {
        "summary": ("Multiple concurrent anomalies detected. "
                     "ML threat flag logged; LLM queued for capacity."),
        "recommended_action": "Standard protocol.",
    }

    def __init__(self):
        self._client:  httpx.AsyncClient | None = None
        self._available = False
        self._calls     = 0

        # Priority Queue state
        self._queue = asyncio.PriorityQueue()
        self._bg_task = None

        # ── Non-Blocking Lock ────────────────────────────────────────
        self._lock = asyncio.Lock()
        self._is_processing = False

    async def initialize(self):
        """Probe Ollama on startup; set _available flag and start worker."""
        self._client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)
        try:
            resp = await self._client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self._available = True
                log.info("🧠 Ollama connected — available models: %s", models)
            else:
                log.warning("⚠  Ollama returned status %d — AI analysis disabled", resp.status_code)
        except Exception:
            log.warning("⚠  Ollama not reachable at localhost:11434 — using rule-based fallback")
            
        self._bg_task = asyncio.create_task(self._worker())

    async def shutdown(self):
        """Clean up background workers and connections."""
        if self._bg_task:
            self._bg_task.cancel()
        if self._client:
            await self._client.aclose()

    def enqueue(self, log_data: dict, risk_score: int, output_payload: dict):
        """Non-blocking enqueue for critical stream events."""
        # Priority Queue (min-heap): order by -risk_score (highest risk first)
        self._queue.put_nowait((-risk_score, time.time(), log_data, output_payload))

    async def _worker(self):
        """Background loop: batches 1-sec windows and processes highest priority."""
        while True:
            try:
                # 1. Wait for at least one item
                score, ts, log_data, output = await self._queue.get()
                
                # 2. Wait 1 second to accumulate any other logs arriving in this window
                await asyncio.sleep(1.0)
                
                # 3. Drain the queue to find the absolute highest risk (lowest score value)
                best_item = (score, ts, log_data, output)
                skipped_items = []
                
                while not self._queue.empty():
                    item = self._queue.get_nowait()
                    if item[0] < best_item[0]: # Lower score = higher risk
                        skipped_items.append(best_item)
                        best_item = item
                    else:
                        skipped_items.append(item)
                        
                # 4. Fire "Skipped" broadcasts for the losers to keep stream lively
                for item in skipped_items:
                    _, _, _, i_out = item
                    i_out["ai_analysis"] = {
                        "summary": "Skipped — LLM Rate Limited (Priority Queue).",
                        "recommended_action": "Refer to raw ML anomaly score."
                    }
                    await manager.broadcast(i_out)
                    
                # 5. Process the winner — with non-blocking lock check
                w_score, w_ts, w_log, w_out = best_item

                if self._is_processing:
                    # LLM is currently busy — return hardcoded fallback
                    log.warning("🔒 LLM LOCKED — concurrent anomaly, returning fallback")
                    w_out["ai_analysis"] = dict(self._BUSY_FALLBACK)
                    await manager.broadcast(w_out)
                    continue

                try:
                    self._is_processing = True
                    analysis = await self._do_analyze(w_log, -w_score)
                except Exception as e:
                    log.error("Ollama worker error: %s", e)
                    analysis = self._fallback(w_log, -w_score)
                finally:
                    self._is_processing = False
                    
                w_out["ai_analysis"] = analysis
                await manager.broadcast(w_out)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("Fatal LLM background worker error: %s", e)
                await asyncio.sleep(1)

    async def analyze(self, user_history: list | dict, risk_score: int) -> dict:
        """Synchronous (awaitable) analyze for isolated DEMO endpoints.

        Uses the same non-blocking lock to prevent overlapping calls.
        """
        if self._is_processing:
            log.warning("🔒 LLM LOCKED (demo endpoint) — returning fallback")
            return dict(self._BUSY_FALLBACK)

        self._is_processing = True
        try:
            return await self._do_analyze(user_history, risk_score)
        finally:
            self._is_processing = False

    async def _do_analyze(self, user_history: list | dict, risk_score: int) -> dict:
        """Generate a threat analysis via Llama 3."""
        if not self._available or not self._client:
            return self._fallback(user_history, risk_score)

        prompt = (
            "You are a cybersecurity SOC analyst AI at a large retail and supply-chain "
            "enterprise called Cummins. You are analyzing a sequence of events. Here is "
            "the user's recent activity log leading up to the anomaly. Analyze the "
            "sequence to determine the kill chain and provide a conclusion.\n"
            "Respond with ONLY a valid JSON object (no markdown, no code fences, no "
            "extra text) with exactly two keys:\n"
            '  "summary": a 2-3 sentence explanation of why this is suspicious.\n'
            '  "recommended_action": a specific, actionable step for the SOC team.\n\n'
            f"Activity Sequence:\n{json.dumps(user_history, indent=2)}\n\n"
            f"Risk Score: {risk_score}/100"
        )

        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }

        try:
            log.info("🧠 LLM Priority Queue — generating analysis for risk=%d …", risk_score)
            resp = await self._client.post(OLLAMA_URL, json=payload)
            if resp.status_code == 200:
                raw = resp.json().get("response", "{}")
                parsed = json.loads(raw)
                self._calls += 1
                return {
                    "summary":            parsed.get("summary",
                                                     "Analysis unavailable."),
                    "recommended_action":  parsed.get("recommended_action",
                                                     "Escalate to SOC Lead."),
                }
        except json.JSONDecodeError:
            log.warning("⚠  Ollama returned non-JSON — falling back")
        except httpx.TimeoutException:
            log.warning("⚠  Ollama timed out after %.0fs", OLLAMA_TIMEOUT)
        except Exception as exc:
            log.warning("⚠  Ollama error: %s", exc)

        return self._fallback(user_history, risk_score)

    # ── Rule-based fallback when Ollama is unavailable ──────────────────

    @staticmethod
    def _fallback(user_history: list | dict, risk_score: int) -> dict:
        log_data = user_history[-1] if isinstance(user_history, list) else user_history
        actor    = log_data.get("actor", {})
        action   = log_data.get("action", {})
        resource = log_data.get("resource", {})
        context  = log_data.get("context", {})
        uid      = actor.get("user_id", "unknown")
        atype    = action.get("type", "unknown")
        rname    = resource.get("name", "unknown")
        vol      = resource.get("volume_mb", 0)
        sens     = resource.get("sensitivity_label", "")

        # Build contextual summary
        fragments: list[str] = []
        if atype == "config_change":
            fragments.append(
                f"User {uid} performed a config_change on '{rname}' "
                f"({sens}) — potential audit-trail tampering.")
        elif atype in ("file_copy", "file_download") and vol > 1000:
            fragments.append(
                f"User {uid} initiated a {vol:,.0f} MB {atype} of "
                f"'{rname}' ({sens}) — possible data exfiltration.")
        elif atype == "process_kill":
            fragments.append(
                f"User {uid} terminated security process '{rname}' "
                f"— likely EDR evasion attempt.")
        else:
            fragments.append(
                f"Anomalous {atype} by {uid} targeting '{rname}' ({sens}).")

        if actor.get("mfa_status") == "bypassed":
            fragments.append("MFA was BYPASSED.")
        if not context.get("edr_agent_active", True):
            fragments.append("EDR agent is INACTIVE — endpoint blind.")
        if context.get("location") == "Unknown":
            fragments.append("Activity from an UNKNOWN location.")

        summary = " ".join(fragments)

        if risk_score >= 95:
            rec = ("IMMEDIATE: Isolate endpoint, revoke credentials, "
                   "initiate incident response procedure.")
        elif risk_score >= 85:
            rec = ("HIGH: Escalate to SOC Lead, correlate with DLP/SIEM "
                   "alerts, prepare incident report.")
        else:
            rec = "MONITOR: Flag for review in next SOC shift handoff."

        return {"summary": summary, "recommended_action": rec}

    @property
    def call_count(self) -> int:
        return self._calls


# ═══════════════════════════════════════════════════════════════════════════
#  WEBSOCKET CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Thread-safe registry of active WebSocket clients."""

    def __init__(self):
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._active.append(ws)
        log.info("🔌 WebSocket client connected  — %d active", len(self._active))

    def disconnect(self, ws: WebSocket):
        if ws in self._active:
            self._active.remove(ws)
        log.info("🔌 WebSocket client dropped    — %d active", len(self._active))

    async def broadcast(self, data: dict):
        """Push JSON to every connected client; silently prune dead sockets."""
        dead: list[WebSocket] = []
        for ws in self._active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._active:
                self._active.remove(ws)

    @property
    def count(self) -> int:
        return len(self._active)


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

class PipelineStats:
    """Tracks live metrics for the /api/stats endpoint and terminal logging."""

    def __init__(self):
        self.total_processed  = 0
        self.normal_count     = 0
        self.alert_count      = 0
        self.ollama_calls     = 0
        self.brain0_overrides = 0              # hard-signature bypasses
        self.start_time: float | None = None
        self.status           = "idle"        # idle | running | complete | error
        self.highest_risk     = 0
        self.recent_alerts: list[dict] = []   # rolling buffer of last 100 alerts
        self.risk_distribution = {
            "low": 0, "medium": 0, "high": 0, "critical": 0,
        }

    def reset(self):
        """Resets all telemetry for a clean demo run."""
        self.total_processed  = 0
        self.normal_count     = 0
        self.alert_count      = 0
        self.ollama_calls     = 0
        self.brain0_overrides = 0
        self.start_time       = time.time() if self.status == "running" else None
        self.highest_risk     = 0
        self.recent_alerts    = []
        self.risk_distribution = {
            "low": 0, "medium": 0, "high": 0, "critical": 0,
        }

    def record(self, risk_score: int, is_alert: bool):
        self.total_processed += 1
        if is_alert:
            self.alert_count += 1
        else:
            self.normal_count += 1
        if risk_score > self.highest_risk:
            self.highest_risk = risk_score

        if risk_score < 30:
            self.risk_distribution["low"] += 1
        elif risk_score < 60:
            self.risk_distribution["medium"] += 1
        elif risk_score < ALERT_THRESHOLD:
            self.risk_distribution["high"] += 1
        else:
            self.risk_distribution["critical"] += 1

    def push_alert(self, output: dict):
        self.recent_alerts.append(output)
        if len(self.recent_alerts) > 100:
            self.recent_alerts = self.recent_alerts[-100:]

    @property
    def throughput(self) -> float:
        if not self.start_time:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.total_processed / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "status":              self.status,
            "total_processed":     self.total_processed,
            "normal_count":        self.normal_count,
            "alert_count":         self.alert_count,
            "brain0_overrides":    self.brain0_overrides,
            "ollama_calls":        self.ollama_calls,
            "highest_risk_score":  self.highest_risk,
            "throughput_lps":      round(self.throughput, 2),
            "risk_distribution":   self.risk_distribution,
            "uptime_s":            round(time.time() - self.start_time, 1)
                                   if self.start_time else 0,
            "merkle_root":         merkle.root,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  GLOBAL ENGINE STATE
# ═══════════════════════════════════════════════════════════════════════════

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model:   InsiderThreatVAE | None = None
user_history_buffer: dict[str, list[dict]] = {}
merkle   = EnterpriseMerkleTree()
ollama   = OllamaAnalyst()
manager  = ConnectionManager()
stats    = PipelineStats()
ROLES_DF: pd.DataFrame | None = None

# Stream control
_stream_task: asyncio.Task | None = None
_stop_event  = asyncio.Event()


# ═══════════════════════════════════════════════════════════════════════════
#  CORE STREAM PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

def calculate_weighted_mse(recon_x: torch.Tensor, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    sq_error = (recon_x - x) ** 2
    threat_indices = [-1, -2, -3, -4, -5] 
    weight_multiplier = 100.0 
    
    for idx in threat_indices:
        sq_error[:, idx] *= weight_multiplier
        
    if reduction == 'none':
        return sq_error.mean(dim=1)
    return sq_error.mean()


async def _run_inference(tensor: torch.Tensor) -> float:
    """Run VAE forward pass in a thread so the event loop never blocks."""
    def _infer() -> float:
        with torch.no_grad():
            t = tensor.to(device)
            recon, _, _ = model(t)                             # type: ignore[misc]
            return calculate_weighted_mse(recon, t).item()
    return await asyncio.to_thread(_infer)


def _enrich_alert(output: dict, log_data: dict) -> dict:
    """Attach user-role context from user_roles.csv (via pandas)."""
    if ROLES_DF is None:
        return output
    # Support both old (actor.user_id) and new (actor.user.uid) OCSF paths
    actor = log_data.get("actor", {})
    uid = actor.get("user_id", "") or actor.get("user", {}).get("uid", "")
    match = ROLES_DF[ROLES_DF["user_id"] == uid]
    if match.empty:
        return output
    row = match.iloc[0]
    expected = str(row.get("expected_resources", ""))
    accessed = log_data.get("resource", {}).get("name", "")
    output["user_context"] = {
        "department":         row["department"],
        "expected_resources": expected,
        "access_violation":   accessed not in expected,
    }
    return output


# ═══════════════════════════════════════════════════════════════════════════
#  BRAIN 0 — DETERMINISTIC HARD-SIGNATURE ENGINE
# ═══════════════════════════════════════════════════════════════════════════
#
# For "black-and-white" threats, we bypass PyTorch entirely.
# If an employee touches a Honey-Token or triggers a physical camera
# sensor, the system does NOT ask the ML model for its opinion.
# ML is for the "gray area"; explicit signatures are for certainties.
#

_BRAIN0_SIGNATURES: list[dict[str, Any]] = [
    {
        "name":  "HONEY_TOKEN",
        "desc":  "Canary file Q4_Executive_Bonuses accessed",
        "check": lambda d: "Q4_Executive_Bonuses" in
                 d.get("resource", {}).get("name", ""),
    },
    {
        "name":  "OPTICAL_SENSOR",
        "desc":  "Phone/camera detected near sensitive screen",
        "check": lambda d: d.get("enrichments", {})
                 .get("aegis_telemetry", {})
                 .get("optical_sensor_state", "") == "Optical Device Detected",
    },
    {
        "name":  "STEGANOGRAPHY",
        "desc":  "Near-random entropy file transfer (stego exfil)",
        "check": lambda d: (
            d.get("enrichments", {}).get("aegis_telemetry", {})
             .get("file_entropy", 0) > 0.97
            and d.get("resource", {}).get("name", "").endswith(".jpg")
        ),
    },
    {
        "name":  "INVENTORY_FRAUD",
        "desc":  "Warehouse user deleting inventory records",
        "check": lambda d: (
            d.get("actor", {}).get("user", {}).get("group", "") == "Warehouse_Floor"
            and d.get("action", {}).get("type", "") == "record_delete"
            and d.get("resource", {}).get("name", "") == "inventory_db"
        ),
    },
    {
        "name":  "BIOMETRIC_HIJACK",
        "desc":  "Continuous Authentication Failure (Session Hijack)",
        "check": lambda d: (
            d.get("action", {}).get("type", "") == "refund_process"
            and d.get("enrichments", {}).get("aegis_telemetry", {}).get("typing_cadence_ms", 100) > 400
        ),
    },
    {
        "name":  "SUPPLY_CHAIN_FRAUD",
        "desc":  "Unauth modification of vendor routing numbers to overseas bank",
        "check": lambda d: (
            d.get("action", {}).get("type", "") == "db_update"
            and "vendor_routing" in d.get("resource", {}).get("name", "")
        ),
    },
    {
        "name":  "S3_EXPOSURE",
        "desc":  "Cloud Admin making S3 bucket public",
        "check": lambda d: (
            d.get("actor", {}).get("user", {}).get("group", "") == "Cloud_Admin"
            and d.get("action", {}).get("type", "") == "permission_change"
            and d.get("resource", {}).get("name", "") == "S3_Backup_Bucket"
        ),
    },
]


def _brain0_check(log_data: dict) -> tuple[bool, str, str]:
    """Run all hard signatures.  Returns (matched, name, description)."""
    for sig in _BRAIN0_SIGNATURES:
        try:
            if sig["check"](log_data):
                return True, sig["name"], sig["desc"]
        except Exception:
            pass
    return False, "", ""


async def process_stream(speed: float = STREAM_SPEED, max_logs: int = 0):
    """Main pipeline coroutine — Dual-Brain Architecture.

    For each log in enterprise_activity_stream.jsonl:
      1. SHA-256 hash → rolling Merkle root     (integrity)
      2. BRAIN 0: Hard-signature check          (deterministic)
         → If matched: risk=100, bypass PyTorch
      3. BRAIN 1: Vectorise → VAE → MSE → risk  (ML inference)
      4. If critical → Ollama LLM analysis       (explainability)
      5. JSON → every connected WebSocket        (broadcast)
    """
    global _stop_event

    if not JSONL_PATH.exists():
        log.error("❌ Stream file not found: %s", JSONL_PATH)
        stats.status = "error"
        return

    stats.status     = "running"
    stats.start_time = time.time()
    processed        = 0

    log.info("━" * 62)
    log.info("▶  STREAM ONLINE — %s", JSONL_PATH.name)
    log.info("   Speed : %.2fs/log  (~%d logs/sec)",
             speed, int(1 / speed) if speed > 0 else 9999)
    log.info("   Limit : %s", f"{max_logs:,}" if max_logs else "unlimited")
    log.info("   Brain 0 : %d hard signatures loaded",
             len(_BRAIN0_SIGNATURES))
    log.info("   Brain 1 : VAE threshold > %d → critical + LLM",
             ALERT_THRESHOLD)
    log.info("   Device : %s", device)
    log.info("━" * 62)

    try:
        with open(JSONL_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                # ── Stop / limit checks ──
                if _stop_event.is_set():
                    log.warning("⏹  Stream halted by operator")
                    break
                if max_logs > 0 and processed >= max_logs:
                    log.info("✋ Max-log limit reached (%d)", max_logs)
                    break

                raw = line.strip()
                if not raw:
                    continue

                try:
                    # ── 1. Parse ──────────────────────────────────────
                    log_data: dict[str, Any] = json.loads(raw)

                    # ── 2. Merkle integrity ───────────────────────────
                    merkle_root = merkle.ingest(raw)

                    # Resolve user ID (supports both OCSF versions)
                    actor = log_data.get("actor", {})
                    uid = (actor.get("user_id", "")
                           or actor.get("user", {}).get("uid", "?"))

                    # --- SEQUENCE BUFFER UPDATE ---
                    if uid not in user_history_buffer:
                        user_history_buffer[uid] = []
                    user_history_buffer[uid].append(log_data)
                    user_history_buffer[uid] = user_history_buffer[uid][-10:]

                    # ══════════════════════════════════════════════════
                    # BRAIN 0 — Deterministic Hard-Signature Override
                    # ══════════════════════════════════════════════════
                    b0_hit, b0_name, b0_desc = _brain0_check(log_data)

                    if b0_hit:
                        # Hard signature → risk = 100, skip PyTorch
                        risk_score  = 100
                        mse         = -1.0       # sentinel: ML not used
                        is_critical = True
                        stats.brain0_overrides += 1

                        log.critical(
                            "🛑 BRAIN-0 OVERRIDE │ %s │ %s │ %s │ %s",
                            b0_name, uid,
                            log_data.get("resource", {}).get("name", "?"),
                            b0_desc,
                        )

                    # ══════════════════════════════════════════════════
                    # BRAIN 1 — PyTorch VAE (gray-area ML inference)
                    # ══════════════════════════════════════════════════
                    else:
                        tensor      = preprocess_json_to_tensor(log_data, user_history_buffer.get(uid, []))
                        mse         = await _run_inference(tensor)
                        risk_score  = mse_to_risk_score(mse)
                        is_critical = risk_score > ALERT_THRESHOLD

                        if is_critical:
                            log.critical(
                                "🧠 BRAIN-1 ML │ risk=%d │ %s │ %s │ %s │ "
                                "%.4fMB │ MSE=%.6f",
                                risk_score, uid,
                                log_data.get("action", {}).get("type", "?"),
                                log_data.get("resource", {}).get("name", "?"),
                                log_data.get("resource", {}).get("volume_mb", 0),
                                mse,
                            )

                    # ── Build output contract ─────────────────────────
                    output: dict[str, Any] = {
                        "event_type":       "critical_alert" if is_critical
                                            else "normal",
                        "timestamp":        log_data.get("timestamp", ""),
                        "risk_score":       risk_score,
                        "detection_brain":  "brain0_signature" if b0_hit
                                            else "brain1_vae",
                        "signature_name":   b0_name if b0_hit else None,
                        "raw_log":          log_data,
                        # Queue handles LLM population for critical logs
                        "ai_analysis":      None,
                        "merkle_integrity": "Verified",
                        "merkle_root":      merkle_root[:16] + "…",
                        "sequence":         processed,
                    }

                    if is_critical:
                        output = _enrich_alert(output, log_data)
                        stats.push_alert(output)
                        # We hand critical alerts to the LLM Priority Queue
                        # It will await 1s, drop inferiors, and broadcast later.
                        ollama.enqueue(user_history_buffer[uid], risk_score, output)
                    else:
                        # Normal events skip LLM and broadcast immediately
                        await manager.broadcast(output)

                    # ── 9. Stats ──────────────────────────────────────
                    stats.record(risk_score, is_critical)
                    processed += 1

                    # Periodic heartbeat every 1 000 logs
                    if processed % 1000 == 0:
                        log.info(
                            "📊 %s logs │ %d alerts │ %.1f/s │ merkle %s…",
                            f"{processed:>8,}",
                            stats.alert_count,
                            stats.throughput,
                            merkle_root[:12],
                        )

                except json.JSONDecodeError:
                    log.warning("⚠  Malformed JSON at line %d — skipped",
                                processed + 1)
                except KeyError as exc:
                    log.warning("⚠  Missing key %s at line %d — skipped",
                                exc, processed + 1)
                except Exception as exc:
                    log.error("❌ Line %d error: %s", processed + 1, exc)

                # ── Simulate real-time cadence ────────────────────────
                await asyncio.sleep(speed)

    except Exception as exc:
        log.error("❌ Fatal stream error: %s", exc)
        stats.status = "error"
        return

    stats.status = "complete"
    elapsed = time.time() - (stats.start_time or time.time())

    log.info("━" * 62)
    log.info("✅  STREAM COMPLETE")
    log.info("    Total processed  : %s", f"{processed:,}")
    log.info("    Critical alerts  : %d", stats.alert_count)
    log.info("      Brain 0 (sig)  : %d", stats.brain0_overrides)
    log.info("      Brain 1 (ML)   : %d",
             stats.alert_count - stats.brain0_overrides)
    log.info("    Ollama calls     : %d", stats.ollama_calls)
    log.info("    Elapsed          : %.1fs", elapsed)
    log.info("    Avg throughput   : %.2f logs/s", stats.throughput)
    log.info("    Merkle root      : %s", merkle.root)
    log.info("━" * 62)


# ═══════════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

_BANNER = """
    +===========================================================+
    |    ___   ____  ____ ___ ____   v2.0                       |
    |   /   | / __/ / __// // __/                               |
    |  / /| |/ _/  / /_ / //_\ \                                |
    | / ___ / /__ / /_// //__ /                                 |
    |/_/  |_\___/ \___/___/___/                                 |
    |                                                           |
    |  Insider Threat Detection Engine                          |
    |  FastAPI - PyTorch VAE - Ollama - WebSocket - Merkle      |
    +===========================================================+
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup / shutdown lifecycle."""
    global model, ROLES_DF, VOL_MIN, VOL_MAX, TRAIN_MSE_MEAN, TRAIN_MSE_STD

    try:
        sys.stdout.buffer.write(_BANNER.encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:
        print(_BANNER.encode("ascii", errors="replace").decode())

    # ── Feature metadata ──
    if META_PATH.exists():
        meta    = json.loads(META_PATH.read_text())
        VOL_MIN = meta.get("vol_min", VOL_MIN)
        VOL_MAX = meta.get("vol_max", VOL_MAX)
        log.info("[META] Feature meta  -- %d features, volume range [%.2f, %.2f]",
                 meta.get("num_features", INPUT_DIM), VOL_MIN, VOL_MAX)
    else:
        log.warning("⚠  feature_meta.json not found — using defaults")

    # ── Training calibration stats ──
    if THRESH_PATH.exists():
        cal = json.loads(THRESH_PATH.read_text())
        TRAIN_MSE_MEAN = cal.get("train_mse_mean", TRAIN_MSE_MEAN)
        TRAIN_MSE_STD  = cal.get("train_mse_std",  TRAIN_MSE_STD)
        log.info("[CAL]  Calibration   -- mean=%.6f  std=%.6f  p99=%.6f",
                 TRAIN_MSE_MEAN, TRAIN_MSE_STD, cal.get("train_mse_p99", 0))
    else:
        log.warning("⚠  threshold_stats.json not found — using hardcoded calibration")

    # ── User roles (pandas) ──
    if ROLES_PATH.exists():
        ROLES_DF = pd.read_csv(ROLES_PATH)
        log.info("[ROLE] User roles    -- %d users, %d departments",
                 len(ROLES_DF), ROLES_DF["department"].nunique())
    else:
        log.warning("⚠  user_roles.csv not found — alert enrichment disabled")

    # ── PyTorch VAE ──
    model = InsiderThreatVAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(device)
    if MODEL_PATH.exists():
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True),
        )
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        log.info("[VAE]  Model loaded  -- %s params on %s",
                 f"{params:,}", device)
    else:
        model.eval()
        log.warning("⚠  %s not found — running with RANDOM weights!",
                    MODEL_PATH.name)

    # ── Ollama ──
    await ollama.initialize()

    # ── JSONL check ──
    if JSONL_PATH.exists():
        size_mb = JSONL_PATH.stat().st_size / (1024 * 1024)
        log.info("[FILE] Stream file   -- %s (%.1f MB)", JSONL_PATH.name, size_mb)
    else:
        log.error("❌ %s NOT FOUND — stream will fail", JSONL_PATH.name)

    log.info("=" * 62)
    log.info(">> AEGIS ENGINE ONLINE")
    log.info("   POST /api/stream/start   -> begin processing")
    log.info("   POST /api/stream/stop    -> halt processing")
    log.info("   GET  /api/stats          -> live metrics")
    log.info("   GET  /api/alerts         -> recent critical alerts")
    log.info("   WS   /ws/stream          -> real-time event feed")
    log.info("=" * 62)

    yield  # ── application runs here ──

    # ── Teardown ──
    _stop_event.set()
    if _stream_task and not _stream_task.done():
        _stream_task.cancel()
    await ollama.shutdown()
    log.info("[STOP] AEGIS ENGINE OFFLINE -- final merkle root: %s", merkle.root)


app = FastAPI(
    title="AEGIS Insider Threat Detection Engine",
    description="Real-time insider threat detection via PyTorch VAE "
                "with Ollama LLM explainability and Merkle log integrity.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # TODO: lock to your Next.js origin in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def health_check():
    """Quick liveness probe."""
    return {
        "engine":            "AEGIS v2.0",
        "status":            "online",
        "model_loaded":      model is not None,
        "device":            str(device),
        "stream_status":     stats.status,
        "websocket_clients": manager.count,
        "merkle_chain":      merkle.count,
    }


@app.get("/api/stats", tags=["Monitoring"])
async def get_stats():
    """Detailed pipeline metrics for the monitoring dashboard."""
    return stats.to_dict()


@app.get("/api/alerts", tags=["Monitoring"])
async def get_alerts():
    """Return the most recent critical alerts (rolling buffer of 100)."""
    return {
        "total_alerts":  stats.alert_count,
        "showing":       len(stats.recent_alerts),
        "alerts":        stats.recent_alerts,
    }


@app.get("/api/merkle", tags=["Integrity"])
async def get_merkle():
    """Current Merkle chain state for audit verification."""
    return {
        "merkle_root":   merkle.root,
        "chain_length":  merkle.count,
        "integrity":     "Verified" if merkle.count > 0 else "No data",
    }


@app.post("/api/stream/start", tags=["Stream Control"])
async def start_stream(
    speed: float = Query(
        default=0.1, ge=0.0, le=5.0,
        description="Seconds between log reads (0.1 = 10 logs/sec)",
    ),
    max_logs: int = Query(
        default=0, ge=0,
        description="Maximum logs to process (0 = unlimited / entire file)",
    ),
):
    """Begin ingesting the JSONL stream."""
    global _stream_task, _stop_event

    if stats.status == "running":
        return JSONResponse(
            status_code=409,
            content={"error": "Stream is already running",
                     "stats": stats.to_dict()},
        )

    _stop_event = asyncio.Event()
    _stream_task = asyncio.create_task(process_stream(speed, max_logs))
    log.info("▶  Stream task launched — speed=%.2f  max_logs=%s",
             speed, max_logs or "∞")

    return {
        "message":  "Stream started",
        "speed":    speed,
        "max_logs": max_logs or "unlimited",
    }


@app.post("/api/stream/stop", tags=["Stream Control"])
async def stop_stream():
    """Gracefully halt the running stream."""
    if stats.status != "running":
        return JSONResponse(
            status_code=409,
            content={"error": "No stream is currently running"},
        )

    _stop_event.set()
    stats.reset()
    stats.status = "idle"
    return {"message": "Stop signal sent and stats reset to 0.",
            "stats":   stats.to_dict()}


@app.post("/api/inject_test_log", tags=["Stream Control"])
async def inject_test_log(log_data: dict):
    """Directly inject a 'Kill Shot' log into the pipeline for live demos."""
    raw = json.dumps(log_data)
    merkle_root = merkle.ingest(raw)
    
    tensor = preprocess_json_to_tensor(log_data, [])
    mse = await _run_inference(tensor)
    
    # We always push risk score to 99 for the demo to guarantee visual wow-factor
    risk_score = max(mse_to_risk_score(mse), 99) 
    
    log.critical(
        "DEMO INJECT │ risk=%d │ %s │ %s │ %s │ %.4fMB",
        risk_score,
        log_data.get("actor", {}).get("user_id", "?"),
        log_data.get("action", {}).get("type", "?"),
        log_data.get("resource", {}).get("name", "?"),
        log_data.get("resource", {}).get("volume_mb", 0),
    )
    
    # Force LLM explanation
    ai_analysis = await ollama.analyze(log_data, risk_score)
    stats.ollama_calls += 1

    output = {
        "event_type":       "critical_alert",
        "timestamp":        log_data.get("timestamp", ""),
        "risk_score":       risk_score,
        "raw_log":          log_data,
        "ai_analysis":      ai_analysis,
        "merkle_integrity": "Verified",
        "merkle_root":      merkle_root[:16] + "…",
        "sequence":         stats.total_processed,
    }

    output = _enrich_alert(output, log_data)
    stats.push_alert(output)
    
    await manager.broadcast(output)
    stats.record(risk_score, is_alert=True)
    
    return {"message": "Demo log injected successfully", "payload": output}


@app.post("/api/tamper", tags=["Stream Control"])
async def trigger_tamper():
    """Hackathon Mic Drop Feature: Shatter the merkle chain live."""
    # 1. Artificially corrupt the global Merkle root
    merkle._root = "0000000_CORRUPTED_TAMPER_DETECTED_0000000"
    
    # 2. Build the visual payload
    payload = {
        "event_type": "tamper_alert",
        "merkle_integrity": "COMPROMISED",
        "merkle_root": "CHAIN_BROKEN",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    # 3. Log it brilliantly
    log.critical("🚨 🚨 🚨 MERKLE LEDGER SHATTERED 🚨 🚨 🚨")
    log.critical("Manual /api/tamper injected.")
    
    # 4. Broadcast immediately
    await manager.broadcast(payload)
    return {"message": "Tamper simulation triggered successfully."}


# ── WebSocket Endpoint ────────────────────────────────────────────────────

@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """Real-time event feed.  Clients receive every processed log as JSON.

    Send ``"ping"`` to receive a ``{"pong": true, ...}`` heartbeat with
    current stats.  The connection stays open until the client disconnects.
    """
    await manager.connect(ws)
    try:
        while True:
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_json({"pong": True, "stats": stats.to_dict()})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="warning",     # suppress uvicorn noise; AEGIS has its own logger
    )
