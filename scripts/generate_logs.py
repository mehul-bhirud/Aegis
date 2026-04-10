"""
generate_logs.py — Aegis-Fusion OCSF Activity Stream Generator v3.0
====================================================================
Produces 500,000 OCSF-aligned nested JSON logs for the Aegis-Fusion
insider threat detection platform.

Distribution:
  480,000  Baseline    — Normal corporate behaviour
   15,000  Noise       — Benign mistakes (failed logins, blocked sites)
    5,000  Kill Shots  — 5 advanced insider threat scenarios × 1,000 each

Kill-Shot Scenarios:
  1. Analog Hole         — Optical device detected near sensitive screen
  2. Steganography       — High-entropy .jpg exfiltration
  3. Retail Fraud        — Warehouse user deleting inventory records
  4. Active Deception    — Honey-token file access (Q4_Executive_Bonuses)
  5. Shadow Admin (LotL) — S3 bucket permission change to Public

Output: enterprise_activity_stream.jsonl  (sorted by timestamp)

Usage:
    python generate_logs.py

Dependencies:
    Python 3.10+ (stdlib only — no external packages required)
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import chain, repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows terminals (box-drawing chars crash cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "enterprise_activity_stream.jsonl"
ROLES_CSV = ROOT / "user_roles.csv"

TOTAL        = 500_000
BASELINE     = 480_000
NOISE        =  15_000
KILLSHOT     =   5_000
KS_PER_SCENE =   1_000          # 5 scenarios × 1,000

CHUNK_SIZE   = 50_000            # logs per multiprocessing chunk
NUM_WORKERS  = max(1, cpu_count() - 1)

# Timeline: March 2026 (31 days)
T_START = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
T_END   = datetime(2026, 3, 31, 23, 59, 59, tzinfo=timezone.utc)
T_SPAN  = (T_END - T_START).total_seconds()


# ═══════════════════════════════════════════════════════════════════════════
#  USER POOL  (200 employees across 12 departments)
# ═══════════════════════════════════════════════════════════════════════════

GROUP_SIZES: dict[str, int] = {
    "Junior_Marketing":      25,
    "Senior_Marketing":      15,
    "Warehouse_Floor":       30,
    "Cloud_Admin":           10,
    "IT_Support":            20,
    "Corporate_Finance":     15,
    "Corporate_HR":          15,
    "Supply_Chain_Ops":      25,
    "Retail_Store_Ops":      20,
    "Data_Analytics":        10,
    "Legal_Compliance":       8,
    "Executive_Leadership":   7,
}

USERS: list[dict[str, str]] = []
_uid = 1
for _grp, _cnt in GROUP_SIZES.items():
    for _ in range(_cnt):
        USERS.append({"uid": f"emp_{_uid:04d}", "group": _grp})
        _uid += 1

# Pre-filtered pools for scenario-specific actor selection
_WAREHOUSE = [u for u in USERS if u["group"] == "Warehouse_Floor"]
_CLOUD     = [u for u in USERS if u["group"] == "Cloud_Admin"]
_TECHNICAL = [u for u in USERS if u["group"] in
              ("Cloud_Admin", "IT_Support", "Data_Analytics")]


# ═══════════════════════════════════════════════════════════════════════════
#  VOCABULARIES
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_ACTIONS: list[tuple[str, str]] = [
    ("Network", "login"),
    ("Network", "vpn_connect"),
    ("Data",    "db_query"),
    ("Data",    "file_download"),
    ("Data",    "file_copy"),
    ("System",  "config_change"),
    ("System",  "usb_mount"),
    ("IAM",     "mfa_enroll"),
]

NOISE_SCENARIOS: list[tuple[str, str, str]] = [
    # (category, type, status)
    ("Network", "login",         "failed"),       # wrong password
    ("Data",    "file_download", "blocked"),       # insufficient perms
    ("System",  "usb_mount",     "blocked"),       # unapproved device
    ("Network", "vpn_connect",   "failed"),        # expired cert
    ("IAM",     "mfa_enroll",    "failed"),        # timeout
    ("Data",    "db_query",      "failed"),        # query denied
]

RESOURCES: dict[str, list[str]] = {
    "login":             ["email_gateway", "erp_portal", "hr_portal",
                          "crm_system", "timesheet_app"],
    "vpn_connect":       ["corp_vpn_primary", "corp_vpn_backup"],
    "db_query":          ["product_catalog_db", "sales_reporting_db",
                          "hr_records_db", "logistics_tracking_db"],
    "file_download":     ["quarterly_report.pdf", "team_handbook.docx",
                          "training_slides.pptx", "policy_update.pdf",
                          "vendor_contract.pdf"],
    "file_copy":         ["meeting_notes.docx", "project_brief.pdf",
                          "expense_report.xlsx", "onboarding_pack.zip"],
    "config_change":     ["firewall_rule_set", "dns_config",
                          "proxy_settings", "log_rotation_policy"],
    "usb_mount":         ["approved_backup_drive", "firmware_update_key"],
    "mfa_enroll":        ["authenticator_app", "hardware_key"],
    "record_delete":     ["temp_cache_data", "expired_tokens",
                          "old_session_logs"],
    "permission_change": ["shared_drive_access", "team_folder_perms",
                          "ci_cd_pipeline_role"],
    "process_kill":      ["hung_browser_process", "stale_print_spooler"],
}

SENSITIVITY: dict[str, str] = {
    "email_gateway": "Internal",      "erp_portal": "Internal",
    "hr_portal": "PII_RESTRICTED",    "crm_system": "Internal",
    "timesheet_app": "Internal",
    "corp_vpn_primary": "Internal",   "corp_vpn_backup": "Internal",
    "product_catalog_db": "Public",   "sales_reporting_db": "Confidential",
    "hr_records_db": "PII_RESTRICTED","logistics_tracking_db": "Internal",
    "quarterly_report.pdf": "Internal",
    "team_handbook.docx": "Public",   "training_slides.pptx": "Public",
    "policy_update.pdf": "Internal",  "vendor_contract.pdf": "Confidential",
    "meeting_notes.docx": "Internal", "project_brief.pdf": "Internal",
    "expense_report.xlsx": "Confidential",
    "onboarding_pack.zip": "Internal",
    "firewall_rule_set": "Confidential", "dns_config": "Internal",
    "proxy_settings": "Internal",     "log_rotation_policy": "Internal",
    "approved_backup_drive": "Internal",
    "firmware_update_key": "Internal",
    "authenticator_app": "Internal",  "hardware_key": "Internal",
    "temp_cache_data": "Public",      "expired_tokens": "Internal",
    "old_session_logs": "Internal",
    "shared_drive_access": "Internal","team_folder_perms": "Internal",
    "ci_cd_pipeline_role": "Confidential",
    "hung_browser_process": "Public", "stale_print_spooler": "Public",
    # ── Kill-shot resources ──
    "customer_loyalty_db":              "PII_RESTRICTED",
    "encoded_product_shots.jpg":        "Internal",
    "inventory_db":                     "Confidential",
    "Q4_Executive_Bonuses_2026.xlsx":   "Confidential",
    "S3_Backup_Bucket":                 "Confidential",
}

LOCATIONS = ["Pune", "Bangalore", "Mumbai", "Singapore"]

VOLUME_RANGE: dict[str, tuple[float, float]] = {
    "login":             (0.01,   0.1),
    "vpn_connect":       (0.01,  0.05),
    "db_query":          (0.1,   50.0),
    "file_download":     (0.5,  500.0),
    "file_copy":         (0.5,  200.0),
    "config_change":     (0.01,   0.5),
    "usb_mount":         (0.01,  0.05),
    "mfa_enroll":        (0.01,  0.05),
    "record_delete":     (0.01,   0.1),
    "permission_change": (0.01,   0.1),
    "process_kill":      (0.001, 0.01),
}


# ═══════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _event_id(ts_str: str, uid: str, atype: str, salt: int) -> str:
    """SHA-256 truncated to 24 hex chars — deterministic + unique."""
    raw = f"{ts_str}|{uid}|{atype}|{salt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _merkle_leaf(ts_str: str, uid: str, salt: int) -> str:
    """Dummy SHA-256 placeholder — the backend recalculates the real root."""
    raw = f"leaf|{ts_str}|{uid}|{salt}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _session_id(uid: str, dt: datetime) -> str:
    """Deterministic session ID: same user + same date + same 4-hour block
    always produces the same session hash.

    Blocks: 00-03:59 │ 04-07:59 │ 08-11:59 │ 12-15:59 │ 16-19:59 │ 20-23:59
    """
    block = dt.hour // 4
    raw = f"{uid}|{dt.strftime('%Y-%m-%d')}|{block}"
    return "sess_" + hashlib.md5(raw.encode()).hexdigest()[:8]


def _ip(rng: random.Random, location: str) -> str:
    """Generate a deterministic-looking internal IP per location."""
    prefix = {"Pune": (10, 1), "Bangalore": (10, 2),
              "Mumbai": (10, 3), "Singapore": (10, 4),
              "Unknown": (192, 168)}
    a, b = prefix.get(location, (10, 99))
    return f"{a}.{b}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"


# ═══════════════════════════════════════════════════════════════════════════
#  CORE LOG ASSEMBLER
# ═══════════════════════════════════════════════════════════════════════════

def _assemble(
    ts: datetime, salt: int,
    user: dict[str, str],
    category: str, atype: str, status: str,
    resource: str, volume: float,
    location: str, ip_addr: str,
    edr: bool,
    mfa: str,
    entropy: float, typing_var: float, optical: str,
) -> dict[str, Any]:
    """Central factory — every builder calls this to avoid code duplication."""
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"
    return {
        "event_id":   _event_id(ts_str, user["uid"], atype, salt),
        "timestamp":  ts_str,
        "session_id": _session_id(user["uid"], ts),
        "metadata": {
            "version":          "1.0",
            "product":          "aegis-fusion",
            "merkle_leaf_hash": _merkle_leaf(ts_str, user["uid"], salt),
        },
        "actor": {
            "user": {"uid": user["uid"], "group": user["group"]},
            "mfa_status": mfa,
        },
        "action": {
            "category": category,
            "type":     atype,
            "status":   status,
        },
        "resource": {
            "name":              resource,
            "sensitivity_label": SENSITIVITY.get(resource, "Internal"),
            "volume_mb":         round(volume, 3),
        },
        "context": {
            "location":         location,
            "ip_address":       ip_addr,
            "edr_agent_active": edr,
        },
        "enrichments": {
            "aegis_telemetry": {
                "file_entropy":            round(entropy, 4),
                "typing_cadence_variance": round(typing_var, 4),
                "optical_sensor_state":    optical,
            }
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  TIER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _build_baseline(rng: random.Random, ts: datetime, salt: int) -> dict:
    """Normal corporate behaviour — nothing suspicious."""
    user = rng.choice(USERS)
    cat, atype = rng.choice(BASELINE_ACTIONS)
    res  = rng.choice(RESOURCES.get(atype, ["unknown"]))
    loc  = rng.choice(LOCATIONS)
    lo, hi = VOLUME_RANGE.get(atype, (0.1, 10.0))
    return _assemble(
        ts, salt, user, cat, atype, "success", res,
        volume   = rng.uniform(lo, hi),
        location = loc,
        ip_addr  = _ip(rng, loc),
        edr      = True,
        mfa      = "success",
        entropy  = rng.uniform(0.01, 0.65),
        typing_var = max(0.0, rng.gauss(0.12, 0.03)),
        optical  = "Clear",
    )


def _build_noise(rng: random.Random, ts: datetime, salt: int) -> dict:
    """Benign mistakes — failed logins, blocked USB, denied queries.
    These should NOT trigger a critical alert from the model."""
    user = rng.choice(USERS)
    cat, atype, status = rng.choice(NOISE_SCENARIOS)
    res = rng.choice(RESOURCES.get(atype, ["unknown"]))
    loc = rng.choice(LOCATIONS + ["Unknown"])  # occasional unknown location
    lo, hi = VOLUME_RANGE.get(atype, (0.1, 10.0))
    # Failed actions → MFA may have failed too
    mfa = rng.choice(["failed", "success", "success"]) if status == "failed" \
          else "success"
    return _assemble(
        ts, salt, user, cat, atype, status, res,
        volume   = rng.uniform(lo, hi),
        location = loc,
        ip_addr  = _ip(rng, loc),
        edr      = True,
        mfa      = mfa,
        entropy  = rng.uniform(0.01, 0.50),
        typing_var = max(0.0, rng.gauss(0.18, 0.05)),  # slightly elevated
        optical  = "Clear",
    )


# ───────────────────────────────────────────────────────────────────
#  KILL-SHOT 1 — The Analog Hole
#  An employee uses a phone/camera to photograph PII on screen.
#  Signal: optical_sensor_state = "Optical Device Detected"
#          + accessing customer_loyalty_db
# ───────────────────────────────────────────────────────────────────
def _build_ks_analog_hole(rng: random.Random, ts: datetime, salt: int) -> dict:
    user = rng.choice(USERS)           # any department
    return _assemble(
        ts, salt, user, "Data", "db_query", "success",
        resource = "customer_loyalty_db",
        volume   = rng.uniform(0.5, 45.0),
        location = rng.choice(LOCATIONS),
        ip_addr  = _ip(rng, rng.choice(LOCATIONS)),
        edr      = True,
        mfa      = "success",
        entropy  = rng.uniform(0.20, 0.60),
        typing_var = max(0.0, rng.gauss(0.22, 0.06)),
        optical  = "Optical Device Detected",             # ← THE SIGNAL
    )


# ───────────────────────────────────────────────────────────────────
#  KILL-SHOT 2 — Steganography
#  Data hidden inside a .jpg with near-random entropy.
#  Signal: file_entropy > 0.98 on a .jpg transfer
# ───────────────────────────────────────────────────────────────────
def _build_ks_stegano(rng: random.Random, ts: datetime, salt: int) -> dict:
    user = rng.choice(_TECHNICAL)      # technical users only
    atype = rng.choice(["file_copy", "file_download"])
    return _assemble(
        ts, salt, user, "Data", atype, "success",
        resource = "encoded_product_shots.jpg",
        volume   = rng.uniform(8.0, 55.0),                # suspiciously large for a JPG
        location = rng.choice(LOCATIONS),
        ip_addr  = _ip(rng, rng.choice(LOCATIONS)),
        edr      = True,
        mfa      = "success",
        entropy  = round(rng.uniform(0.981, 0.999), 4),   # ← THE SIGNAL
        typing_var = max(0.0, rng.gauss(0.11, 0.02)),     # deliberately calm
        optical  = "Clear",
    )


# ───────────────────────────────────────────────────────────────────
#  KILL-SHOT 3 — Retail Fraud
#  Warehouse worker deleting inventory records to cover theft.
#  Signal: Warehouse_Floor + record_delete + inventory_db
# ───────────────────────────────────────────────────────────────────
def _build_ks_retail_fraud(rng: random.Random, ts: datetime, salt: int) -> dict:
    user = rng.choice(_WAREHOUSE)      # must be warehouse
    return _assemble(
        ts, salt, user, "Data", "record_delete", "success",
        resource = "inventory_db",
        volume   = rng.uniform(0.01, 0.08),
        location = rng.choice(["Pune", "Mumbai"]),         # warehouses in Pune/Mumbai
        ip_addr  = _ip(rng, "Pune"),
        edr      = True,
        mfa      = "success",
        entropy  = rng.uniform(0.05, 0.30),
        typing_var = max(0.0, rng.gauss(0.35, 0.08)),     # nervous / erratic
        optical  = "Clear",
    )


# ───────────────────────────────────────────────────────────────────
#  KILL-SHOT 4 — Active Deception (Honey Token)
#  A canary file that should NEVER be accessed by real employees.
#  Signal: resource = "Q4_Executive_Bonuses_2026.xlsx"
# ───────────────────────────────────────────────────────────────────
def _build_ks_honey_token(rng: random.Random, ts: datetime, salt: int) -> dict:
    user  = rng.choice(USERS)          # any curious employee
    atype = rng.choice(["file_download", "file_copy"])
    return _assemble(
        ts, salt, user, "Data", atype, "success",
        resource = "Q4_Executive_Bonuses_2026.xlsx",       # ← THE SIGNAL
        volume   = rng.uniform(1.0, 4.5),
        location = rng.choice(LOCATIONS),
        ip_addr  = _ip(rng, rng.choice(LOCATIONS)),
        edr      = True,
        mfa      = "success",
        entropy  = rng.uniform(0.30, 0.65),
        typing_var = max(0.0, rng.gauss(0.14, 0.04)),
        optical  = "Clear",
    )


# ───────────────────────────────────────────────────────────────────
#  KILL-SHOT 5 — Shadow Admin / Living off the Land
#  Cloud Admin changes S3 bucket ACL from Private → Public.
#  Signal: Cloud_Admin + permission_change + S3_Backup_Bucket
# ───────────────────────────────────────────────────────────────────
def _build_ks_shadow_admin(rng: random.Random, ts: datetime, salt: int) -> dict:
    user = rng.choice(_CLOUD)          # must be Cloud_Admin
    log = _assemble(
        ts, salt, user, "IAM", "permission_change", "success",
        resource = "S3_Backup_Bucket",
        volume   = rng.uniform(0.01, 0.05),
        location = rng.choice(LOCATIONS + ["Unknown"]),    # sometimes unknown VPN
        ip_addr  = _ip(rng, "Unknown"),
        edr      = rng.choice([True, True, False]),        # sometimes EDR off
        mfa      = rng.choice(["success", "success", "bypassed"]),
        entropy  = rng.uniform(0.01, 0.15),
        typing_var = max(0.0, rng.gauss(0.10, 0.02)),
        optical  = "Clear",
    )
    # Add the permission detail that makes this a kill shot
    log["resource"]["permission_before"] = "Private"
    log["resource"]["permission_after"]  = "Public"
    return log


# ═══════════════════════════════════════════════════════════════════════════
#  DISPATCH TABLE
# ═══════════════════════════════════════════════════════════════════════════

_BUILDERS = {
    0: _build_baseline,
    1: _build_noise,
    2: _build_ks_analog_hole,
    3: _build_ks_stegano,
    4: _build_ks_retail_fraud,
    5: _build_ks_honey_token,
    6: _build_ks_shadow_admin,
}

# Tier labels:
#   0 = baseline, 1 = noise
#   2-6 = kill-shot scenarios (Analog, Stegano, Fraud, Honey, Shadow)


# ═══════════════════════════════════════════════════════════════════════════
#  MULTIPROCESSING WORKER
# ═══════════════════════════════════════════════════════════════════════════

def _generate_chunk(args: tuple) -> list[str]:
    """Worker function — receives a chunk of (ts_offset, tier, salt) tuples,
    generates full JSON log lines, returns them as a list of strings.

    Each worker gets its own Random instance seeded deterministically
    so results are reproducible across runs.
    """
    chunk_data: list[tuple[float, int, int]]
    seed: int
    chunk_data, seed = args

    rng = random.Random(seed)
    lines: list[str] = []

    for ts_offset, tier, salt in chunk_data:
        ts = T_START + timedelta(seconds=ts_offset)
        builder = _BUILDERS[tier]
        log = builder(rng, ts, salt)
        lines.append(json.dumps(log, ensure_ascii=False, separators=(",", ":")))

    return lines


# ═══════════════════════════════════════════════════════════════════════════
#  USER ROLES CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def _export_user_roles() -> None:
    """Write user_roles.csv matching the new schema for downstream use."""
    with open(ROLES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "department", "expected_resources"])
        for u in USERS:
            # Assign plausible expected resources per group
            expected = {
                "Junior_Marketing":     "crm_system,email_gateway",
                "Senior_Marketing":     "crm_system,email_gateway,sales_reporting_db",
                "Warehouse_Floor":      "logistics_tracking_db,erp_portal",
                "Cloud_Admin":          "firewall_rule_set,dns_config,S3_Backup_Bucket",
                "IT_Support":           "erp_portal,firewall_rule_set,proxy_settings",
                "Corporate_Finance":    "sales_reporting_db,expense_report.xlsx",
                "Corporate_HR":         "hr_portal,hr_records_db",
                "Supply_Chain_Ops":     "logistics_tracking_db,vendor_contract.pdf",
                "Retail_Store_Ops":     "product_catalog_db,erp_portal",
                "Data_Analytics":       "sales_reporting_db,product_catalog_db",
                "Legal_Compliance":     "policy_update.pdf,vendor_contract.pdf",
                "Executive_Leadership": "sales_reporting_db,quarterly_report.pdf",
            }.get(u["group"], "erp_portal")
            writer.writerow([u["uid"], u["group"], expected])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║  AEGIS-FUSION  Log Generator v3.0                           ║
╠══════════════════════════════════════════════════════════════╣
║  Total logs     : {TOTAL:>10,}                              ║
║  Baseline       : {BASELINE:>10,}                              ║
║  Noise          : {NOISE:>10,}                              ║
║  Kill Shots     : {KILLSHOT:>10,}   (5 × {KS_PER_SCENE:,})               ║
║  Workers        : {NUM_WORKERS:>10}                              ║
║  Chunk size     : {CHUNK_SIZE:>10,}                              ║
║  Timeline       : {T_START.strftime('%Y-%m-%d')} → {T_END.strftime('%Y-%m-%d')}                 ║
║  Output         : {OUTPUT.name:<40s} ║
╚══════════════════════════════════════════════════════════════╝"""
    print(banner)

    t0 = time.perf_counter()
    rng = random.Random(42)

    # ── Step 1: Generate sorted timestamps ─────────────────────────────
    print("\n[1/5] Generating timestamps …", end=" ", flush=True)
    all_ts = sorted(rng.uniform(0, T_SPAN) for _ in range(TOTAL))
    print(f"done ({len(all_ts):,} offsets)")

    # ── Step 2: Assign tier labels ─────────────────────────────────────
    print("[2/5] Assigning tier labels …", end=" ", flush=True)
    # Build label array: 480k×0, 15k×1, 1k×2, 1k×3, 1k×4, 1k×5, 1k×6
    tier_labels: list[int] = list(chain(
        repeat(0, BASELINE),
        repeat(1, NOISE),
        repeat(2, KS_PER_SCENE),
        repeat(3, KS_PER_SCENE),
        repeat(4, KS_PER_SCENE),
        repeat(5, KS_PER_SCENE),
        repeat(6, KS_PER_SCENE),
    ))
    assert len(tier_labels) == TOTAL, f"Label count mismatch: {len(tier_labels)}"
    rng.shuffle(tier_labels)
    print("done")

    # ── Step 3: Build assignment tuples (ts_offset, tier, salt) ────────
    print("[3/5] Building assignment index …", end=" ", flush=True)
    assignments: list[tuple[float, int, int]] = [
        (all_ts[i], tier_labels[i], i) for i in range(TOTAL)
    ]
    # Timestamps are already sorted → assignments are in chronological order
    # with tiers scattered randomly across the timeline — exactly what we want.
    del all_ts, tier_labels            # free ~12 MB

    # Count distribution for verification
    tier_counts = [0] * 7
    for _, t, _ in assignments:
        tier_counts[t] += 1
    print(f"done  [B={tier_counts[0]:,} N={tier_counts[1]:,} "
          f"KS={sum(tier_counts[2:]):,}]")

    # ── Step 4: Chunk & multiprocess ───────────────────────────────────
    print(f"[4/5] Generating logs ({NUM_WORKERS} workers × "
          f"{CHUNK_SIZE:,} chunk) …")

    chunks = []
    for i in range(0, TOTAL, CHUNK_SIZE):
        chunk_slice = assignments[i : i + CHUNK_SIZE]
        chunk_seed  = 1337 + i // CHUNK_SIZE
        chunks.append((chunk_slice, chunk_seed))
    del assignments                     # free ~28 MB

    written = 0
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        with Pool(processes=NUM_WORKERS) as pool:
            for chunk_lines in pool.imap(_generate_chunk, chunks):
                for line in chunk_lines:
                    fh.write(line)
                    fh.write("\n")
                written += len(chunk_lines)
                pct = written / TOTAL * 100
                print(f"      ✓ {written:>9,} / {TOTAL:,}  ({pct:5.1f}%)",
                      flush=True)
                del chunk_lines        # free chunk RAM immediately

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # ── Step 5: Export user_roles.csv ──────────────────────────────────
    print("[5/5] Exporting user_roles.csv …", end=" ", flush=True)
    _export_user_roles()
    print(f"done ({len(USERS)} users)")

    # ── Summary ────────────────────────────────────────────────────────
    size_mb = OUTPUT.stat().st_size / (1024 * 1024)
    rate    = TOTAL / elapsed if elapsed > 0 else 0

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  GENERATION COMPLETE                                        ║
╠══════════════════════════════════════════════════════════════╣
║  File     : {OUTPUT.name:<46s} ║
║  Size     : {size_mb:>8.1f} MB                                       ║
║  Rows     : {written:>10,}                                    ║
║  Time     : {elapsed:>8.1f}s                                        ║
║  Speed    : {rate:>10,.0f} logs/sec                               ║
╠══════════════════════════════════════════════════════════════╣
║  Tier Breakdown:                                            ║
║    Baseline (0)        : {tier_counts[0]:>10,}                       ║
║    Noise    (1)        : {tier_counts[1]:>10,}                       ║
║    KS Analog Hole  (2) : {tier_counts[2]:>10,}                       ║
║    KS Steganography(3) : {tier_counts[3]:>10,}                       ║
║    KS Retail Fraud (4) : {tier_counts[4]:>10,}                       ║
║    KS Honey Token  (5) : {tier_counts[5]:>10,}                       ║
║    KS Shadow Admin (6) : {tier_counts[6]:>10,}                       ║
╚══════════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # multiprocessing on Windows requires the spawn guard
    main()
