"""
preprocess.py — Aegis-Fusion Session-Level Feature Engineering v3.1
===================================================================
Converts raw enterprise_activity_stream.jsonl into SESSION-level
PyTorch tensors for the VAE anomaly detector.

Key Innovation:  Instead of treating each log as an independent row,
we GROUP logs by session_id (4-hour user windows) and produce a
single 61-dimensional "mathematical fingerprint" per session.

v3.1 change:  Added 5 Binary Threat Flags (max-pooled) so that a
single kill-shot log in a 13-log session is NOT diluted away.

Pipeline:
  1. Parse 500k JSONL logs → flat DataFrame
  2. Cyclical time encoding  (sin/cos so 11 PM ≈ 1 AM)
  3. Detect kill-shot patterns → flag anomalous sessions
  4. Indicator columns + Binary Threat Flags
  5. GroupBy session_id → aggregate into 61-dim vectors
  6. Split: normal sessions → train, kill-shot sessions → test
  7. MinMax normalise using train-only statistics
  8. Save train_tensor.pt, test_tensor.pt, feature_meta.json

Feature Vector Layout (61 dimensions per session):
  ┌───────────┬────────────────────────────────────────────────┐
  │ [0–3]     │ Temporal:  hour_sin(μ,σ), hour_cos(μ,σ)       │
  │ [4–5]     │ Session:   duration_s, log_count               │
  │ [6–10]    │ Velocity:  delta(μ,σ,min,max), velocity_lps    │
  │ [11–13]   │ Volume:    total, max, mean                    │
  │ [14–24]   │ Action-type fractions       (11 categories)    │
  │ [25–36]   │ User-group one-hot          (12 departments)   │
  │ [37–40]   │ Sensitivity fractions       (4 levels)         │
  │ [41–45]   │ Location fractions          (5 cities)         │
  │ [46–48]   │ MFA-status fractions        (3 states)         │
  │ [49–50]   │ Entropy:   max, mean                           │
  │ [51–52]   │ Typing cadence: mean, max                      │
  │ [53]      │ Optical-device-detected ratio                  │
  │ [54]      │ EDR-inactive ratio                             │
  │ [55]      │ Action-failed/blocked ratio                    │
  │ [56]      │ 🚩 FLAG: Honey-token access       (max-pool)  │
  │ [57]      │ 🚩 FLAG: Destructive action       (max-pool)  │
  │ [58]      │ 🚩 FLAG: Critical resource access  (max-pool)  │
  │ [59]      │ 🚩 FLAG: Optical sensor triggered  (max-pool)  │
  │ [60]      │ 🚩 FLAG: High-entropy file xfer   (max-pool)  │
  └───────────┴────────────────────────────────────────────────┘

Usage:  python preprocess.py
Deps:   pandas, numpy, torch  (pip install pandas numpy torch)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════

ROOT          = Path(__file__).parent
JSONL_PATH    = ROOT / "enterprise_activity_stream.jsonl"
TRAIN_TENSOR  = ROOT / "train_tensor.pt"
TEST_TENSOR   = ROOT / "test_tensor.pt"
TEST_SESSIONS = ROOT / "test_anomalies.jsonl"
META_PATH     = ROOT / "feature_meta.json"


# ═══════════════════════════════════════════════════════════════════════════
#  FIXED VOCABULARIES  (must match generate_logs.py exactly)
# ═══════════════════════════════════════════════════════════════════════════

ACTION_TYPES = [
    "login", "vpn_connect", "db_query", "file_download", "file_copy",
    "config_change", "usb_mount", "mfa_enroll", "record_delete",
    "permission_change", "process_kill",
]

USER_GROUPS = [
    "Junior_Marketing", "Senior_Marketing", "Warehouse_Floor",
    "Cloud_Admin", "IT_Support", "Corporate_Finance", "Corporate_HR",
    "Supply_Chain_Ops", "Retail_Store_Ops", "Data_Analytics",
    "Legal_Compliance", "Executive_Leadership",
]

LOCATIONS          = ["Pune", "Bangalore", "Mumbai", "Singapore", "Unknown"]
SENSITIVITY_LEVELS = ["Public", "Internal", "Confidential", "PII_RESTRICTED"]
MFA_STATES         = ["success", "failed", "bypassed"]


# ═══════════════════════════════════════════════════════════════════════════
#  KILL-SHOT DETECTION (vectorised for speed)
# ═══════════════════════════════════════════════════════════════════════════

def _flag_killshots(df: pd.DataFrame) -> pd.Series:
    """Return a boolean Series — True for logs matching any of the 5
    kill-shot signatures."""
    return (
        # KS-1  Analog Hole: camera near PII screen
        ((df["optical"] == "Optical Device Detected") &
         (df["resource_name"] == "customer_loyalty_db"))
        # KS-2  Steganography: near-random entropy .jpg
        | ((df["resource_name"] == "encoded_product_shots.jpg") &
           (df["file_entropy"] > 0.97))
        # KS-3  Retail Fraud: warehouse deleting inventory
        | ((df["group"] == "Warehouse_Floor") &
           (df["action_type"] == "record_delete") &
           (df["resource_name"] == "inventory_db"))
        # KS-4  Honey Token: canary file accessed
        | (df["resource_name"] == "Q4_Executive_Bonuses_2026.xlsx")
        # KS-5  Shadow Admin: S3 bucket made public
        | ((df["group"] == "Cloud_Admin") &
           (df["action_type"] == "permission_change") &
           (df["resource_name"] == "S3_Backup_Bucket"))
    )


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────
    # STEP 1 — Load & Flatten JSONL
    # ──────────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  AEGIS-FUSION  Preprocessing Engine v3.0")
    print("=" * 62)
    print(f"\n[1/8] Loading {JSONL_PATH.name} …", flush=True)

    records: list[dict] = []
    raw_logs_by_session: dict[str, list[dict]] = {}   # for test output

    with open(JSONL_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            log = json.loads(line)
            sid = log["session_id"]

            records.append({
                "session_id":    sid,
                "timestamp":     log["timestamp"],
                "uid":           log["actor"]["user"]["uid"],
                "group":         log["actor"]["user"]["group"],
                "mfa_status":    log["actor"]["mfa_status"],
                "action_type":   log["action"]["type"],
                "action_status": log["action"]["status"],
                "resource_name": log["resource"]["name"],
                "sensitivity":   log["resource"]["sensitivity_label"],
                "volume_mb":     log["resource"]["volume_mb"],
                "location":      log["context"]["location"],
                "edr_active":    log["context"]["edr_agent_active"],
                "file_entropy":  log["enrichments"]["aegis_telemetry"]["file_entropy"],
                "typing_var":    log["enrichments"]["aegis_telemetry"]["typing_cadence_variance"],
                "optical":       log["enrichments"]["aegis_telemetry"]["optical_sensor_state"],
            })

            # Keep raw logs for test traceability
            raw_logs_by_session.setdefault(sid, []).append(log)

    df = pd.DataFrame(records)
    del records
    n_logs     = len(df)
    n_sessions = df["session_id"].nunique()
    print(f"      {n_logs:,} logs  →  {n_sessions:,} unique sessions")

    # ──────────────────────────────────────────────────────────────────
    # STEP 2 — Cyclical Time Encoding
    # ──────────────────────────────────────────────────────────────────
    print("[2/8] Cyclical hour encoding (sin/cos) …", flush=True)

    df["ts"] = pd.to_datetime(df["timestamp"])
    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour_frac / 24.0).astype(np.float32)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_frac / 24.0).astype(np.float32)
    print("      hour_sin = sin(2π·hour/24)   hour_cos = cos(2π·hour/24)")

    # ──────────────────────────────────────────────────────────────────
    # STEP 3 — Kill-Shot Detection
    # ──────────────────────────────────────────────────────────────────
    print("[3/8] Detecting kill-shot patterns …", flush=True)

    df["is_killshot"] = _flag_killshots(df)
    ks_log_count = int(df["is_killshot"].sum())
    ks_sessions  = set(df.loc[df["is_killshot"], "session_id"].unique())
    print(f"      {ks_log_count:,} kill-shot logs in {len(ks_sessions):,} sessions")

    # ──────────────────────────────────────────────────────────────────
    # STEP 4 — Indicator Columns for Aggregation
    # ──────────────────────────────────────────────────────────────────
    print("[4/8] Building indicator columns …", flush=True)

    # Action types  → groupby mean = fraction of session
    for a in ACTION_TYPES:
        df[f"act_{a}"] = (df["action_type"] == a).astype(np.float32)

    # User groups   → groupby max = one-hot (constant per session)
    for g in USER_GROUPS:
        df[f"grp_{g}"] = (df["group"] == g).astype(np.float32)

    # Sensitivity   → groupby mean = fraction
    for s in SENSITIVITY_LEVELS:
        df[f"sens_{s}"] = (df["sensitivity"] == s).astype(np.float32)

    # Location      → groupby mean = fraction
    for loc in LOCATIONS:
        df[f"loc_{loc}"] = (df["location"] == loc).astype(np.float32)

    # MFA status    → groupby mean = fraction
    for m in MFA_STATES:
        df[f"mfa_{m}"] = (df["mfa_status"] == m).astype(np.float32)

    # Security booleans
    df["optical_det"]   = (df["optical"] == "Optical Device Detected").astype(np.float32)
    df["edr_off"]       = (~df["edr_active"]).astype(np.float32)
    df["action_failed"] = (df["action_status"] != "success").astype(np.float32)

    # ── BINARY THREAT FLAGS ───────────────────────────────────────────
    # These are the kill-shot discriminators.  Each flag checks for the
    # PRESENCE of one specific threat signal at the per-log level.
    #
    # During session aggregation we apply MAX pooling (not mean), so
    # even 1 suspicious log out of 13 produces a glaring 1.0 in a
    # dimension the VAE has only ever seen as 0.0 during training.
    # This causes reconstruction error to SPIKE on anomalous sessions.

    # KS-4  Honey Token — canary file that should never be accessed
    df["flag_honey_token"] = (
        df["resource_name"].str.contains("Q4_Executive_Bonuses", na=False)
    ).astype(np.float32)

    # KS-3  Destructive Action — record deletion / service sabotage
    df["flag_destructive_action"] = (
        df["action_type"].isin(["record_delete", "process_kill", "config_change"])
    ).astype(np.float32)

    # KS-1/3/5  Critical Resource — databases/buckets that are kill-shot targets
    CRITICAL_RESOURCES = {"inventory_db", "S3_Backup_Bucket",
                          "customer_loyalty_db", "encoded_product_shots.jpg"}
    df["flag_critical_resource"] = (
        df["resource_name"].isin(CRITICAL_RESOURCES)
    ).astype(np.float32)

    # KS-1  Optical Sensor — camera/phone detected near screen
    df["flag_optical_sensor"] = (
        df["optical"] == "Optical Device Detected"
    ).astype(np.float32)

    # KS-2  High-Entropy File — steganography indicator (entropy > 0.95)
    df["flag_high_entropy"] = (
        df["file_entropy"] > 0.95
    ).astype(np.float32)

    total_indicators = (len(ACTION_TYPES) + len(USER_GROUPS) +
                        len(SENSITIVITY_LEVELS) + len(LOCATIONS) +
                        len(MFA_STATES) + 3 + 5)
    print(f"      {total_indicators} indicator columns created (incl. 5 threat flags)")

    # ──────────────────────────────────────────────────────────────────
    # STEP 5 — Velocity (intra-session time deltas)
    # ──────────────────────────────────────────────────────────────────
    print("[5/8] Computing velocity (time deltas) …", flush=True)

    df.sort_values(["session_id", "ts"], inplace=True)
    df["delta_s"] = (
        df.groupby("session_id")["ts"]
          .diff()
          .dt.total_seconds()
          .fillna(0.0)
          .astype(np.float32)
    )

    # ──────────────────────────────────────────────────────────────────
    # STEP 6 — GroupBy Session & Aggregate
    # ──────────────────────────────────────────────────────────────────
    print("[6/8] Aggregating by session_id …", flush=True)

    agg: dict = {
        # ── Temporal ──
        "hour_sin":      ["mean", "std"],
        "hour_cos":      ["mean", "std"],
        # ── Session metadata ──
        "ts":            ["min", "max", "count"],
        # ── Velocity ──
        "delta_s":       ["mean", "std", "min", "max"],
        # ── Volume ──
        "volume_mb":     ["sum", "max", "mean"],
        # ── Telemetry ──
        "file_entropy":  ["max", "mean"],
        "typing_var":    ["mean", "max"],
        # ── Security booleans (mean = ratio) ──
        "optical_det":   "mean",
        "edr_off":       "mean",
        "action_failed": "mean",
        # ── THREAT FLAGS: MAX pooling — 1 hit in session = 1.0 ──
        "flag_honey_token":      "max",
        "flag_destructive_action": "max",
        "flag_critical_resource": "max",
        "flag_optical_sensor":   "max",
        "flag_high_entropy":     "max",
    }
    # Categorical fractions / one-hot
    for a in ACTION_TYPES:
        agg[f"act_{a}"] = "mean"
    for g in USER_GROUPS:
        agg[f"grp_{g}"] = "max"          # constant per session → max=first
    for s in SENSITIVITY_LEVELS:
        agg[f"sens_{s}"] = "mean"
    for loc in LOCATIONS:
        agg[f"loc_{loc}"] = "mean"
    for m in MFA_STATES:
        agg[f"mfa_{m}"] = "mean"

    sess = df.groupby("session_id").agg(agg)

    # Flatten MultiIndex columns:  ("hour_sin", "mean") → "hour_sin_mean"
    sess.columns = [
        "_".join(str(c) for c in col).rstrip("_") if isinstance(col, tuple) else str(col)
        for col in sess.columns
    ]

    # ── Derived features ──────────────────────────────────────────────
    sess["session_duration_s"] = (
        (sess["ts_max"] - sess["ts_min"]).dt.total_seconds().astype(np.float32)
    )
    sess["log_count"] = sess["ts_count"].astype(np.float32)
    sess["velocity_lps"] = np.where(
        sess["session_duration_s"] > 0,
        sess["log_count"] / sess["session_duration_s"],
        sess["log_count"],                     # single-action fallback
    ).astype(np.float32)

    # Drop intermediate ts columns
    sess.drop(columns=["ts_min", "ts_max", "ts_count"], inplace=True)

    # Fill NaN (std on single-log sessions = NaN)
    sess.fillna(0.0, inplace=True)

    print(f"      {len(sess):,} sessions  ×  {sess.shape[1]} raw columns")

    # ── Ordered feature vector ────────────────────────────────────────
    THREAT_FLAGS = [
        "flag_honey_token_max",
        "flag_destructive_action_max",
        "flag_critical_resource_max",
        "flag_optical_sensor_max",
        "flag_high_entropy_max",
    ]

    FEATURE_ORDER: list[str] = (
        # Temporal (4)
        ["hour_sin_mean", "hour_sin_std", "hour_cos_mean", "hour_cos_std"]
        # Session (2)
        + ["session_duration_s", "log_count"]
        # Velocity (5)
        + ["delta_s_mean", "delta_s_std", "delta_s_min", "delta_s_max",
           "velocity_lps"]
        # Volume (3)
        + ["volume_mb_sum", "volume_mb_max", "volume_mb_mean"]
        # Action fractions (11)
        + [f"act_{a}_mean" for a in ACTION_TYPES]
        # User group one-hot (12)
        + [f"grp_{g}_max" for g in USER_GROUPS]
        # Sensitivity fractions (4)
        + [f"sens_{s}_mean" for s in SENSITIVITY_LEVELS]
        # Location fractions (5)
        + [f"loc_{loc}_mean" for loc in LOCATIONS]
        # MFA fractions (3)
        + [f"mfa_{m}_mean" for m in MFA_STATES]
        # Telemetry (4)
        + ["file_entropy_max", "file_entropy_mean",
           "typing_var_mean", "typing_var_max"]
        # Security ratios (3)
        + ["optical_det_mean", "edr_off_mean", "action_failed_mean"]
        # 🚩 Binary Threat Flags — MAX pooled (5)
        + THREAT_FLAGS
    )

    # Verify all features exist
    missing = [f for f in FEATURE_ORDER if f not in sess.columns]
    if missing:
        raise KeyError(f"Missing columns after aggregation: {missing}")
    assert len(FEATURE_ORDER) == 61, f"Expected 61 features, got {len(FEATURE_ORDER)}"

    feat = sess[FEATURE_ORDER].copy()
    print(f"      Feature vector: {len(FEATURE_ORDER)} dimensions ✓")

    # ──────────────────────────────────────────────────────────────────
    # STEP 7 — Train / Test Split + MinMax Normalisation
    # ──────────────────────────────────────────────────────────────────
    print("[7/8] Splitting & normalising …", flush=True)

    is_anomalous = feat.index.isin(ks_sessions)
    train_df = feat[~is_anomalous].copy()
    test_df  = feat[is_anomalous].copy()

    print(f"      Train (normal) : {len(train_df):,} sessions")
    print(f"      Test  (anomaly): {len(test_df):,} sessions")

    # MinMax normalisation using TRAIN statistics only
    #
    # CRITICAL DESIGN: For "trap" columns (threat flags) that are
    # constant 0.0 in training, we keep train=0 but PASS THROUGH
    # the raw test values.  The VAE learns to reconstruct 0.0.
    # At inference, a 1.0 in that dimension is impossible to
    # reconstruct → MSE violently spikes.  That IS the detection.
    scaling: dict[str, dict[str, float]] = {}
    for col in FEATURE_ORDER:
        col_min = float(train_df[col].min())
        col_max = float(train_df[col].max())
        col_range = col_max - col_min

        if col_range < 1e-12:
            # Constant column in training data (e.g., threat flags = 0.0)
            # Train stays at 0.0, but test KEEPS its raw value so that
            # a 1.0 spike passes through to the VAE untouched.
            train_df[col] = 0.0
            # DO NOT zero out test — let the 1.0 hit the VAE raw
            scaling[col]  = {"min": col_min, "max": col_max, "range": 0.0,
                             "trap": True}
        else:
            train_df[col] = (train_df[col] - col_min) / col_range
            test_df[col]  = ((test_df[col] - col_min) / col_range).clip(0.0, 1.0)
            scaling[col]  = {"min": col_min, "max": col_max, "range": col_range}

    # Convert to tensors
    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    test_tensor  = torch.tensor(test_df.values,  dtype=torch.float32)

    print(f"      train_tensor : {list(train_tensor.shape)}")
    print(f"      test_tensor  : {list(test_tensor.shape)}")

    # Sanity: no NaN / Inf
    assert not torch.isnan(train_tensor).any(), "NaN in train tensor!"
    assert not torch.isnan(test_tensor).any(),  "NaN in test tensor!"
    assert not torch.isinf(train_tensor).any(), "Inf in train tensor!"
    assert not torch.isinf(test_tensor).any(),  "Inf in test tensor!"
    print("      NaN/Inf check: PASSED ✓")

    # ──────────────────────────────────────────────────────────────────
    # STEP 8 — Save Artefacts
    # ──────────────────────────────────────────────────────────────────
    print("[8/8] Saving artefacts …", flush=True)

    # Tensors
    torch.save(train_tensor, TRAIN_TENSOR)
    torch.save(test_tensor,  TEST_TENSOR)
    print(f"      → {TRAIN_TENSOR.name}  ({train_tensor.shape[0]:,} × {train_tensor.shape[1]})")
    print(f"      → {TEST_TENSOR.name}   ({test_tensor.shape[0]:,} × {test_tensor.shape[1]})")

    # Test sessions JSONL (for traceability / inference)
    test_session_ids = list(test_df.index)
    with open(TEST_SESSIONS, "w", encoding="utf-8") as fh:
        for sid in test_session_ids:
            logs = raw_logs_by_session.get(sid, [])
            # Detect which kill-shot types are present
            ks_types = set()
            for log in logs:
                enr = log.get("enrichments", {}).get("aegis_telemetry", {})
                rname  = log.get("resource", {}).get("name", "")
                grp    = log.get("actor", {}).get("user", {}).get("group", "")
                atype  = log.get("action", {}).get("type", "")
                opt    = enr.get("optical_sensor_state", "")
                ent    = enr.get("file_entropy", 0)

                if opt == "Optical Device Detected" and rname == "customer_loyalty_db":
                    ks_types.add("analog_hole")
                if rname == "encoded_product_shots.jpg" and ent > 0.97:
                    ks_types.add("steganography")
                if grp == "Warehouse_Floor" and atype == "record_delete" and rname == "inventory_db":
                    ks_types.add("retail_fraud")
                if rname == "Q4_Executive_Bonuses_2026.xlsx":
                    ks_types.add("honey_token")
                if grp == "Cloud_Admin" and atype == "permission_change" and rname == "S3_Backup_Bucket":
                    ks_types.add("shadow_admin")

            entry = {
                "session_id":    sid,
                "log_count":     len(logs),
                "killshot_types": sorted(ks_types),
                "logs":          logs,
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"      → {TEST_SESSIONS.name}  ({len(test_session_ids):,} sessions)")

    # Feature metadata
    meta = {
        "num_features":      len(FEATURE_ORDER),
        "feature_names":     FEATURE_ORDER,
        "scaling":           scaling,
        "total_logs":        n_logs,
        "total_sessions":    n_sessions,
        "train_sessions":    len(train_df),
        "test_sessions":     len(test_df),
        "killshot_logs":     ks_log_count,
        "vocabularies": {
            "action_types":       ACTION_TYPES,
            "user_groups":        USER_GROUPS,
            "locations":          LOCATIONS,
            "sensitivity_levels": SENSITIVITY_LEVELS,
            "mfa_states":         MFA_STATES,
        },
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"      → {META_PATH.name}")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print(f"""
{'=' * 62}
  PREPROCESSING COMPLETE
{'=' * 62}
  Total logs parsed     : {n_logs:>10,}
  Unique sessions       : {n_sessions:>10,}
  Feature dimensions    : {len(FEATURE_ORDER):>10}

  Train (normal)        : {len(train_df):>10,} sessions
  Test  (kill-shot)     : {len(test_df):>10,} sessions

  Kill-shot logs found  : {ks_log_count:>10,}
  Kill-shot sessions    : {len(ks_sessions):>10,}

  Time elapsed          : {elapsed:>10.1f}s
{'=' * 62}
""")


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
