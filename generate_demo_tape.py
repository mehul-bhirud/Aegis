import json
import hashlib
import random
from datetime import datetime, timedelta, timezone

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION & VOCAB
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT = "demo_activity_stream.jsonl"
TOTAL_LOGS = 300

T_START = datetime(2026, 4, 10, 8, 30, 0, tzinfo=timezone.utc)

USERS = [
    {"uid": "emp_0001", "group": "IT_Support"},
    {"uid": "emp_0002", "group": "Warehouse_Floor"},
    {"uid": "emp_0003", "group": "Corporate_Finance"},
    {"uid": "emp_0004", "group": "Executive_Leadership"},
    {"uid": "emp_0005", "group": "Cloud_Admin"},
]

LOCATIONS = ["Pune", "Bangalore", "Mumbai", "Singapore"]

BASELINE_ACTIONS = [
    ("Network", "login"),
    ("Network", "vpn_connect"),
    ("Data",    "db_query"),
    ("Data",    "file_download"),
    ("System",  "config_change"),
]

RESOURCES = {
    "login": ["erp_portal", "hr_portal", "crm_system"],
    "vpn_connect": ["corp_vpn_primary"],
    "db_query": ["sales_reporting_db", "product_catalog_db"],
    "file_download": ["training_slides.pptx", "policy_update.pdf"],
    "config_change": ["proxy_settings"],
}

SENSITIVITY = {
    "erp_portal": "Internal", "hr_portal": "PII_RESTRICTED",
    "corp_vpn_primary": "Internal", "sales_reporting_db": "Confidential",
    "product_catalog_db": "Public", "training_slides.pptx": "Public",
    "policy_update.pdf": "Internal", "proxy_settings": "Internal",
    "Q4_Executive_Bonuses_2026.xlsx": "Confidential",
    "customer_loyalty_db": "PII_RESTRICTED",
    "edr_agent": "Confidential",
}

# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _event_id(ts_str: str, uid: str, atype: str, salt: int) -> str:
    raw = f"{ts_str}|{uid}|{atype}|{salt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]

def _session_id(uid: str, dt: datetime) -> str:
    block = dt.hour // 4
    raw = f"{uid}|{dt.strftime('%Y-%m-%d')}|{block}"
    return "sess_" + hashlib.md5(raw.encode()).hexdigest()[:8]

def _ip(rng: random.Random, location: str) -> str:
    prefix = {"Pune": (10, 1), "Bangalore": (10, 2),
              "Mumbai": (10, 3), "Singapore": (10, 4)}
    a, b = prefix.get(location, (10, 99))
    return f"{a}.{b}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"

def _assemble(
    ts: datetime, salt: int,
    user: dict, category: str, atype: str, status: str,
    resource: str, volume: float, location: str, ip_addr: str,
    edr: bool, mfa: str, entropy: float, typing_var: float, optical: str,
) -> dict:
    ts_str = ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"
    return {
        "event_id":   _event_id(ts_str, user["uid"], atype, salt),
        "timestamp":  ts_str,
        "session_id": _session_id(user["uid"], ts),
        "metadata": {
            "version":          "1.0",
            "product":          "aegis-fusion",
            "merkle_leaf_hash": hashlib.sha256(f"leaf|{ts_str}|{user['uid']}|{salt}".encode()).hexdigest(),
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
#  BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_baseline(rng, ts, salt):
    user = rng.choice(USERS)
    cat, atype = rng.choice(BASELINE_ACTIONS)
    res = rng.choice(RESOURCES.get(atype, ["unknown"]))
    loc = rng.choice(LOCATIONS)
    return _assemble(
        ts, salt, user, cat, atype, "success", res,
        volume=rng.uniform(0.1, 5.0), location=loc, ip_addr=_ip(rng, loc),
        edr=True, mfa="success", entropy=rng.uniform(0.1, 0.6),
        typing_var=max(0.0, rng.gauss(0.12, 0.03)), optical="Clear"
    )

def build_noise(rng, ts, salt):
    user = rng.choice(USERS)
    loc = rng.choice(LOCATIONS)
    return _assemble(
        ts, salt, user, "Network", "login", "failed", "erp_portal",
        volume=0.1, location=loc, ip_addr=_ip(rng, loc),
        edr=True, mfa="failed", entropy=rng.uniform(0.1, 0.3),
        typing_var=max(0.0, rng.gauss(0.18, 0.05)), optical="Clear"
    )

def build_brain_0_killshot(rng, ts, salt):
    user = USERS[2] # Corporate_Finance
    loc = "Bangalore"
    return _assemble(
        ts, salt, user, "Data", "file_download", "success",
        resource="Q4_Executive_Bonuses_2026.xlsx",
        volume=rng.uniform(1.5, 3.0), location=loc, ip_addr=_ip(rng, loc),
        edr=True, mfa="success", entropy=rng.uniform(0.3, 0.6),
        typing_var=max(0.0, rng.gauss(0.14, 0.04)), optical="Clear"
    )

def build_brain_1_edr_stop(rng, ts, salt, mal_user, loc, ip):
    return _assemble(
        ts, salt, mal_user, "System", "service_stop", "success",
        resource="edr_agent",
        volume=0.01, location=loc, ip_addr=ip,
        edr=False, mfa="success", entropy=rng.uniform(0.1, 0.2),
        typing_var=0.25, optical="Clear"
    )

def build_brain_1_file_copy(rng, ts, salt, mal_user, loc, ip):
    return _assemble(
        ts, salt, mal_user, "Data", "file_copy", "success",
        resource="customer_loyalty_db",
        volume=rng.uniform(150.0, 300.0), location=loc, ip_addr=ip,
        edr=False, mfa="success", entropy=rng.uniform(0.7, 0.9),
        typing_var=0.25, optical="Clear"
    )

# ═══════════════════════════════════════════════════════════════════════════
#  MAIN SCRIPT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rng = random.Random(999)
    logs = []
    
    current_ts = T_START
    
    mal_user = USERS[0]
    mal_loc = "Pune"
    mal_ip = _ip(rng, mal_loc)
    
    for i in range(1, TOTAL_LOGS + 1):
        salt = i
        current_ts += timedelta(seconds=rng.uniform(10, 60))
        
        # 1-50: Baseline
        if 1 <= i <= 50:
            log = build_baseline(rng, current_ts, salt)
        # 51: Noise
        elif i == 51:
            log = build_noise(rng, current_ts, salt)
        # 52-120: Baseline
        elif 52 <= i <= 120:
            log = build_baseline(rng, current_ts, salt)
        # 121: Brain 0 Killshot
        elif i == 121:
            log = build_brain_0_killshot(rng, current_ts, salt)
        # 122-180: Baseline
        elif 122 <= i <= 180:
            log = build_baseline(rng, current_ts, salt)
        # 181: Brain 1 Killshot (EDR stop)
        elif i == 181:
            log = build_brain_1_edr_stop(rng, current_ts, salt, mal_user, mal_loc, mal_ip)
        # 182: Brain 1 Killshot (Mass file copy immediately after)
        elif i == 182:
            current_ts += timedelta(seconds=2) # Immediately after
            log = build_brain_1_file_copy(rng, current_ts, salt, mal_user, mal_loc, mal_ip)
        # 183-300: Baseline
        else:
            log = build_baseline(rng, current_ts, salt)
            
        logs.append(log)
        
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
            
    print(f"Successfully generated {TOTAL_LOGS} choreographed logs to {OUTPUT}.")

if __name__ == "__main__":
    main()
