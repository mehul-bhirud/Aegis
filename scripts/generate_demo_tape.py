"""
╔══════════════════════════════════════════════════════════════════════════╗
║  AEGIS Demo Tape Generator v3.0 — Fortune 500 Scale                    ║
║  3,000 choreographed logs with a "Golden Reel" kill-chain narrative    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json
import hashlib
import random
from datetime import datetime, timedelta, timezone

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

OUTPUT     = "demo_activity_stream.jsonl"
TOTAL_LOGS = 3000

T_START = datetime(2026, 4, 10, 8, 0, 0, tzinfo=timezone.utc)

# ═══════════════════════════════════════════════════════════════════════════
#  VOCABULARY — 40 Users × 8 Groups
# ═══════════════════════════════════════════════════════════════════════════

USERS = [
    # Engineering (6)
    {"uid": "emp_0001", "group": "Engineering",          "department": "Platform"},
    {"uid": "emp_0002", "group": "Engineering",          "department": "Platform"},
    {"uid": "emp_0003", "group": "Engineering",          "department": "Backend"},
    {"uid": "emp_0004", "group": "Engineering",          "department": "Frontend"},
    {"uid": "emp_0005", "group": "Engineering",          "department": "SRE"},
    {"uid": "emp_0006", "group": "Engineering",          "department": "DevOps"},
    # IT Support (5)
    {"uid": "emp_0010", "group": "IT_Support",           "department": "HelpDesk"},
    {"uid": "emp_0011", "group": "IT_Support",           "department": "HelpDesk"},
    {"uid": "emp_0012", "group": "IT_Support",           "department": "Infrastructure"},
    {"uid": "emp_0013", "group": "IT_Support",           "department": "Networking"},
    {"uid": "emp_0014", "group": "IT_Support",           "department": "Security_Ops"},
    # Corporate Finance (5)
    {"uid": "emp_0020", "group": "Corporate_Finance",    "department": "Treasury"},
    {"uid": "emp_0021", "group": "Corporate_Finance",    "department": "Accounts_Payable"},
    {"uid": "emp_0022", "group": "Corporate_Finance",    "department": "Revenue"},
    {"uid": "emp_0023", "group": "Corporate_Finance",    "department": "Internal_Audit"},
    {"uid": "emp_0024", "group": "Corporate_Finance",    "department": "Tax"},
    # Executive Leadership (4)
    {"uid": "emp_0030", "group": "Executive_Leadership", "department": "C_Suite"},
    {"uid": "emp_0031", "group": "Executive_Leadership", "department": "C_Suite"},
    {"uid": "emp_0032", "group": "Executive_Leadership", "department": "VP_Operations"},
    {"uid": "emp_0033", "group": "Executive_Leadership", "department": "VP_Engineering"},
    # Cloud Admin (4)
    {"uid": "emp_0040", "group": "Cloud_Admin",          "department": "AWS"},
    {"uid": "emp_0041", "group": "Cloud_Admin",          "department": "Azure"},
    {"uid": "emp_0042", "group": "Cloud_Admin",          "department": "GCP"},
    {"uid": "emp_0043", "group": "Cloud_Admin",          "department": "Multi_Cloud"},
    # HR (4)
    {"uid": "emp_0050", "group": "HR",                   "department": "Talent"},
    {"uid": "emp_0051", "group": "HR",                   "department": "Benefits"},
    {"uid": "emp_0052", "group": "HR",                   "department": "Compliance"},
    {"uid": "emp_0053", "group": "HR",                   "department": "Learning"},
    # Warehouse / Logistics (5)
    {"uid": "emp_0060", "group": "Warehouse_Floor",      "department": "Receiving"},
    {"uid": "emp_0061", "group": "Warehouse_Floor",      "department": "Shipping"},
    {"uid": "emp_0062", "group": "Logistics",            "department": "Fleet_Mgmt"},
    {"uid": "emp_0063", "group": "Logistics",            "department": "Route_Planning"},
    {"uid": "emp_0064", "group": "Logistics",            "department": "Customs"},
    # Data Science & Legal (4)
    {"uid": "emp_0070", "group": "Data_Science",         "department": "ML_Platform"},
    {"uid": "emp_0071", "group": "Data_Science",         "department": "Analytics"},
    {"uid": "emp_0080", "group": "Legal",                "department": "IP_Counsel"},
    {"uid": "emp_0081", "group": "Legal",                "department": "Regulatory"},
    # Supply Chain / Procurement (3)
    {"uid": "emp_0090", "group": "Procurement",          "department": "Vendor_Mgmt"},
    {"uid": "emp_0091", "group": "Procurement",          "department": "Sourcing"},
    {"uid": "emp_0092", "group": "Procurement",          "department": "Supply_Chain"},
]

# ═══════════════════════════════════════════════════════════════════════════
#  VOCABULARY — 20 Global Cities
# ═══════════════════════════════════════════════════════════════════════════

LOCATIONS = [
    "Pune", "Bangalore", "Mumbai", "Delhi",
    "Singapore", "Tokyo", "Sydney", "Seoul",
    "London", "Frankfurt", "Amsterdam", "Dublin",
    "New_York", "Austin", "Chicago", "Seattle",
    "Toronto", "Sao_Paulo", "Dubai", "Johannesburg",
]

IP_PREFIXES = {
    "Pune": (10, 1),       "Bangalore": (10, 2),    "Mumbai": (10, 3),
    "Delhi": (10, 4),      "Singapore": (10, 5),     "Tokyo": (10, 6),
    "Sydney": (10, 7),     "Seoul": (10, 8),         "London": (10, 10),
    "Frankfurt": (10, 11), "Amsterdam": (10, 12),    "Dublin": (10, 13),
    "New_York": (10, 20),  "Austin": (10, 21),       "Chicago": (10, 22),
    "Seattle": (10, 23),   "Toronto": (10, 24),      "Sao_Paulo": (10, 25),
    "Dubai": (10, 30),     "Johannesburg": (10, 31),
}

# ═══════════════════════════════════════════════════════════════════════════
#  VOCABULARY — Massive Action & Resource Matrix
# ═══════════════════════════════════════════════════════════════════════════

BASELINE_ACTIONS = [
    # Network
    ("Network", "login"),
    ("Network", "vpn_connect"),
    ("Network", "vpn_disconnect"),
    ("Network", "proxy_auth"),
    ("Network", "dns_query"),
    # Data / File
    ("Data",    "db_query"),
    ("Data",    "file_download"),
    ("Data",    "file_upload"),
    ("Data",    "file_view"),
    ("Data",    "report_generate"),
    # Development
    ("Dev",     "git_commit"),
    ("Dev",     "git_push"),
    ("Dev",     "git_clone"),
    ("Dev",     "code_review"),
    ("Dev",     "ci_pipeline_run"),
    ("Dev",     "container_deploy"),
    # Collaboration
    ("Collab",  "email_sent"),
    ("Collab",  "email_received"),
    ("Collab",  "chat_message"),
    ("Collab",  "video_call"),
    ("Collab",  "calendar_invite"),
    ("Collab",  "ticket_update"),
    ("Collab",  "ticket_create"),
    ("Collab",  "wiki_edit"),
    # System / Admin
    ("System",  "config_change"),
    ("System",  "password_reset"),
    ("System",  "mfa_enrollment"),
    ("System",  "audit_log_view"),
    ("System",  "backup_initiate"),
    # HR / Finance
    ("HR",      "payroll_view"),
    ("HR",      "timesheet_submit"),
    ("HR",      "benefits_update"),
    ("Finance", "invoice_approve"),
    ("Finance", "expense_submit"),
    ("Finance", "budget_review"),
]

RESOURCES = {
    "login":            ["erp_portal", "hr_portal", "crm_system", "sso_gateway", "admin_console"],
    "vpn_connect":      ["corp_vpn_primary", "corp_vpn_failover", "split_tunnel_vpn"],
    "vpn_disconnect":   ["corp_vpn_primary", "corp_vpn_failover"],
    "proxy_auth":       ["squid_proxy", "zscaler_edge"],
    "dns_query":        ["internal_dns", "cloudflare_dns"],
    "db_query":         ["sales_reporting_db", "product_catalog_db", "customer_analytics_db",
                         "inventory_master_db", "order_fulfillment_db", "compliance_db"],
    "file_download":    ["training_slides.pptx", "policy_update.pdf", "brand_guidelines.pdf",
                         "onboarding_checklist.docx", "vendor_contracts.zip", "meeting_notes.md"],
    "file_upload":      ["project_proposal.docx", "data_export.csv", "architecture_diagram.png",
                         "test_results.xlsx", "compliance_report.pdf"],
    "file_view":        ["company_handbook.pdf", "org_chart.pdf", "roadmap_Q3.pptx"],
    "report_generate":  ["monthly_kpi_report", "incident_summary", "utilization_dashboard"],
    "git_commit":       ["aegis-fusion-repo", "microservices-core", "data-pipeline-etl",
                         "frontend-react-app", "infra-terraform", "ml-model-training"],
    "git_push":         ["aegis-fusion-repo", "microservices-core", "frontend-react-app"],
    "git_clone":        ["docs-internal", "onboarding-scripts", "shared-libraries"],
    "code_review":      ["PR-1842", "PR-2201", "PR-3094", "PR-4517"],
    "ci_pipeline_run":  ["jenkins_main", "github_actions_deploy", "gitlab_ci_staging"],
    "container_deploy": ["kubernetes_cluster_prod", "kubernetes_cluster_staging", "ecs_fargate"],
    "email_sent":       ["exchange_server", "gmail_workspace"],
    "email_received":   ["exchange_server", "gmail_workspace"],
    "chat_message":     ["slack_general", "slack_engineering", "teams_hr", "teams_finance"],
    "video_call":       ["zoom_meeting", "teams_meeting", "google_meet"],
    "calendar_invite":  ["outlook_calendar", "google_calendar"],
    "ticket_update":    ["jira_board", "servicenow_itsm", "zendesk_queue"],
    "ticket_create":    ["jira_board", "servicenow_itsm"],
    "wiki_edit":        ["confluence_engineering", "notion_team", "sharepoint_docs"],
    "config_change":    ["proxy_settings", "firewall_rules", "dns_records",
                         "load_balancer_config", "iam_policy"],
    "password_reset":   ["active_directory", "okta_iam"],
    "mfa_enrollment":   ["duo_security", "google_authenticator"],
    "audit_log_view":   ["splunk_siem", "cloudtrail_logs", "datadog_apm"],
    "backup_initiate":  ["s3_backup_vault", "azure_blob_archive", "glacier_deep_archive"],
    "payroll_view":     ["adp_payroll_portal"],
    "timesheet_submit": ["kronos_time_system", "harvest_app"],
    "benefits_update":  ["benefits_portal"],
    "invoice_approve":  ["sap_ap_module", "coupa_procurement"],
    "expense_submit":   ["concur_expense", "expensify"],
    "budget_review":    ["anaplan_budgets", "adaptive_planning"],
}

SENSITIVITY = {
    "erp_portal": "Internal",  "hr_portal": "PII_RESTRICTED",  "crm_system": "Confidential",
    "sso_gateway": "Internal", "admin_console": "Restricted",
    "corp_vpn_primary": "Internal",  "corp_vpn_failover": "Internal",  "split_tunnel_vpn": "Internal",
    "squid_proxy": "Internal",       "zscaler_edge": "Internal",
    "internal_dns": "Public",        "cloudflare_dns": "Public",
    "sales_reporting_db": "Confidential",  "product_catalog_db": "Public",
    "customer_analytics_db": "PII_RESTRICTED",  "inventory_master_db": "Confidential",
    "order_fulfillment_db": "Internal",         "compliance_db": "Restricted",
    "training_slides.pptx": "Public",  "policy_update.pdf": "Internal",
    "brand_guidelines.pdf": "Public",  "onboarding_checklist.docx": "Internal",
    "vendor_contracts.zip": "Confidential",  "meeting_notes.md": "Internal",
    "project_proposal.docx": "Internal",  "data_export.csv": "Confidential",
    "architecture_diagram.png": "Internal",  "test_results.xlsx": "Internal",
    "compliance_report.pdf": "Restricted",
    "company_handbook.pdf": "Public",  "org_chart.pdf": "Internal",  "roadmap_Q3.pptx": "Confidential",
    "monthly_kpi_report": "Confidential",  "incident_summary": "Restricted",
    "utilization_dashboard": "Internal",
    "aegis-fusion-repo": "Confidential",  "microservices-core": "Confidential",
    "data-pipeline-etl": "Internal",      "frontend-react-app": "Internal",
    "infra-terraform": "Restricted",      "ml-model-training": "Confidential",
    "docs-internal": "Internal",  "onboarding-scripts": "Public",  "shared-libraries": "Internal",
    "PR-1842": "Internal",  "PR-2201": "Internal",  "PR-3094": "Internal",  "PR-4517": "Internal",
    "jenkins_main": "Restricted",  "github_actions_deploy": "Restricted",
    "gitlab_ci_staging": "Internal",
    "kubernetes_cluster_prod": "Restricted",  "kubernetes_cluster_staging": "Internal",
    "ecs_fargate": "Internal",
    "exchange_server": "Internal",  "gmail_workspace": "Internal",
    "slack_general": "Public",  "slack_engineering": "Internal",
    "teams_hr": "PII_RESTRICTED",  "teams_finance": "Confidential",
    "zoom_meeting": "Internal",  "teams_meeting": "Internal",  "google_meet": "Internal",
    "outlook_calendar": "Internal",  "google_calendar": "Internal",
    "jira_board": "Internal",  "servicenow_itsm": "Internal",  "zendesk_queue": "Internal",
    "confluence_engineering": "Internal",  "notion_team": "Internal",  "sharepoint_docs": "Internal",
    "proxy_settings": "Restricted",  "firewall_rules": "Restricted",
    "dns_records": "Internal",  "load_balancer_config": "Restricted",  "iam_policy": "Restricted",
    "active_directory": "Restricted",  "okta_iam": "Restricted",
    "duo_security": "Internal",  "google_authenticator": "Internal",
    "splunk_siem": "Restricted",  "cloudtrail_logs": "Restricted",  "datadog_apm": "Internal",
    "s3_backup_vault": "Restricted",  "azure_blob_archive": "Restricted",
    "glacier_deep_archive": "Restricted",
    "adp_payroll_portal": "PII_RESTRICTED",  "kronos_time_system": "Internal",
    "harvest_app": "Internal",  "benefits_portal": "PII_RESTRICTED",
    "sap_ap_module": "Confidential",  "coupa_procurement": "Confidential",
    "concur_expense": "Internal",  "expensify": "Internal",
    "anaplan_budgets": "Confidential",  "adaptive_planning": "Confidential",
    # Kill-shot resources
    "Q4_Executive_Bonuses_2026.xlsx": "Confidential",
    "customer_loyalty_db": "PII_RESTRICTED",
    "edr_agent": "Restricted",
    "vendor_routing_numbers": "Confidential",
    "POS_Terminal_4": "Internal",
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
    a, b = IP_PREFIXES.get(location, (10, 99))
    return f"{a}.{b}.{rng.randint(1, 254)}.{rng.randint(1, 254)}"

def _ts_str(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts.microsecond // 1000:03d}Z"

def _assemble(
    ts: datetime, salt: int,
    user: dict, category: str, atype: str, status: str,
    resource: str, volume: float, location: str, ip_addr: str,
    edr: bool, mfa: str, entropy: float, typing_var: float, optical: str,
    extra_enrichments: dict | None = None,
) -> dict:
    ts_s = _ts_str(ts)
    enrichments = {
        "aegis_telemetry": {
            "file_entropy":            round(entropy, 4),
            "typing_cadence_variance": round(typing_var, 4),
            "optical_sensor_state":    optical,
        }
    }
    if extra_enrichments:
        enrichments["aegis_telemetry"].update(extra_enrichments)

    return {
        "event_id":   _event_id(ts_s, user["uid"], atype, salt),
        "timestamp":  ts_s,
        "session_id": _session_id(user["uid"], ts),
        "metadata": {
            "version":  "1.0",
            "product":  "aegis-fusion",
            "merkle_leaf_hash": hashlib.sha256(
                f"leaf|{ts_s}|{user['uid']}|{salt}".encode()
            ).hexdigest(),
        },
        "actor": {
            "user": {
                "uid":        user["uid"],
                "group":      user["group"],
                "department": user.get("department", user["group"]),
            },
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
        "enrichments": enrichments,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BUILDERS — Baseline & Noise
# ═══════════════════════════════════════════════════════════════════════════

def build_baseline(rng: random.Random, ts: datetime, salt: int) -> dict:
    user = rng.choice(USERS)
    cat, atype = rng.choice(BASELINE_ACTIONS)
    res = rng.choice(RESOURCES.get(atype, ["unknown"]))
    loc = rng.choice(LOCATIONS)
    return _assemble(
        ts, salt, user, cat, atype, "success", res,
        volume=rng.uniform(0.01, 8.0), location=loc, ip_addr=_ip(rng, loc),
        edr=True, mfa="success",
        entropy=rng.uniform(0.05, 0.55),
        typing_var=max(0.0, rng.gauss(0.12, 0.03)),
        optical="Clear",
    )

def build_noise(rng: random.Random, ts: datetime, salt: int) -> dict:
    """Failed login / MFA failure — looks suspicious but harmless."""
    user = rng.choice(USERS)
    loc = rng.choice(LOCATIONS)
    action = rng.choice(["login", "vpn_connect", "password_reset"])
    res = rng.choice(RESOURCES.get(action, ["sso_gateway"]))
    return _assemble(
        ts, salt, user, "Network", action, "failed", res,
        volume=0.1, location=loc, ip_addr=_ip(rng, loc),
        edr=True, mfa="failed",
        entropy=rng.uniform(0.05, 0.25),
        typing_var=max(0.0, rng.gauss(0.18, 0.05)),
        optical="Clear",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  BUILDERS — Kill Shots (Golden Reel)
# ═══════════════════════════════════════════════════════════════════════════

def build_brain_0_killshot(rng: random.Random, ts: datetime, salt: int) -> dict:
    """Honey-token canary file access — instant Brain 0 override."""
    user = [u for u in USERS if u["group"] == "Corporate_Finance"][0]
    loc = "Bangalore"
    return _assemble(
        ts, salt, user, "Data", "file_download", "success",
        resource="Q4_Executive_Bonuses_2026.xlsx",
        volume=rng.uniform(1.5, 3.0), location=loc, ip_addr=_ip(rng, loc),
        edr=True, mfa="success",
        entropy=rng.uniform(0.3, 0.6),
        typing_var=max(0.0, rng.gauss(0.14, 0.04)),
        optical="Clear",
    )

def build_brain_1_edr_stop(rng: random.Random, ts: datetime, salt: int,
                           mal_user: dict, loc: str, ip: str) -> dict:
    """EDR agent killed — evasion technique."""
    return _assemble(
        ts, salt, mal_user, "System", "service_stop", "success",
        resource="edr_agent",
        volume=0.01, location=loc, ip_addr=ip,
        edr=False, mfa="success",
        entropy=rng.uniform(0.1, 0.2),
        typing_var=0.25, optical="Clear",
    )

def build_brain_1_file_copy(rng: random.Random, ts: datetime, salt: int,
                            mal_user: dict, loc: str, ip: str) -> dict:
    """Massive data exfiltration immediately after EDR kill."""
    return _assemble(
        ts, salt, mal_user, "Data", "file_copy", "success",
        resource="customer_loyalty_db",
        volume=rng.uniform(150.0, 300.0), location=loc, ip_addr=ip,
        edr=False, mfa="success",
        entropy=rng.uniform(0.7, 0.9),
        typing_var=0.25, optical="Clear",
    )

def build_trojan_horse(ts: datetime, salt: int) -> dict:
    """Supply-chain fraud — vendor routing switched to Cyprus."""
    ts_s = _ts_str(ts)
    return {
        "event_id":   _event_id(ts_s, "EMP-802", "db_update", salt),
        "timestamp":  ts_s,
        "actor": {
            "user": {
                "uid": "EMP-802",
                "group": "Procurement_Manager",
                "department": "Supply Chain",
            }
        },
        "action": {
            "type": "db_update",
            "description": "Modify vendor banking details",
        },
        "resource": {
            "name": "vendor_routing_numbers",
            "sensitivity_label": "Confidential",
            "volume_mb": 0.01,
        },
        "context": {
            "location": "Chicago",
            "edr_agent_active": True,
        },
        "enrichments": {
            "aegis_telemetry": {
                "optical_sensor_state": "Clear",
                "transaction_details": {
                    "vendor_id": "VND-4491 (Samsung Electronics)",
                    "previous_bank": "Chase Bank (USA)",
                    "new_bank": "Hellenic Bank Public Company Ltd (Cyprus)",
                    "amount_pending": "$2,450,000.00",
                },
            }
        },
    }

def build_biometric_baseline(ts: datetime, salt: int) -> dict:
    """Store Manager normal POS check — clean biometric fingerprint."""
    ts_s = _ts_str(ts)
    return {
        "event_id":   _event_id(ts_s, "EMP-241", "inventory_check", salt),
        "timestamp":  ts_s,
        "actor": {
            "user": {"uid": "EMP-241", "group": "Store_Manager"},
            "mfa_status": "success",
        },
        "action": {"category": "Data", "type": "inventory_check", "status": "success"},
        "resource": {
            "name": "POS_Terminal_4",
            "sensitivity_label": "Internal",
            "volume_mb": 0.02,
        },
        "context": {"location": "Mumbai", "ip_address": "10.3.44.12", "edr_agent_active": True},
        "enrichments": {
            "aegis_telemetry": {
                "optical_sensor_state":    "Clear",
                "typing_cadence_ms":       110,
                "mouse_velocity":          45.2,
                "file_entropy":            0.22,
                "typing_cadence_variance": 0.08,
            }
        },
    }

def build_biometric_hijack(ts: datetime, salt: int) -> dict:
    """Same session, but a different human is behind the keyboard."""
    ts_s = _ts_str(ts)
    return {
        "event_id":   _event_id(ts_s, "EMP-241", "refund_process", salt),
        "timestamp":  ts_s,
        "actor": {
            "user": {"uid": "EMP-241", "group": "Store_Manager"},
            "mfa_status": "success",
        },
        "action": {
            "category": "Data",
            "type":     "refund_process",
            "status":   "success",
            "description": "Manual override refund: $850.00",
        },
        "resource": {
            "name": "POS_Terminal_4",
            "sensitivity_label": "Confidential",
            "volume_mb": 0.03,
        },
        "context": {"location": "Mumbai", "ip_address": "10.3.44.12", "edr_agent_active": True},
        "enrichments": {
            "aegis_telemetry": {
                "optical_sensor_state":    "Clear",
                "typing_cadence_ms":       480,
                "mouse_velocity":          210.5,
                "file_entropy":            0.15,
                "typing_cadence_variance": 0.42,
            }
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN SCRIPT — The Golden Reel Choreography
# ═══════════════════════════════════════════════════════════════════════════

def main():
    rng = random.Random(42)         # deterministic seed
    logs: list[dict] = []

    current_ts = T_START

    # Persistent malicious actor for Brain 1 kill-chain
    mal_user = [u for u in USERS if u["group"] == "Engineering"][0]
    mal_loc  = "Pune"
    mal_ip   = _ip(rng, mal_loc)

    for i in range(1, TOTAL_LOGS + 1):
        salt = i
        current_ts += timedelta(seconds=rng.uniform(3, 25))

        # ══════════════════════════════════════════════════════════════
        # THE GOLDEN REEL (Logs 1 – 250)
        # Precise kill-shot choreography for the 3-minute pitch
        # ══════════════════════════════════════════════════════════════

        # ── Phase 1: Warm-up baseline (1-59) ─────────────────────────
        if 1 <= i <= 49:
            log = build_baseline(rng, current_ts, salt)

        # Sprinkle noise at 50
        elif i == 50:
            log = build_noise(rng, current_ts, salt)

        elif 51 <= i <= 59:
            log = build_baseline(rng, current_ts, salt)

        # ── Kill-Shot 1: Brain 0 — Honey Token (Log 60) ─────────────
        elif i == 60:
            log = build_brain_0_killshot(rng, current_ts, salt)

        # ── Buffer zone (61-118) ─────────────────────────────────────
        elif 61 <= i <= 118:
            # Drop 2-3 noise events for realism
            if i in (75, 95, 110):
                log = build_noise(rng, current_ts, salt)
            else:
                log = build_baseline(rng, current_ts, salt)

        # ── Kill-Shot 2: Brain 1 — EDR Kill + Exfil (Logs 119-120) ──
        elif i == 119:
            log = build_brain_1_edr_stop(rng, current_ts, salt, mal_user, mal_loc, mal_ip)
        elif i == 120:
            current_ts += timedelta(seconds=2)  # immediately after
            log = build_brain_1_file_copy(rng, current_ts, salt, mal_user, mal_loc, mal_ip)

        # ── Buffer zone (121-178) ────────────────────────────────────
        elif 121 <= i <= 178:
            if i in (135, 150, 165):
                log = build_noise(rng, current_ts, salt)
            else:
                log = build_baseline(rng, current_ts, salt)

        # ── Kill-Shot 3: Trojan Horse — Supply Chain Fraud (Log 180) ─
        elif i == 179:
            # Pre-fill one normal procurement log for context buffer
            proc_user = [u for u in USERS if u["group"] == "Procurement"][0]
            log = _assemble(
                current_ts, salt, proc_user, "Finance", "invoice_approve", "success",
                "sap_ap_module", volume=0.01, location="Chicago", ip_addr=_ip(rng, "Chicago"),
                edr=True, mfa="success", entropy=0.15, typing_var=0.10, optical="Clear",
            )
        elif i == 180:
            log = build_trojan_horse(current_ts, salt)

        # ── Buffer zone (181-228) ────────────────────────────────────
        elif 181 <= i <= 228:
            if i in (195, 210, 220):
                log = build_noise(rng, current_ts, salt)
            else:
                log = build_baseline(rng, current_ts, salt)

        # ── Kill-Shot 4: Biometric Hijack (Logs 229-230) ────────────
        elif i == 229:
            log = build_biometric_baseline(current_ts, salt)
        elif i == 230:
            current_ts += timedelta(seconds=45)
            log = build_biometric_hijack(current_ts, salt)

        # ── Remaining Golden Reel tail (231-250) ─────────────────────
        elif 231 <= i <= 250:
            log = build_baseline(rng, current_ts, salt)

        # ══════════════════════════════════════════════════════════════
        # THE LONG TAIL (Logs 251 – 3000)
        # Highly randomized enterprise chatter — safe but lively
        # ══════════════════════════════════════════════════════════════
        else:
            dice = rng.random()
            if dice < 0.06:
                # ~6% noise (failed logins, MFA failures)
                log = build_noise(rng, current_ts, salt)
            else:
                log = build_baseline(rng, current_ts, salt)

        logs.append(log)

    # ── Write ─────────────────────────────────────────────────────────
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

    print(f"Successfully generated {TOTAL_LOGS:,} choreographed logs to {OUTPUT}.")
    print(f"   Golden Reel:  Logs 1-250  (4 kill-shots at #60, #120, #180, #230)")
    print(f"   Long Tail:    Logs 251-3000 (randomized enterprise noise)")


if __name__ == "__main__":
    main()
