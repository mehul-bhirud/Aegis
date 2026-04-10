import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ANSI color codes
GREEN = '\033[92;1m'
RED = '\033[91;1m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_pass(msg):
    print(f"[{GREEN}PASS{RESET}] {msg}")

def print_fail(msg, fix_cmd=None):
    print(f"[{RED}FAIL{RESET}] {msg}")
    if fix_cmd:
        print(f"       {YELLOW}Fix Action:{RESET} {fix_cmd}")

def check_file(filename):
    if Path(filename).exists():
        print_pass(f"Found local file: {filename}")
        return True
    else:
        print_fail(f"Missing local file: {filename}", fix_cmd=f"Ensure {filename} is in the current directory.")
        return False

def check_endpoint(url, name, fix_cmd, extract_json=None):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                if extract_json:
                    data = json.loads(response.read().decode())
                    if extract_json(data):
                        print_pass(f"Service online: {name} ({url})")
                        return True
                    else:
                        print_fail(f"Service running but validation failed: {name}", fix_cmd=fix_cmd)
                        return False
                print_pass(f"Service online: {name} ({url})")
                return True
            else:
                print_fail(f"Service returned {response.status}: {name}", fix_cmd=fix_cmd)
                return False
    except urllib.error.URLError as e:
        print_fail(f"Service offline: {name} ({e.reason})", fix_cmd=fix_cmd)
        return False
    except Exception as e:
        print_fail(f"Service error: {name} ({str(e)})", fix_cmd=fix_cmd)
        return False

def trigger_live_demo():
    print(f"\n{CYAN}{BOLD}▶ INITIATING LIVE DEMO KILL SHOT...{RESET}")
    time.sleep(1)
    
    # Steganography Payload (Brain 0 Signature)
    payload = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "actor": {
            "user": {
                "uid": "U-DEMO-HACKER",
                "group": "Marketing"
            },
            "mfa_status": "success"
        },
        "action": {
            "type": "file_copy"
        },
        "resource": {
            "name": "encoded_product_shots.jpg",
            "volume_mb": 5120.0,
            "sensitivity_label": "Confidential"
        },
        "context": {
            "location": "Unknown",
            "edr_agent_active": False
        },
        "enrichments": {
            "aegis_telemetry": {
                "file_entropy": 0.99,
                "typing_cadence_variance": 0.12,
                "optical_sensor_state": "Clear"
            }
        }
    }
    
    url = "http://localhost:8000/api/inject_test_log"
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method="POST", headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode())
            print(f"[{GREEN}SUCCESS{RESET}] Demo injected! Target websocket stream should be flashing red.")
            print(f"         {YELLOW}Response:{RESET} {res_data.get('message', 'OK')}")
    except urllib.error.URLError as e:
        print(f"[{RED}FAILED{RESET}] Could not inject demo payload: {e.reason}")
        print(f"         {YELLOW}Fix Action:{RESET} Ensure FastAPI backend is running on 8000.")

def main():
    print(f"{CYAN}{BOLD}╔═══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║  AEGIS-FUSION DIAGNOSTIC PRE-FLIGHT CHECK                 ║{RESET}")
    print(f"{CYAN}{BOLD}╚═══════════════════════════════════════════════════════════╝{RESET}\n")
    
    all_passed = True
    
    print(f"{BOLD}--- File Integrity Checks ---{RESET}")
    all_passed &= check_file("enterprise_activity_stream.jsonl")
    all_passed &= check_file("aegis_vae_model_v2.pth")
    
    print(f"\n{BOLD}--- Service Health Checks ---{RESET}")
    all_passed &= check_endpoint(
        "http://localhost:11434/api/tags", 
        "Ollama LLM Server", 
        "Run `ollama serve` and ensure `llama3` is pulled (`ollama pull llama3`)",
        extract_json=lambda d: any(m.get("name", "").startswith("llama3") for m in d.get("models", []))
    )
    
    all_passed &= check_endpoint(
        "http://localhost:8000/", 
        "FastAPI Backend", 
        "Run `python main.py` in the backend directory."
    )
    
    all_passed &= check_endpoint(
        "http://localhost:3000/", 
        "Next.js Frontend", 
        "Run `npm run dev` in the frontend directory."
    )
    
    print("\n" + "="*61)
    if all_passed:
        print(f"{GREEN}{BOLD}SYSTEM STATUS: GREEN. ALL SYSTEMS GO FOR DEMO.{RESET}")
        
        if len(sys.argv) > 1 and sys.argv[1] == "--demo":
            trigger_live_demo()
        else:
            print(f"\nTo trigger the demo 'Kill Shot', run: {CYAN}python verify_system.py --demo{RESET}")
    else:
        print(f"{RED}{BOLD}SYSTEM STATUS: RED. DIAGNOSTIC TESTS FAILED.{RESET}")
        print("Please resolve the errors above before the live pitch.")
    print("="*61)

if __name__ == "__main__":
    main()
