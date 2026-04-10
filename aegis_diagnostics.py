import asyncio
import json
import os
import urllib.request
import urllib.error
import websockets

# ANSI color codes for terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_pass(msg):
    print(f"{GREEN}✅ PASS: {msg}{RESET}")

def print_fail(msg):
    print(f"{RED}❌ FAIL: {msg}{RESET}")

def test_backend_api():
    print(f"\n{BOLD}{CYAN}--- 1. Backend API Test ---{RESET}")
    try:
        req = urllib.request.Request("http://localhost:8000/")
        with urllib.request.urlopen(req, timeout=5) as response:
            if response.status == 200:
                print_pass("Backend API is reachable at http://localhost:8000/")
            else:
                print_fail(f"Backend API returned status code {response.status}")
    except urllib.error.URLError as e:
        print_fail(f"Failed to connect to backend API: {e}")

def test_file_existence():
    print(f"\n{BOLD}{CYAN}--- 2. Log File Test ---{RESET}")
    if os.path.exists("enterprise_activity_stream.jsonl"):
        print_pass("enterprise_activity_stream.jsonl exists in the directory.")
    else:
        print_fail("enterprise_activity_stream.jsonl NOT found.")

async def test_websocket_stream():
    print(f"\n{BOLD}{CYAN}--- 3. WebSocket Live Test ---{RESET}")
    ws_url = "ws://localhost:8000/ws/stream"
    try:
        async with websockets.connect(ws_url, ping_interval=None) as websocket:
            print_pass(f"Connected to WebSocket stream at {ws_url}")
            
            # Send heartbeat
            await websocket.send("ping")
            print_pass("Heartbeat 'ping' sent to server.")
            
            logs_captured = []
            print(f"{YELLOW}Waiting to capture 3 incoming logs... (Auto-starting stream if idle){RESET}")
            
            # Start the stream automatically to ensure logs are flowing
            try:
                req = urllib.request.Request("http://localhost:8000/api/stream/start", method="POST")
                urllib.request.urlopen(req, timeout=2)
            except urllib.error.HTTPError as e:
                # 409 Conflict means it's already running, which is fine
                if e.code != 409:
                    print_fail(f"Could not start stream: {e}")
            except Exception as e:
                print_fail(f"Could not start stream: {e}")
            
            while len(logs_captured) < 3:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    if "pong" in data:
                        print_pass("Received heartbeat response ('pong').")
                        continue
                        
                    logs_captured.append(data)
                    print(f"  - Captured log {len(logs_captured)}/3")
                except asyncio.TimeoutError:
                    print_fail("Timeout while waiting for incoming logs.")
                    return
            
            print_pass("Successfully captured 3 incoming logs.")
            
            # Schema Validation
            print(f"\n{BOLD}{CYAN}--- 4. Schema Validation (Auto-Detect) ---{RESET}")
            for i, data in enumerate(logs_captured, 1):
                mismatches = []
                
                # Check risk_score (can be root or inside aegis_analysis)
                risk = data.get("risk_score")
                if risk is None and "aegis_analysis" in data:
                    risk = data["aegis_analysis"].get("risk_score")
                
                if risk is None:
                    mismatches.append("Missing key: 'risk_score'")
                elif not isinstance(risk, (int, float)):
                    mismatches.append(f"'risk_score' should be int/float, got {type(risk).__name__}")
                
                # Check merkle_root (can be root or inside aegis_analysis)
                merkle = data.get("merkle_root")
                if merkle is None and "aegis_analysis" in data:
                    merkle = data["aegis_analysis"].get("merkle_root")
                
                if merkle is None:
                    mismatches.append("Missing key: 'merkle_root'")
                elif not isinstance(merkle, str):
                    mismatches.append(f"'merkle_root' should be str, got {type(merkle).__name__}")
                
                # Check raw_log (must be root)
                raw_log = data.get("raw_log")
                if raw_log is None:
                    mismatches.append("Missing key: 'raw_log'")
                elif not isinstance(raw_log, dict):
                    mismatches.append(f"'raw_log' should be dict, got {type(raw_log).__name__}")
                    
                if mismatches:
                    print_fail(f"Log {i} Schema Invalid. Mismatches -> {', '.join(mismatches)}")
                else:
                    print_pass(f"Log {i} Schema Validated successfully (risk_score, merkle_root, raw_log correct).")
                    
    except Exception as e:
        print_fail(f"WebSocket connection failed: {e}")

if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}=== AEGIS FUSION E2E DIAGNOSTICS ==={RESET}")
    test_backend_api()
    test_file_existence()
    asyncio.run(test_websocket_stream())
    print(f"\n{BOLD}{CYAN}=== DIAGNOSTICS COMPLETE ==={RESET}\n")
