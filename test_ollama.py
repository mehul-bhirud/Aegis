import sys
import json
import asyncio
sys.path.append('.')
from main import OllamaAnalyst, _brain0_check

async def main():
    a = OllamaAnalyst()
    await a.initialize()
    
    with open('enterprise_activity_stream.jsonl', 'r', encoding='utf-8') as f:
        logs = [json.loads(line) for line in f]
        
    exact_log = next((l for l in logs if _brain0_check(l)[0]), None)
    behavior_log = next((l for l in logs if not _brain0_check(l)[0] and l.get('resource', {}).get('volume_mb', 0) > 4000), None)
    
    print('===== EXACT ANOMALY (Brain-0) =====')
    print(json.dumps(exact_log, indent=2))
    print('\nOLLAMA Output:')
    print(json.dumps(await a.analyze(exact_log, 100), indent=2))
    
    print('\n===== BEHAVIOR ANOMALY (Brain-1) =====')
    print(json.dumps(behavior_log, indent=2))
    print('\nOLLAMA Output:')
    print(json.dumps(await a.analyze(behavior_log, 96), indent=2))
    
    await a.shutdown()

asyncio.run(main())
