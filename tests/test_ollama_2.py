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
        
    behavior_log = next((l for l in logs if not _brain0_check(l)[0] and l.get('action', {}).get('type') == 'file_download'), None)
    
    print('===== BEHAVIOR ANOMALY (Brain-1) =====')
    print(json.dumps(behavior_log, indent=2))
    print('\nOLLAMA Output:')
    print(json.dumps(await a.analyze(behavior_log, 96), indent=2))
    
    await a.shutdown()

asyncio.run(main())
