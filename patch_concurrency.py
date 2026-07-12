# 1. orchestrator.py
orch_path = 'api/services/pipeline/orchestrator.py'
with open(orch_path, 'r', encoding='utf-8') as f:
    orch_content = f.read()

# Replace defaults of 4 and 8 with 10
orch_content = orch_content.replace('AUDIO_CONCURRENCY", "4"', 'AUDIO_CONCURRENCY", "10"')
orch_content = orch_content.replace('MINDMAP_CONCURRENCY", "4"', 'MINDMAP_CONCURRENCY", "10"')
orch_content = orch_content.replace('TRANSLATE_CONCURRENCY", "8"', 'TRANSLATE_CONCURRENCY", "10"')

with open(orch_path, 'w', encoding='utf-8') as f:
    f.write(orch_content)

# 2. sonnet.py
sonnet_path = 'api/services/summarizer/sonnet.py'
with open(sonnet_path, 'r', encoding='utf-8') as f:
    sonnet_content = f.read()

sonnet_content = sonnet_content.replace('asyncio.Semaphore(4)', 'asyncio.Semaphore(10)')

with open(sonnet_path, 'w', encoding='utf-8') as f:
    f.write(sonnet_content)

print("OK")
