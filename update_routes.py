import pathlib
p = pathlib.Path('api/routes/pipeline.py')
text = p.read_text('utf-8')

# Signature
text = text.replace(
    'async def pipeline_jobs(limit: int = 50, offset: int = 0, status: str | None = None):',
    'async def pipeline_jobs(limit: int = 50, offset: int = 0, status: str | None = None, date: str | None = None):'
)

# Call
text = text.replace(
    'return await list_jobs(limit=limit, offset=offset, status=status or None)',
    'return await list_jobs(limit=limit, offset=offset, status=status or None, date_filter=date or None)'
)

p.write_text(text, 'utf-8')
print("OK")
