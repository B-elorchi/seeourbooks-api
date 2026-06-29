import pathlib
p = pathlib.Path('api/services/pipeline/epub.py')
text = p.read_text('utf-8')

# Signature 1
text = text.replace(
    '    mindmap_url: str | None,\n) -> object:',
    '    mindmap_url: str | None,\n    audio_url_translated: str | None = None,\n    mindmap_url_translated: str | None = None,\n) -> object:'
)

# Signature 2
text = text.replace(
    '    mindmap_url: str | None,\n) -> None:',
    '    mindmap_url: str | None,\n    audio_url_translated: str | None = None,\n    mindmap_url_translated: str | None = None,\n) -> None:'
)

# Signature 3
text = text.replace(
    '    mindmap_url:     str | None = None,\n) -> str:',
    '    mindmap_url:     str | None = None,\n    audio_url_translated: str | None = None,\n    mindmap_url_translated: str | None = None,\n) -> str:'
)

# Function call
text = text.replace(
    '        audio_url=audio_url, mindmap_url=mindmap_url,\n    )',
    '        audio_url=audio_url, mindmap_url=mindmap_url,\n        audio_url_translated=audio_url_translated, mindmap_url_translated=mindmap_url_translated,\n    )'
)

p.write_text(text, 'utf-8')
