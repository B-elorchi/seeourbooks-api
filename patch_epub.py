import re

# 1. Update epub.py
epub_path = 'api/services/pipeline/epub.py'
with open(epub_path, 'r', encoding='utf-8') as f:
    epub_content = f.read()

# Fix _build_chapter_insights signature
old_sig = '''def _build_chapter_insights(
    epub_mod,
    *,
    slug: str,
    ch: dict,
    language: str,
    translated_lang: str | None = None,
    chapter_audio: dict[int, str],
    chapter_mindmap: dict[int, dict],
) -> object | None:'''
new_sig = '''def _build_chapter_insights(
    epub_mod,
    *,
    slug: str,
    ch: dict,
    language: str,
    translated_lang: str | None = None,
    chapter_audio: dict[int, str],
    chapter_mindmap: dict[int, dict],
    chapter_audio_translated: dict[int, str] | None = None,
    chapter_mindmap_translated: dict[int, dict] | None = None,
) -> object | None:'''
epub_content = epub_content.replace(old_sig, new_sig)

# Fix _build_chapter_insights body (chapter_audio_translated)
old_body1 = '''    # Chapter audio
    audio_url = chapter_audio.get(idx)
    if audio_url:
        body_parts += [
            f'<div class="asset-box">',
            f'<h3>dYZ  {audio_label}</h3>',
            f'<a class="asset-link" href="{escape(audio_url)}">{escape(audio_url)}</a>',
            f'</div>',
        ]'''
new_body1 = '''    # Chapter audio
    audio_url = chapter_audio.get(idx)
    if audio_url:
        body_parts += [
            f'<div class="asset-box">',
            f'<h3>dYZ  {audio_label}</h3>',
            f'<a class="asset-link" href="{escape(audio_url)}">{escape(audio_url)}</a>',
            f'</div>',
        ]
        
    if chapter_audio_translated and translated_lang:
        audio_url_tr = chapter_audio_translated.get(idx)
        if audio_url_tr:
            is_tr_arabic   = translated_lang.lower() == "ar"
            tr_audio_label = 'OU^O O U,U?OU, (U.OOOU.)' if is_tr_arabic else 'Chapter Audio (Translated)'
            body_parts += [
                f'<div class="asset-box">',
                f'<h3>dYZ  {tr_audio_label}</h3>',
                f'<a class="asset-link" href="{escape(audio_url_tr)}">{escape(audio_url_tr)}</a>',
                f'</div>',
            ]'''
# Only do the replacement once
if 'chapter_audio_translated.get(idx)' not in epub_content:
    epub_content = epub_content.replace(old_body1, new_body1)

# Fix _build_chapter_insights body (chapter_mindmap_translated is already partially there, but using ch.get)
old_body2 = '''    if translated_lang:
        tr_mm = ch.get(f"mindmap_{translated_lang}_url")
        if tr_mm:
            is_tr_arabic   = translated_lang.lower() == 'ar'
            tr_mm_label    = 'OrOUSOOc OUU+USOc (U.OOOU.Oc)' if is_tr_arabic else 'Chapter Mind Map (Translated)'
            body_parts += [
                f'<div class="asset-box">',
                f'<h3>dY  {tr_mm_label}</h3>',
                f'<a class="asset-link" href="{escape(tr_mm)}">{escape(tr_mm)}</a>',
                f'</div>',
            ]'''
new_body2 = '''    if translated_lang and chapter_mindmap_translated:
        cm_tr = chapter_mindmap_translated.get(idx)
        tr_mm = cm_tr.get("url") if cm_tr else None
        if tr_mm:
            is_tr_arabic   = translated_lang.lower() == 'ar'
            tr_mm_label    = 'OrOUSOOc OUU+USOc (U.OOOU.Oc)' if is_tr_arabic else 'Chapter Mind Map (Translated)'
            body_parts += [
                f'<div class="asset-box">',
                f'<h3>dY  {tr_mm_label}</h3>',
                f'<a class="asset-link" href="{escape(tr_mm)}">{escape(tr_mm)}</a>',
                f'</div>',
            ]'''
if 'chapter_mindmap_translated.get(idx)' not in epub_content:
    epub_content = epub_content.replace(old_body2, new_body2)

# Fix inject_summary_into_epub signature
old_inj_sig = '''async def inject_summary_into_epub(
    epub_path: str,
    output_path: str,
    *,
    title: str,
    author: str,
    summary_text: str,
    language: str,
    translated_summary: str | None = None,
    translated_lang: str | None = None,
    cover_path: str | None = None,
    chapters: list[dict],
    chapter_audio: dict[int, str],
    chapter_mindmap: dict[int, dict],
    audio_url: str | None = None,
    mindmap_url: str | None = None,
    audio_url_translated: str | None = None,
    mindmap_url_translated: str | None = None,
) -> str:'''
new_inj_sig = '''async def inject_summary_into_epub(
    epub_path: str,
    output_path: str,
    *,
    title: str,
    author: str,
    summary_text: str,
    language: str,
    translated_summary: str | None = None,
    translated_lang: str | None = None,
    cover_path: str | None = None,
    chapters: list[dict],
    chapter_audio: dict[int, str],
    chapter_mindmap: dict[int, dict],
    chapter_audio_translated: dict[int, str] | None = None,
    chapter_mindmap_translated: dict[int, dict] | None = None,
    audio_url: str | None = None,
    mindmap_url: str | None = None,
    audio_url_translated: str | None = None,
    mindmap_url_translated: str | None = None,
) -> str:'''
epub_content = epub_content.replace(old_inj_sig, new_inj_sig)

# Fix call to _build_chapter_insights inside inject_summary_into_epub
old_call = '''            item = _build_chapter_insights(
                book,
                slug=f"chapter_{ch_idx}",
                ch=ch,
                language=language,
                translated_lang=translated_lang,
                chapter_audio=chapter_audio,
                chapter_mindmap=chapter_mindmap,
            )'''
new_call = '''            item = _build_chapter_insights(
                book,
                slug=f"chapter_{ch_idx}",
                ch=ch,
                language=language,
                translated_lang=translated_lang,
                chapter_audio=chapter_audio,
                chapter_mindmap=chapter_mindmap,
                chapter_audio_translated=chapter_audio_translated,
                chapter_mindmap_translated=chapter_mindmap_translated,
            )'''
epub_content = epub_content.replace(old_call, new_call)

with open(epub_path, 'w', encoding='utf-8') as f:
    f.write(epub_content)

# 2. Update orchestrator.py
orch_path = 'api/services/pipeline/orchestrator.py'
with open(orch_path, 'r', encoding='utf-8') as f:
    orch_content = f.read()

old_orch_call = '''                await inject_summary_into_epub(
                    src_path,
                    out_path,
                    title            = req.title or req.book_id,
                    author           = req.author or "",
                    summary_text       = full_summary,
                    language           = req.language,
                    translated_summary = translated_summary,
                    translated_lang    = target_lang,
                    cover_path       = _cover_for_epub,
                    chapters         = chapter_results,
                    chapter_audio    = chapter_audio,
                    chapter_mindmap  = chapter_mindmap,
                    audio_url        = (full_audio or {}).get("url"),
                    mindmap_url      = mindmap_url,
                    audio_url_translated   = (translated_audio or {}).get("url") if "translated_audio" in locals() else None,
                    mindmap_url_translated = translated_mindmap_url if "translated_mindmap_url" in locals() else None,
                )'''

new_orch_call = '''                await inject_summary_into_epub(
                    src_path,
                    out_path,
                    title            = req.title or req.book_id,
                    author           = req.author or "",
                    summary_text       = full_summary,
                    language           = req.language,
                    translated_summary = translated_summary,
                    translated_lang    = target_lang,
                    cover_path       = _cover_for_epub,
                    chapters         = chapter_results,
                    chapter_audio    = chapter_audio,
                    chapter_mindmap  = chapter_mindmap,
                    chapter_audio_translated = chapter_audio_translated if "chapter_audio_translated" in locals() else None,
                    chapter_mindmap_translated = chapter_mindmap_translated if "chapter_mindmap_translated" in locals() else None,
                    audio_url        = (full_audio or {}).get("url"),
                    mindmap_url      = mindmap_url,
                    audio_url_translated   = (translated_audio or {}).get("url") if "translated_audio" in locals() else None,
                    mindmap_url_translated = translated_mindmap_url if "translated_mindmap_url" in locals() else None,
                )'''

if 'chapter_audio_translated = chapter_audio_translated' not in orch_content:
    orch_content = orch_content.replace(old_orch_call, new_orch_call)

with open(orch_path, 'w', encoding='utf-8') as f:
    f.write(orch_content)

print("OK")
