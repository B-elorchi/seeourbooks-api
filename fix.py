import pathlib
p = pathlib.Path('api/services/pipeline/epub.py')
text = p.read_text('utf-8')

import re

# 1. Front matter Full Audio
tr_audio_front = '''
    if audio_url_translated and translated_lang:
        is_tr_arabic   = translated_lang.lower() == 'ar'
        audio_label_tr  = '????? ?????? (?????)' if is_tr_arabic else 'Full Audio (Translated)'
        dl_label_tr     = '?????' if is_tr_arabic else 'Download MP3'
        body_parts.append(
            f'<div class="asset-box">'
            f'<h3>?? {audio_label_tr}</h3>'
            f'<a class="asset-link" href="{escape(audio_url_translated)}">{escape(audio_url_translated)}</a>'
            f'<p style="font-size:0.85em;color:#777;margin-top:0.4em">'
            f'<a href="{escape(audio_url_translated)}">{dl_label_tr} ?</a></p>'
            f'</div>'
        )
'''
if 'audio_url_translated and translated_lang' not in text:
    text = re.sub(r'(    # -- Mind map [^\n]+)', lambda m: tr_audio_front.lstrip('\n') + '\n' + m.group(1), text)

# 2. Front matter Mind Map
tr_mm_front = '''
    if mindmap_url_translated and translated_lang:
        is_tr_arabic   = translated_lang.lower() == 'ar'
        mm_label_tr = '????? ????? (??????)' if is_tr_arabic else 'Mind Map (Translated)'
        body_parts.append(
            f'<div class="asset-box">'
            f'<h3>?? {mm_label_tr}</h3>'
            f'<a class="asset-link" href="{escape(mindmap_url_translated)}">{escape(mindmap_url_translated)}</a>'
            f'</div>'
        )
'''
if 'mindmap_url_translated and translated_lang' not in text:
    text = re.sub(r'(    html = _xhtml_page\()', lambda m: tr_mm_front.lstrip('\n') + '\n' + m.group(1), text)

# 3. Chapter Insights Audio
tr_audio_ch = '''
    if translated_lang:
        tr_audio = ch.get(f"audio_{translated_lang}")
        if tr_audio:
            is_tr_arabic   = translated_lang.lower() == 'ar'
            tr_audio_label = '??? ????? (?????)' if is_tr_arabic else 'Chapter Audio (Translated)'
            dl_label_tr    = '?????' if is_tr_arabic else 'Download MP3'
            body_parts += [
                f'<div class="asset-box">',
                f'<h3>?? {tr_audio_label}</h3>',
                f'<a class="asset-link" href="{escape(tr_audio)}">{escape(tr_audio)}</a>',
                f'<p style="font-size:0.85em;color:#777;margin-top:0.4em"><a href="{escape(tr_audio)}">{dl_label_tr} ?</a></p>',
                f'</div>',
            ]
'''
if 'tr_audio = ch.get(f"audio_{translated_lang}")' not in text:
    text = re.sub(r'(def _build_chapter_insights.*?)(    # -- Mind map)', lambda m: m.group(1) + tr_audio_ch.lstrip('\n') + '\n' + m.group(2), text, flags=re.DOTALL)

# 4. Chapter Insights Mind Map
tr_mm_ch = '''
    if translated_lang:
        tr_mm = ch.get(f"mindmap_{translated_lang}_url")
        if tr_mm:
            is_tr_arabic   = translated_lang.lower() == 'ar'
            tr_mm_label    = '????? ????? (??????)' if is_tr_arabic else 'Chapter Mind Map (Translated)'
            body_parts += [
                f'<div class="asset-box">',
                f'<h3>?? {tr_mm_label}</h3>',
                f'<a class="asset-link" href="{escape(tr_mm)}">{escape(tr_mm)}</a>',
                f'</div>',
            ]
'''
if 'tr_mm = ch.get(f"mindmap_{translated_lang}_url")' not in text:
    text = re.sub(r'(def _build_chapter_insights.*?)(    html = _xhtml_page)', lambda m: m.group(1) + tr_mm_ch.lstrip('\n') + '\n' + m.group(2), text, flags=re.DOTALL)

p.write_text(text, 'utf-8')
print('OK')
