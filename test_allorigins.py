import urllib.request
import urllib.parse
import json
import re

url = 'https://api.allorigins.win/get?url=' + urllib.parse.quote('https://www.youtube.com/watch?v=8Ij7A1VCB7I')
try:
    resp = urllib.request.urlopen(url).read().decode('utf-8')
    html = json.loads(resp)['contents']
    if 'Sign in to confirm' in html:
        print("AllOrigins is BLOCKED by YouTube.")
    else:
        match = re.search(r'"captionTracks":\s*(\[.*?\])', html)
        if match:
            print("Tracks found!")
            tracks = json.loads(match.group(1))
            for t in tracks:
                print(t.get('languageCode'), t.get('baseUrl')[:50])
        else:
            print("No tracks found but not blocked. HTML snippet:", html[:200])
except Exception as e:
    print("Error:", e)
