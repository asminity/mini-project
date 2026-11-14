"""Look up candidate references in Crossref and DBLP so the bibliography is built
from retrieved records, not from memory. Output: results/reference_lookup.json"""

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

QUERIES = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
OUT = Path(sys.argv[2])
UA = {"User-Agent": "reference-check/1.0"}


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            time.sleep(2 + 2 * attempt)
    return {"error": err}


def crossref(q):
    d = get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q))
    out = []
    for it in d.get("message", {}).get("items", []):
        out.append({"title": (it.get("title") or [""])[0],
                    "authors": [f"{a.get('given', '')} {a.get('family', '')}".strip()
                                for a in it.get("author", [])][:12],
                    "venue": (it.get("container-title") or [""])[0],
                    "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "volume": it.get("volume"), "issue": it.get("issue"), "pages": it.get("page"),
                    "doi": it.get("DOI"), "type": it.get("type"), "publisher": it.get("publisher")})
    return out or d


def dblp(q):
    d = get("https://dblp.org/search/publ/api?format=json&h=3&q=" + urllib.parse.quote(q))
    out = []
    for h in d.get("result", {}).get("hits", {}).get("hit", []) or []:
        i = h["info"]
        au = i.get("authors", {}).get("author", [])
        au = [au] if isinstance(au, dict) else au
        out.append({"title": i.get("title"), "authors": [a.get("text") for a in au][:12],
                    "venue": i.get("venue"), "year": i.get("year"), "volume": i.get("volume"),
                    "pages": i.get("pages"), "doi": i.get("doi"), "ee": i.get("ee"),
                    "type": i.get("type")})
    return out or d


res = {}
for key, q in QUERIES.items():
    res[key] = {"query": q, "crossref": crossref(q), "dblp": dblp(q)}
    print(key, "ok", flush=True)
    time.sleep(1.0)
OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
