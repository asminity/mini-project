"""Fallback build of the manuscript for machines without a TeX installation.

The reference PDF (paper.pdf) is produced with LaTeX:

    tectonic -X compile main.tex --outdir build/tectonic --keep-logs

This script is only an approximation for when no TeX engine is available. It
renders the same source to an IEEE-style two-column HTML page (MathML
equations, hyphenated justified text) and prints it with Microsoft Edge.

    python build.py            # main.tex -> build/main.html -> build/paper_browser.pdf

It understands only the LaTeX subset used in main.tex, and it re-orders the
bibliography in main.tex by first citation (IEEE numbering), in place.
"""

from __future__ import annotations

import html
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import latex2mathml.converter as l2m
import pyphen

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BUILD = HERE / "build"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]
HYPH = pyphen.Pyphen(lang="en_US")
ACCENTS = {"'": "́", "`": "̀", "~": "̃", "^": "̂", '"': "̈",
           "v": "̌", ".": "̇", "c": "̧"}
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII",
         "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

CSS = r"""
@page { size: 8.5in 11in; margin: 0.75in 0.625in 1in 0.625in; }
* { box-sizing: border-box; }
html { font-size: 10pt; }
body { font-family: "Times New Roman", Times, serif; font-size: 10pt; line-height: 1.16;
       color: #000; margin: 0; }
math { font-family: "Cambria Math", "STIX Two Math", serif; font-size: 1.0em; }
mi, mn, mtext { font-family: "Times New Roman", "Cambria Math", serif; }
mi { text-transform: none; font-style: italic; }
mi[mathvariant="normal"], mi[mathvariant="bold"], mi.fn { font-style: normal; }
mi[mathvariant="bold"] { font-weight: bold; }
h1.title { font-size: 23pt; font-weight: normal; text-align: center; line-height: 1.12;
           margin: 0 0.4in 12pt; }
.authors { text-align: center; font-size: 11pt; margin: 0 0 4pt; }
.affil { text-align: center; font-size: 10pt; font-style: italic; margin: 0 0 18pt; }
.cols { column-count: 2; column-gap: 0.25in; text-align: justify; }
p { margin: 0; text-indent: 1em; orphans: 2; widows: 2; }
p.noindent, .abstract p, li p { text-indent: 0; }
.abstract { font-weight: bold; font-size: 9pt; line-height: 1.12; margin-bottom: 6pt; }
.abstract .lead { font-style: italic; }
.keywords { font-weight: bold; font-style: italic; font-size: 9pt; line-height: 1.12;
            margin-bottom: 8pt; text-indent: 1em; }
h2 { font-size: 10pt; font-weight: normal; font-variant: small-caps; text-align: center;
     margin: 11pt 0 5pt; letter-spacing: 0.02em; break-after: avoid; }
h3 { font-size: 10pt; font-weight: normal; font-style: italic; text-align: left;
     margin: 8pt 0 4pt; break-after: avoid; }
h4 { display: inline; font-size: 10pt; font-weight: normal; font-style: italic; }
ul, ol { margin: 3pt 0 3pt 0; padding-left: 1.7em; }
li { margin: 1pt 0; }
.eq { display: flex; align-items: center; margin: 6pt 0; text-indent: 0; break-inside: avoid; }
.eq .m { flex: 1; text-align: center; overflow: hidden; }
.eq .n { width: 2.4em; text-align: right; }
figure, .tbl { margin: 8pt 0 10pt; break-inside: avoid; text-indent: 0; }
.wide { column-span: all; margin: 4pt 0 12pt; }
figure img { display: block; margin: 0 auto; max-width: 100%; }
figcaption { font-size: 8pt; line-height: 1.15; text-align: justify; margin-top: 5pt; }
.tbl .cap { font-size: 8pt; text-align: center; line-height: 1.2; margin-bottom: 4pt; }
.tbl .cap .lab { display: block; font-variant: normal; letter-spacing: 0.03em; }
.tbl .cap .txt { font-variant: small-caps; }
table { border-collapse: collapse; margin: 0 auto; font-size: 8pt; line-height: 1.18; }
td, th { padding: 1.3pt 4.2pt; font-weight: normal; white-space: nowrap; }
td.wrap { white-space: normal; }
tr.top td { border-top: 0.9pt solid #000; padding-top: 2.6pt; }
tr.mid td { border-top: 0.5pt solid #000; padding-top: 2.6pt; }
tr.bot td { border-bottom: 0.9pt solid #000; padding-bottom: 2.6pt; }
td.cm { border-bottom: 0.4pt solid #000; }
.l { text-align: left; } .c { text-align: center; } .r { text-align: right; }
.tnote { font-size: 7.5pt; text-align: justify; margin: 3pt auto 0; line-height: 1.15; }
.refs { font-size: 8pt; line-height: 1.14; }
.refs .ref { display: flex; margin: 0 0 1.5pt; text-indent: 0; }
.refs .ref .k { flex: 0 0 2.3em; }
.refs .ref .t { flex: 1; text-align: justify; word-break: normal; overflow-wrap: anywhere; }
code, .tt { font-family: "Courier New", monospace; font-size: 0.88em; }
a { color: inherit; text-decoration: none; }
.sc { font-variant: small-caps; }
"""


# --------------------------------------------------------------------------- #
# Bibliography ordering (rewrites main.tex in place when the order changes)
# --------------------------------------------------------------------------- #
def reorder_bibliography(src: str) -> str:
    m = re.search(r"(\\begin\{thebibliography\}\{\d+\})(.*?)(\\end\{thebibliography\})", src, re.S)
    if not m:
        return src
    body = strip_comments(resolve_inputs(src[:m.start()]))
    cited = []
    for group in re.findall(r"\\cite\{([^}]*)\}", body):
        for key in group.split(","):
            if key.strip() not in cited:
                cited.append(key.strip())
    items = re.split(r"(?=\\bibitem\{)", m.group(2).strip())
    items = {re.match(r"\\bibitem\{([^}]*)\}", it).group(1): it.strip() for it in items if it.strip()}
    missing = [k for k in cited if k not in items]
    unused = [k for k in items if k not in cited]
    if missing:
        sys.exit(f"Citation keys without a \\bibitem: {missing}")
    if unused:
        print(f"[warn] bibliography entries never cited (dropped from PDF order end): {unused}")
    ordered = [items[k] for k in cited] + [items[k] for k in unused]
    new = m.group(1) + "\n\n" + "\n\n".join(ordered) + "\n\n" + m.group(3)
    return src[:m.start()] + new + src[m.end():]


# --------------------------------------------------------------------------- #
# Inline conversion
# --------------------------------------------------------------------------- #
def balanced(s: str, i: int) -> tuple[str, int]:
    """s[i] == '{' -> (content, index after the matching '}')."""
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced braces near: " + s[i:i + 60])


def hyphenate(text: str) -> str:
    def fix(m):
        w = m.group(0)
        return HYPH.inserted(w, "\u00ad") if len(w) > 6 else w
    return re.sub(r"[A-Za-z]{7,}", fix, text)


class Doc:
    def __init__(self):
        self.labels: dict[str, str] = {}
        self.cites: dict[str, int] = {}
        self.sec = self.sub = self.subsub = self.eq = self.fig = self.tab = 0

    # ---- math ----
    def math(self, tex: str, display: bool = False) -> str:
        tex = tex.strip()
        tex = re.sub(r"\\label\{[^}]*\}", "", tex)
        tex = tex.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
        tex = re.sub(r"\\mathbbm\{1\}", r"\\mathbb{1}", tex)
        tex = re.sub(r"\\operatorname\{([^}]*)\}", r"\\mathrm{\1}", tex)
        tex = tex.replace(r"\lVert", r"\|").replace(r"\rVert", r"\|")
        tex = tex.replace(r"\arg\max", r"\mathrm{arg\,max}").replace(r"\arg\min", r"\mathrm{arg\,min}")
        out = l2m.convert(tex, display="block" if display else "inline")
        # Function names (max, log, ...) and multi-letter identifiers stay upright.
        out = re.sub(r"<mi>([A-Za-z]{2,})</mi>", r'<mi class="fn">\1</mi>', out)
        return out

    # ---- inline text ----
    def inline(self, s: str) -> str:
        out, i, n = [], 0, len(s)

        def text(t: str) -> str:
            t = t.replace("---", "\u2014").replace("--", "\u2013")
            t = t.replace("``", "\u201c").replace("''", "\u201d").replace("`", "\u2018").replace("'", "\u2019")
            t = t.replace("~", "\u00a0")
            return hyphenate(html.escape(t, quote=False))

        buf = []

        def flush():
            if buf:
                out.append(text("".join(buf)))
                buf.clear()

        while i < n:
            ch = s[i]
            if ch == "$":
                j = s.index("$", i + 1)
                flush()
                out.append(self.math(s[i + 1:j]))
                i = j + 1
            elif ch == "\\":
                m = re.match(r"\\([A-Za-z]+\*?|.)", s[i:])
                cmd = m.group(1)
                i += m.end()
                if cmd in ("%", "&", "_", "#", "$", "{", "}"):
                    buf.append(cmd)
                elif cmd in ACCENTS and i < n and (cmd in "'`~^\"." or s[i] in "{ "):
                    while i < n and s[i] == " ":
                        i += 1
                    if s[i] == "{":
                        base, i = balanced(s, i)
                    else:
                        base, i = s[i], i + 1
                    buf.append(unicodedata.normalize("NFC", base + ACCENTS[cmd]))
                elif cmd == ",":
                    buf.append("\u2009")
                elif cmd == " ":
                    buf.append(" ")
                elif cmd == "\\":
                    flush(); out.append("<br>")
                elif cmd in ("textbf", "emph", "textit", "texttt", "textsc", "mbox", "text"):
                    arg, i = balanced(s, i)
                    flush()
                    tag = {"textbf": ("<b>", "</b>"), "emph": ("<i>", "</i>"), "textit": ("<i>", "</i>"),
                           "texttt": ('<span class="tt">', "</span>"), "textsc": ('<span class="sc">', "</span>"),
                           "mbox": ('<span style="white-space:nowrap">', "</span>"), "text": ("", "")}[cmd]
                    inner = html.escape(arg, quote=False) if cmd == "texttt" else self.inline(arg)
                    if cmd == "texttt":
                        inner = inner.replace("\\_", "_").replace("\\", "")
                    out.append(tag[0] + inner + tag[1])
                elif cmd == "cite":
                    arg, i = balanced(s, i)
                    flush()
                    nums = sorted(self.cites[k.strip()] for k in arg.split(","))
                    out.append("".join(f"[{x}]" for x in self._ranges(nums)))
                elif cmd in ("ref", "eqref"):
                    arg, i = balanced(s, i)
                    flush()
                    val = self.labels.get(arg, "??")
                    if val == "??":
                        print(f"[warn] undefined reference {arg}")
                    out.append(f"({val})" if cmd == "eqref" else val)
                elif cmd == "url":
                    arg, i = balanced(s, i)
                    flush()
                    out.append(f'<span class="tt">{html.escape(arg)}</span>')
                elif cmd == "label":
                    _, i = balanced(s, i)
                elif cmd in ("noindent", "centering", "footnotesize", "small", "scriptsize", "normalsize",
                             "hfill", "par", "quad", "qquad", "IEEEPARstart", "protect"):
                    if cmd in ("quad", "qquad"):
                        buf.append("\u2003")
                elif cmd == "ldots" or cmd == "dots":
                    buf.append("\u2026")
                elif cmd == "textasciitilde":
                    buf.append("~")
                elif cmd == "times":
                    buf.append("\u00d7")
                else:
                    print(f"[warn] unknown inline command \\{cmd}")
            elif ch in "{}":
                i += 1
            else:
                buf.append(ch)
                i += 1
        flush()
        return "".join(out)

    @staticmethod
    def _ranges(nums):
        res, k = [], 0
        while k < len(nums):
            j = k
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            res.append(f"{nums[k]}\u2013{nums[j]}" if j - k >= 2 else None)
            if res[-1] is None:
                res.pop()
                res.extend(str(x) for x in nums[k:j + 1])
            k = j + 1
        return res


# --------------------------------------------------------------------------- #
# Block conversion
# --------------------------------------------------------------------------- #
def resolve_inputs(src: str) -> str:
    """Inline \\input{file} (sections live in separate .tex files next to main.tex)."""
    def load(m):
        path = HERE / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")
        return resolve_inputs(path.read_text(encoding="utf-8"))
    return re.sub(r"\\input\{([^}]*)\}", load, src)


def strip_comments(src: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in src.splitlines())


def env_blocks(body: str):
    """Split the body into ('env', name, content) and ('text', None, content) pieces."""
    pattern = re.compile(r"\\begin\{(abstract|IEEEkeywords|equation|itemize|enumerate|table\*?|figure\*?|thebibliography)\}")
    pos = 0
    while True:
        m = pattern.search(body, pos)
        if not m:
            yield ("text", None, body[pos:])
            return
        yield ("text", None, body[pos:m.start()])
        name = m.group(1)
        end = body.index(f"\\end{{{name}}}", m.end())
        yield ("env", name, body[m.end():end])
        pos = end + len(f"\\end{{{name}}}")


def number_everything(doc: Doc, body: str) -> None:
    """First pass: assign numbers to labels and citations."""
    bib = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", body, re.S)
    if bib:
        for k, key in enumerate(re.findall(r"\\bibitem\{([^}]*)\}", bib.group(0)), start=1):
            doc.cites[key] = k
    sec = sub = subsub = eq = fig = tab = 0
    token = re.compile(r"\\(section|subsection|subsubsection)\{|\\begin\{(equation|figure\*?|table\*?)\}|\\label\{([^}]*)\}")
    current = ""
    for m in token.finditer(body):
        if m.group(1) == "section":
            sec += 1; sub = subsub = 0; current = ROMAN[sec]
        elif m.group(1) == "subsection":
            sub += 1; subsub = 0; current = f"{ROMAN[sec]}-{chr(64 + sub)}"
        elif m.group(1) == "subsubsection":
            subsub += 1; current = f"{ROMAN[sec]}-{chr(64 + sub)}{subsub}"
        elif m.group(2) == "equation":
            eq += 1; current = str(eq)
        elif m.group(2) and m.group(2).startswith("figure"):
            fig += 1; current = str(fig)
        elif m.group(2) and m.group(2).startswith("table"):
            tab += 1; current = ROMAN[tab]
        elif m.group(3):
            doc.labels[m.group(3)] = current


def convert_tabular(doc: Doc, tex: str) -> str:
    start = tex.index("\\begin{tabular}") + len("\\begin{tabular}")
    spec_raw, body_start = balanced(tex, start)
    body_tex = tex[body_start:tex.index("\\end{tabular}")]
    spec = re.sub(r"@\{[^}]*\}|\|", "", spec_raw)
    aligns = re.findall(r"p\{[^}]*\}|[lcr]", spec)
    rows_html, pending = [], "top"
    raw_rows = [r.strip() for r in re.split(r"\\\\(?:\[[^\]]*\])?", body_tex)]
    for raw in raw_rows:
        cls = None
        while True:
            mm = re.match(r"\\(toprule|midrule|bottomrule|hline|addlinespace)(\[[^\]]*\])?\s*|\\cmidrule(\([^)]*\))?\{[^}]*\}\s*", raw)
            if not mm:
                break
            if mm.group(1) in ("toprule",):
                cls = "top"
            elif mm.group(1) in ("midrule", "hline"):
                cls = "mid"
            elif mm.group(1) == "bottomrule":
                if rows_html:
                    rows_html[-1] = rows_html[-1].replace("<tr", '<tr class="bot"', 1) if 'class="' not in rows_html[-1][:12] \
                        else rows_html[-1].replace('class="', 'class="bot ', 1)
            raw = raw[mm.end():]
        if not raw.strip():
            continue
        cells, col = [], 0
        for cell in re.split(r"(?<!\\)&", raw):
            cell = cell.strip()
            span, align = 1, None
            mc = re.match(r"\\multicolumn\{(\d+)\}\{([^}]*)\}", cell)
            if mc:
                span = int(mc.group(1)); align = re.sub(r"[|@{}]", "", mc.group(2))[:1]
                content, _ = balanced(cell, mc.end())
                cell = content
            if align is None:
                a = aligns[col] if col < len(aligns) else "l"
                align = "l" if a.startswith("p") else a
            wrap = " wrap" if (col < len(aligns) and aligns[col].startswith("p")) else ""
            cells.append(f'<td class="{align}{wrap}"' + (f' colspan="{span}"' if span > 1 else "") + f">{doc.inline(cell)}</td>")
            col += span
        rows_html.append(f'<tr class="{cls}">' + "".join(cells) + "</tr>" if cls else "<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(rows_html) + "</table>"


def convert(src: str) -> str:
    src = strip_comments(resolve_inputs(src))
    title = balanced(src, src.index("\\title{") + 6)[0]
    body = src[src.index("\\maketitle") + len("\\maketitle"):src.index("\\end{document}")]
    doc = Doc()
    number_everything(doc, body)
    authors = re.search(r"\\IEEEauthorblockN\{", src)
    a_name = balanced(src, authors.end() - 1)[0] if authors else ""
    aff = re.search(r"\\IEEEauthorblockA\{", src)
    a_aff = balanced(src, aff.end() - 1)[0] if aff else ""

    head = [f'<h1 class="title">{doc.inline(re.sub(chr(92) * 2 + r"\\s*", " ", title))}</h1>',
            f'<div class="authors">{doc.inline(a_name)}</div>',
            f'<div class="affil">{doc.inline(a_aff.replace(chr(92) * 2, " \u00b7 "))}</div>']
    out = []
    sec = sub = subsub = eq = fig = tab = 0

    def paragraphs(text: str):
        nonlocal sec, sub, subsub
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            starred = re.match(r"\\section\*\{", para)
            if starred:      # unnumbered heading, e.g. the reproducibility statement
                titl, end = balanced(para, starred.end() - 1)
                out.append(f"<h2>{doc.inline(titl)}</h2>")
                para = para[end:].lstrip()
            m = re.match(r"\\(section|subsection|subsubsection)\{", para)
            while m:
                titl, end = balanced(para, m.end() - 1)
                rest = para[end:].lstrip()
                rest = re.sub(r"^\\label\{[^}]*\}\s*", "", rest)
                if m.group(1) == "section":
                    sec += 1; sub = subsub = 0
                    out.append(f"<h2>{ROMAN[sec]}. {doc.inline(titl)}</h2>")
                elif m.group(1) == "subsection":
                    sub += 1; subsub = 0
                    out.append(f"<h3>{chr(64 + sub)}. {doc.inline(titl)}</h3>")
                else:
                    subsub += 1
                    out.append(f'<p><h4>{subsub}) {doc.inline(titl)}:</h4> {doc.inline(rest)}</p>')
                    rest = ""
                para = rest
                m = re.match(r"\\(section|subsection|subsubsection)\{", para)
            if para:
                cls = ' class="noindent"' if para.startswith("\\noindent") else ""
                out.append(f"<p{cls}>{doc.inline(para)}</p>")

    for kind, name, content in env_blocks(body):
        if kind == "text":
            paragraphs(content)
            continue
        if name == "abstract":
            out.append(f'<div class="abstract"><p><span class="lead">Abstract</span>\u2014{doc.inline(content.strip())}</p></div>')
        elif name == "IEEEkeywords":
            out.append(f'<div class="keywords">Index Terms\u2014{doc.inline(content.strip())}</div>')
        elif name == "equation":
            eq += 1
            out.append(f'<div class="eq"><div class="m">{doc.math(content, True)}</div><div class="n">({eq})</div></div>')
        elif name in ("itemize", "enumerate"):
            tag = "ul" if name == "itemize" else "ol"
            items = [x.strip() for x in re.split(r"\\item\b", content) if x.strip()]
            out.append(f"<{tag}>" + "".join(f"<li>{doc.inline(x)}</li>" for x in items) + f"</{tag}>")
        elif name.startswith("table"):
            tab += 1
            cap = balanced(content, content.index("\\caption{") + 8)[0]
            note = re.search(r"%%NOTE-BEGIN|\\tablenote\{", content)
            note_html = ""
            mnote = re.search(r"\\tablenote\{", content)
            if mnote:
                note_txt, _ = balanced(content, mnote.end() - 1)
                note_html = f'<div class="tnote">{doc.inline(note_txt)}</div>'
            wide = " wide" if name.endswith("*") else ""
            out.append(f'<div class="tbl{wide}"><div class="cap"><span class="lab">TABLE {ROMAN[tab]}</span>'
                       f'<span class="txt">{doc.inline(cap)}</span></div>{convert_tabular(doc, content)}{note_html}</div>')
        elif name.startswith("figure"):
            fig += 1
            cap = balanced(content, content.index("\\caption{") + 8)[0]
            imgs = re.findall(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]*)\}", content)
            wide = " wide" if name.endswith("*") else ""
            img_html = ""
            for opt, path in imgs:
                w = re.search(r"width=([\d.]*)\\(linewidth|textwidth|columnwidth)", opt or "")
                pct = float(w.group(1) or 1) * 100 if w else 100
                svg = Path(path).with_suffix(".svg").as_posix()
                img_html += f'<img src="../{svg}" style="width:{pct:.0f}%">'
            out.append(f'<figure class="{wide.strip()}">{img_html}<figcaption>Fig. {fig}.\u2002{doc.inline(cap)}</figcaption></figure>')
        elif name == "thebibliography":
            out.append("<h2>References</h2><div class=\"refs\">")
            for it in re.split(r"(?=\\bibitem\{)", content):
                it = it.strip()
                mm = re.match(r"\\bibitem\{([^}]*)\}", it)
                if mm:
                    txt = re.sub(r"\\newblock\s*", "", it[mm.end():]).strip()
                    out.append(f'<div class="ref"><span class="k">[{doc.cites[mm.group(1)]}]</span>'
                               f'<span class="t">{doc.inline(txt)}</span></div>')
            out.append("</div>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>"
            + html.escape(re.sub(r"\\\\\s*", " ", title)) + f"</title><style>{CSS}</style></head><body>"
            + "".join(head) + '<div class="cols">' + "\n".join(out) + "</div></body></html>")


def main():
    src = TEX.read_text(encoding="utf-8")
    ordered = reorder_bibliography(src)
    if ordered != src:
        TEX.write_text(ordered, encoding="utf-8")
        src = ordered
        print("[info] bibliography re-ordered by first citation")
    BUILD.mkdir(exist_ok=True)
    page = BUILD / "main.html"
    page.write_text(convert(src), encoding="utf-8")
    edge = next((e for e in EDGE if Path(e).exists()), None)
    if not edge:
        sys.exit("Microsoft Edge not found; open build/main.html and print to PDF manually.")
    pdf = BUILD / "paper_browser.pdf"
    print_pdf(edge, page, pdf)
    print(f"[ok] {pdf}")


def print_pdf(edge: str, page: Path, pdf: Path) -> None:
    """Headless print with a private browser profile so a running Edge cannot block it."""
    profile = BUILD / "edge-profile"
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--no-first-run",
                    "--disable-extensions", f"--user-data-dir={profile}", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True,
                   timeout=300)


if __name__ == "__main__":
    main()
