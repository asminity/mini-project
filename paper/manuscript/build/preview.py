"""Render pages of a built PDF to PNG for visual checking: preview.py <pdf> [dpi]"""
import sys
from pathlib import Path

import pymupdf

pdf = Path(sys.argv[1])
dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 100
doc = pymupdf.open(pdf)
for i, page in enumerate(doc, start=1):
    page.get_pixmap(dpi=dpi).save(pdf.with_name(f"{pdf.stem}_p{i}.png"))
print(len(doc), "pages")
