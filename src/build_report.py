"""Render the Markdown report and figures into a paginated A4 PDF.

Run from the repository root:

    python src/build_report.py
    python src/build_report.py --input report/report.md --output report/report.pdf

Reads report/report.md and any relative figures it embeds, writes the PDF
atomically, and verifies that the page count stays within --max-pages (default 2).
"""

from __future__ import annotations

import argparse
import html
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import ROOT

DEFAULT_INPUT = ROOT / "report" / "report.md"
DEFAULT_OUTPUT = ROOT / "report" / "report.pdf"
DEFAULT_HEADER = "TEXT MINING / M4 EVALUATION"
DEFAULT_FOOTER = "RAG for technical documentation"
DEFAULT_AUTHOR = "RAG for Text Mining project team"
DEFAULT_SUBJECT = "Milestone M4 evaluation of RAG over Brazilian Portuguese technical documentation"
DEFAULT_MAX_PAGES = 2

INK = colors.HexColor("#152739")
BODY = colors.HexColor("#1f2d3d")
TEAL = colors.HexColor("#007c78")
BLUE = colors.HexColor("#23486b")
MUTED = colors.HexColor("#4b5d6d")
HEADER_BG = colors.HexColor("#edf4f7")
ROW_HIGHLIGHT = colors.HexColor("#f4f8fa")
RULE = colors.HexColor("#cfdbe4")

_INLINE_TOKEN = re.compile(r"`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)|\*\*([^*]+)\*\*|\*([^*]+)\*")
_IMAGE_LINE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)$")
_ALIGN_CELL = re.compile(r"^:?-{3,}:?$")
_NUMBERED_ITEM = re.compile(r"^(\d+\.)\s+(.*)$")


@dataclass(frozen=True)
class FontSet:
    regular: str
    bold: str
    italic: str
    bold_italic: str
    mono: str


def _font_search_dirs() -> list[Path]:
    dirs = [
        Path("/usr/share/fonts/truetype/lato"),
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".local" / "share" / "fonts",
    ]
    try:
        import matplotlib

        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:
        pass
    return [d for d in dirs if d.is_dir()]


def _find_font(filename: str, search_dirs: list[Path]) -> Path | None:
    for base in search_dirs:
        direct = base / filename
        if direct.is_file():
            return direct
    return None


def _register_ttfont(name: str, path: Path) -> None:
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(name, str(path)))


def register_fonts() -> FontSet:
    """Register Lato + DejaVuSansMono when present, falling back to DejaVuSans or Helvetica."""
    dirs = _font_search_dirs()

    mono_path = _find_font("DejaVuSansMono.ttf", dirs)
    if mono_path is not None:
        _register_ttfont("DejaVuSansMono", mono_path)
        mono = "DejaVuSansMono"
    else:
        mono = "Courier"

    candidates = [
        (
            "Lato",
            ("Lato-Regular", "Lato-Regular.ttf"),
            ("Lato-Bold", "Lato-Bold.ttf"),
            ("Lato-Italic", "Lato-Italic.ttf"),
            ("Lato-BoldItalic", "Lato-BoldItalic.ttf"),
        ),
        (
            "DejaVuSans",
            ("DejaVuSans", "DejaVuSans.ttf"),
            ("DejaVuSans-Bold", "DejaVuSans-Bold.ttf"),
            ("DejaVuSans-Oblique", "DejaVuSans-Oblique.ttf"),
            ("DejaVuSans-BoldOblique", "DejaVuSans-BoldOblique.ttf"),
        ),
    ]
    for family, reg, bold, italic, bold_italic in candidates:
        paths = [_find_font(filename, dirs) for _, filename in (reg, bold, italic, bold_italic)]
        if all(p is not None for p in paths):
            for (font_name, _), path in zip((reg, bold, italic, bold_italic), paths):
                _register_ttfont(font_name, path)
            pdfmetrics.registerFontFamily(
                family,
                normal=reg[0],
                bold=bold[0],
                italic=italic[0],
                boldItalic=bold_italic[0],
            )
            return FontSet(reg[0], bold[0], italic[0], bold_italic[0], mono)

    return FontSet("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique", mono)


def inline_markup(text: str, fonts: FontSet) -> str:
    """Convert inline Markdown (`code`, **bold**, *italic*, [label](target)) to ReportLab XML."""
    parts: list[str] = []
    pos = 0
    for match in _INLINE_TOKEN.finditer(text):
        if match.start() > pos:
            parts.append(html.escape(text[pos : match.start()]))
        code, link_label, link_target, bold, italic = match.groups()
        if code is not None:
            parts.append(f'<font name="{fonts.mono}" size="8.2">{html.escape(code)}</font>')
        elif link_label is not None:
            inner = inline_markup(link_label, fonts)
            if link_target.startswith(("http://", "https://")):
                parts.append(f'<link href="{html.escape(link_target, quote=True)}" color="#23486b">{inner}</link>')
            else:
                parts.append(f'<font color="#23486b">{inner}</font>')
        elif bold is not None:
            parts.append(f"<b>{inline_markup(bold, fonts)}</b>")
        elif italic is not None:
            parts.append(f"<i>{inline_markup(italic, fonts)}</i>")
        pos = match.end()
    if pos < len(text):
        parts.append(html.escape(text[pos:]))
    return "".join(parts)


def _strip_markup(text: str) -> str:
    return re.sub(r"[`*]|\[([^\]]+)\]\([^)]+\)", r"\1", text)


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _parse_alignments(line: str, ncols: int) -> list[str] | None:
    cells = _split_table_row(line)
    if len(cells) != ncols or not all(_ALIGN_CELL.match(c) for c in cells):
        return None
    aligns = []
    for cell in cells:
        if cell.startswith(":") and cell.endswith(":"):
            aligns.append("CENTER")
        elif cell.endswith(":"):
            aligns.append("RIGHT")
        else:
            aligns.append("LEFT")
    return aligns


def parse_markdown(text: str) -> list[dict]:
    """Parse a clean subset of Markdown into structured blocks for PDF rendering."""
    lines = text.splitlines()
    blocks: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped in ("---", "<!-- pagebreak -->"):
            blocks.append({"type": "pagebreak"})
            i += 1
            continue
        if stripped.startswith("### "):
            blocks.append({"type": "h3", "text": stripped[4:].strip()})
            i += 1
            continue
        if stripped.startswith("## "):
            blocks.append({"type": "h2", "text": stripped[3:].strip()})
            i += 1
            continue
        if stripped.startswith("# "):
            blocks.append({"type": "h1", "text": stripped[2:].strip()})
            i += 1
            continue

        img_match = _IMAGE_LINE.match(stripped)
        if img_match:
            alt, src = img_match.groups()
            i += 1
            caption = None
            j = i
            while j < n and not lines[j].strip():
                j += 1
            if j < n:
                cand = lines[j].strip()
                if cand.startswith("*") and cand.endswith("*") and not cand.startswith("**") and len(cand) > 2:
                    caption = cand[1:-1].strip()
                    i = j + 1
            blocks.append({"type": "figure", "alt": alt, "src": src, "caption": caption})
            continue

        if stripped.startswith("|") and i + 1 < n:
            headers = _split_table_row(stripped)
            aligns = _parse_alignments(lines[i + 1].strip(), len(headers))
            if aligns is not None:
                i += 2
                rows = []
                while i < n and lines[i].strip().startswith("|"):
                    row = _split_table_row(lines[i])
                    if len(row) != len(headers):
                        raise ValueError(f"table row at line {i + 1} has {len(row)} columns, expected {len(headers)}")
                    rows.append(row)
                    i += 1
                blocks.append({"type": "table", "headers": headers, "alignments": aligns, "rows": rows})
                continue

        if stripped.startswith("- "):
            items = []
            while i < n and lines[i].strip().startswith("- "):
                items.append(lines[i].strip()[2:].strip())
                i += 1
            blocks.append({"type": "bullets", "items": items})
            continue

        num_match = _NUMBERED_ITEM.match(stripped)
        if num_match:
            items = []
            while i < n:
                m = _NUMBERED_ITEM.match(lines[i].strip())
                if not m:
                    break
                items.append((m.group(1), m.group(2).strip()))
                i += 1
            blocks.append({"type": "numbered", "items": items})
            continue

        para_lines = [stripped]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if (
                not nxt
                or nxt.startswith(("#", "|", "- ", "!["))
                or nxt in ("---", "<!-- pagebreak -->")
                or _NUMBERED_ITEM.match(nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        joined = " ".join(para_lines)
        is_subtitle = (
            len(blocks) == 1
            and blocks[0]["type"] == "h1"
            and joined.startswith("**")
            and joined.endswith("**")
            and joined.count("**") == 2
        )
        if is_subtitle:
            blocks.append({"type": "subtitle", "text": joined[2:-2].strip()})
        else:
            blocks.append({"type": "paragraph", "text": joined})
    return blocks


def _build_styles(fonts: FontSet) -> dict[str, ParagraphStyle]:
    return {
        "h1": ParagraphStyle(
            "ReportH1", fontName=fonts.bold, fontSize=20.5, leading=24.5, textColor=INK, spaceAfter=3, keepWithNext=True,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", fontName=fonts.bold, fontSize=10.0, leading=13.5, textColor=MUTED, spaceAfter=9, keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2", fontName=fonts.bold, fontSize=11.0, leading=14.5, textColor=TEAL, spaceBefore=8, spaceAfter=3, keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ReportH3", fontName=fonts.bold, fontSize=9.6, leading=12.8, textColor=INK, spaceBefore=6, spaceAfter=2.5, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody", fontName=fonts.regular, fontSize=9.0, leading=12.4, textColor=BODY, spaceAfter=4.5,
        ),
        "caption": ParagraphStyle(
            "ReportCaption", fontName=fonts.italic, fontSize=8.0, leading=10.6, textColor=MUTED, spaceBefore=2, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet", fontName=fonts.regular, fontSize=9.0, leading=12.2, textColor=BODY, leftIndent=12, firstLineIndent=-8, spaceAfter=2.5,
        ),
        "th_left": ParagraphStyle("THLeft", fontName=fonts.bold, fontSize=8.4, leading=10.8, textColor=INK, alignment=TA_LEFT),
        "th_right": ParagraphStyle("THRight", fontName=fonts.bold, fontSize=8.4, leading=10.8, textColor=INK, alignment=TA_RIGHT),
        "th_center": ParagraphStyle("THCenter", fontName=fonts.bold, fontSize=8.4, leading=10.8, textColor=INK, alignment=TA_CENTER),
        "td_left": ParagraphStyle("TDLeft", fontName=fonts.regular, fontSize=8.3, leading=10.6, textColor=BODY, alignment=TA_LEFT),
        "td_right": ParagraphStyle("TDRight", fontName=fonts.regular, fontSize=8.3, leading=10.6, textColor=BODY, alignment=TA_RIGHT),
        "td_center": ParagraphStyle("TDCenter", fontName=fonts.regular, fontSize=8.3, leading=10.6, textColor=BODY, alignment=TA_CENTER),
    }


def _make_table(block: dict, content_width: float, fonts: FontSet, styles: dict[str, ParagraphStyle]) -> Table:
    headers, aligns, rows = block["headers"], block["alignments"], block["rows"]
    align_key = {"LEFT": "left", "RIGHT": "right", "CENTER": "center"}

    data = [
        [Paragraph(inline_markup(h, fonts), styles[f"th_{align_key[a]}"]) for h, a in zip(headers, aligns)]
    ]
    highlight_rows: list[int] = []
    for idx, row in enumerate(rows, start=1):
        if any(cell.startswith("**") and cell.endswith("**") for cell in row):
            highlight_rows.append(idx)
        data.append([Paragraph(inline_markup(c, fonts), styles[f"td_{align_key[a]}"]) for c, a in zip(row, aligns)])

    # Compute column widths proportional to visible character length with a reasonable floor.
    weights = []
    for col in range(len(headers)):
        lengths = [len(_strip_markup(headers[col]))] + [len(_strip_markup(r[col])) for r in rows]
        weights.append(max(max(lengths) + 2, 7))
    total_weight = sum(weights)
    col_widths = [content_width * w / total_weight for w in weights]

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("LINEABOVE", (0, 0), (-1, 0), 0.75, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.75, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.4, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.75, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r_idx in highlight_rows:
        style_cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), ROW_HIGHLIGHT))

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle(style_cmds))
    return table


def _make_figure_flow(
    block: dict, base_dir: Path, content_width: float, fonts: FontSet, styles: dict[str, ParagraphStyle],
) -> list:
    img_path = (base_dir / block["src"]).resolve()
    if not img_path.is_file():
        raise FileNotFoundError(f"referenced figure not found: {img_path}")
    px_w, px_h = ImageReader(str(img_path)).getSize()
    width = content_width
    height = width * (px_h / px_w)
    max_height = 175.0
    if height > max_height:
        scale = max_height / height
        width *= scale
        height *= scale
    flow: list = [Spacer(1, 3), Image(str(img_path), width=width, height=height)]
    if block["caption"]:
        flow.append(Paragraph(inline_markup(block["caption"], fonts), styles["caption"]))
    else:
        flow.append(Spacer(1, 5))
    return flow


def _make_numbered_canvas(header: str, footer: str, fonts: FontSet, page_counter: list[int]):
    class NumberedCanvas(canvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict] = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_page_states)
            page_counter.append(total)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_chrome(total)
                super().showPage()
            super().save()

        def _draw_chrome(self, total: int):
            width, height = self._pagesize
            margin_x = 44
            self.saveState()
            # Header
            self.setFont(fonts.bold, 7.6)
            self.setFillColor(BLUE)
            self.drawString(margin_x, height - 30, header)
            self.setStrokeColor(RULE)
            self.setLineWidth(0.5)
            self.line(margin_x, height - 35, width - margin_x, height - 35)
            # Footer
            self.line(margin_x, 34, width - margin_x, 34)
            self.setFont(fonts.regular, 7.6)
            self.setFillColor(MUTED)
            self.drawString(margin_x, 22, footer)
            self.drawRightString(width - margin_x, 22, f"{self._pageNumber} / {total}")
            self.restoreState()

    return NumberedCanvas


def _build_story(
    blocks: list[dict], base_dir: Path, content_width: float, fonts: FontSet, styles: dict[str, ParagraphStyle],
) -> list:
    story: list = []
    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        btype = block["type"]
        # ReportLab's keepWithNext does not bind a heading to a following KeepTogether; group them explicitly.
        if btype in ("h2", "h3") and idx + 1 < len(blocks) and blocks[idx + 1]["type"] == "figure":
            heading = Paragraph(inline_markup(block["text"], fonts), styles[btype])
            fig_flow = _make_figure_flow(blocks[idx + 1], base_dir, content_width, fonts, styles)
            story.append(KeepTogether([heading, *fig_flow]))
            idx += 2
            continue
        if btype == "pagebreak":
            story.append(PageBreak())
        elif btype == "h1":
            story.append(Paragraph(inline_markup(block["text"], fonts), styles["h1"]))
        elif btype == "subtitle":
            story.append(Paragraph(inline_markup(block["text"], fonts), styles["subtitle"]))
        elif btype == "h2":
            story.append(Paragraph(inline_markup(block["text"], fonts), styles["h2"]))
        elif btype == "h3":
            story.append(Paragraph(inline_markup(block["text"], fonts), styles["h3"]))
        elif btype == "paragraph":
            story.append(Paragraph(inline_markup(block["text"], fonts), styles["body"]))
        elif btype == "table":
            story.append(Spacer(1, 2))
            story.append(_make_table(block, content_width, fonts, styles))
            story.append(Spacer(1, 5))
        elif btype == "figure":
            story.append(KeepTogether(_make_figure_flow(block, base_dir, content_width, fonts, styles)))
        elif btype == "bullets":
            for item in block["items"]:
                story.append(Paragraph(f"• {inline_markup(item, fonts)}", styles["bullet"]))
            story.append(Spacer(1, 2))
        elif btype == "numbered":
            for prefix, item in block["items"]:
                story.append(Paragraph(f"{html.escape(prefix)} {inline_markup(item, fonts)}", styles["bullet"]))
            story.append(Spacer(1, 2))
        idx += 1
    return story


def build_pdf(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    header: str = DEFAULT_HEADER,
    footer: str = DEFAULT_FOOTER,
    author: str = DEFAULT_AUTHOR,
    subject: str = DEFAULT_SUBJECT,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> dict:
    """Compile Markdown + referenced images into an A4 PDF and enforce max_pages."""
    if not input_path.is_file():
        raise FileNotFoundError(f"report markdown not found: {input_path}")
    text = input_path.read_text(encoding="utf-8")
    blocks = parse_markdown(text)
    if not blocks:
        raise ValueError(f"{input_path}: report markdown is empty")

    title = next((b["text"] for b in blocks if b["type"] == "h1"), input_path.stem)
    fonts = register_fonts()
    styles = _build_styles(fonts)

    margin_x = 44
    content_width = A4[0] - 2 * margin_x
    story = _build_story(blocks, input_path.parent, content_width, fonts, styles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    page_counter: list[int] = []
    try:
        doc = SimpleDocTemplate(
            str(tmp_path),
            pagesize=A4,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=46,
            bottomMargin=42,
            title=title,
            author=author,
            subject=subject,
            creator="src/build_report.py",
        )
        doc.build(story, canvasmaker=_make_numbered_canvas(header, footer, fonts, page_counter))
        pages = page_counter[-1] if page_counter else 0
        if max_pages > 0 and pages > max_pages:
            raise ValueError(f"rendered PDF has {pages} pages, exceeding --max-pages={max_pages}")
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"input": input_path, "output": output_path, "pages": pages, "fonts": fonts.regular}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source Markdown file (default: report/report.md)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="destination PDF file (default: report/report.pdf)")
    parser.add_argument("--header", default=DEFAULT_HEADER, help="running header text")
    parser.add_argument("--footer", default=DEFAULT_FOOTER, help="running footer text")
    parser.add_argument("--author", default=DEFAULT_AUTHOR, help="PDF Author metadata")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="fail if PDF exceeds this many pages (0 disables)")
    args = parser.parse_args()
    try:
        result = build_pdf(
            input_path=args.input,
            output_path=args.output,
            header=args.header,
            footer=args.footer,
            author=args.author,
            max_pages=args.max_pages,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Wrote {result['output']} ({result['pages']} pages, font={result['fonts']}).")


if __name__ == "__main__":
    main()
