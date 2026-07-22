#!/usr/bin/env python3
"""Build docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf from the polished markdown.

PI venue polish helper (ReportLab). No invented numbers — renders prose + embeds
existing figures under docs/manuscript/figures/. Re-run after any manuscript edit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs/manuscript/CAD_CBA_v1_MANUSCRIPT.md"
OUT = ROOT / "docs/manuscript/CAD_CBA_v1_MANUSCRIPT.pdf"
FIG = ROOT / "docs/manuscript/figures"


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def inline_md(s: str) -> str:
    """Minimal markdown → ReportLab XML for inline spans.

    Protect code spans first so paths like results/** never break bold parsing.
    """
    # Extract fenced-style inline code before escaping/bold
    code_slots: list[str] = []

    def _stash_code(m: re.Match) -> str:
        code_slots.append(m.group(1))
        return f"\x00CODE{len(code_slots) - 1}\x00"

    s = re.sub(r"`([^`]+)`", _stash_code, s)
    s = esc(s)
    # bold then italic (non-greedy, no nested *)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    # restore code as Courier (already escaped content when stashed? re-escape)
    for idx, code in enumerate(code_slots):
        s = s.replace(
            f"\x00CODE{idx}\x00",
            f'<font face="Courier" size="8">{esc(code)}</font>',
        )
    # leftover unpaired ** markers (paths etc.)
    s = s.replace("**", "")
    return s


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for ln in lines:
        ln = ln.strip()
        if not ln.startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells and re.match(r"^:?-+:?$", cells[0].replace(" ", "")):
            # separator row
            if all(re.match(r"^:?-+:?$", c.replace(" ", "")) for c in cells):
                continue
        rows.append(cells)
    return rows


def make_styles():
    ss = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "MSTitle",
            parent=ss["Title"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "meta": ParagraphStyle(
            "MSMeta",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#333333"),
            spaceAfter=4,
        ),
        "h1": ParagraphStyle(
            "MSH1",
            parent=ss["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=14,
            spaceAfter=6,
            textColor=colors.HexColor("#111111"),
        ),
        "h2": ParagraphStyle(
            "MSH2",
            parent=ss["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            spaceBefore=10,
            spaceAfter=4,
            textColor=colors.HexColor("#222222"),
        ),
        "h3": ParagraphStyle(
            "MSH3",
            parent=ss["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            leading=12,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "MSBody",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "caption": ParagraphStyle(
            "MSCaption",
            parent=ss["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            spaceBefore=2,
            spaceAfter=8,
            textColor=colors.HexColor("#222222"),
        ),
        "quote": ParagraphStyle(
            "MSQuote",
            parent=ss["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            leftIndent=10,
            rightIndent=10,
            textColor=colors.HexColor("#333333"),
            spaceAfter=6,
            spaceBefore=4,
        ),
        "code": ParagraphStyle(
            "MSCode",
            parent=ss["Code"],
            fontName="Courier",
            fontSize=7,
            leading=9,
            leftIndent=4,
            spaceAfter=6,
            spaceBefore=4,
        ),
        "bullet": ParagraphStyle(
            "MSBullet",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=11,
            leftIndent=12,
            spaceAfter=2,
        ),
        "footer": ParagraphStyle(
            "MSFooter",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=7,
            alignment=TA_CENTER,
            textColor=colors.grey,
        ),
        "cell": ParagraphStyle(
            "MSCell",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
        ),
        "cell_h": ParagraphStyle(
            "MSCellH",
            parent=ss["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=9,
        ),
    }
    return styles


def table_flowable(rows: list[list[str]], styles) -> Table:
    if not rows:
        return Spacer(1, 1)
    ncol = max(len(r) for r in rows)
    data = []
    for i, r in enumerate(rows):
        padded = r + [""] * (ncol - len(r))
        st = styles["cell_h"] if i == 0 else styles["cell"]
        data.append([Paragraph(inline_md(c), st) for c in padded])
    avail = 170 * mm
    col_w = avail / ncol
    t = Table(data, colWidths=[col_w] * ncol, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f9fb")]),
            ]
        )
    )
    return t


def resolve_fig(path_str: str) -> Path | None:
    p = path_str.strip()
    candidates = [
        FIG / Path(p).name,
        ROOT / "docs/manuscript" / p,
        ROOT / p,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def build_story(md_text: str, styles):
    story = []
    lines = md_text.splitlines()
    i = 0
    title_done = False
    in_code = False
    code_buf: list[str] = []
    para_buf: list[str] = []

    def flush_para():
        nonlocal para_buf
        if not para_buf:
            return
        text = " ".join(para_buf).strip()
        para_buf = []
        if not text:
            return
        if text.startswith(">"):
            story.append(Paragraph(inline_md(text.lstrip("> ").strip()), styles["quote"]))
        else:
            story.append(Paragraph(inline_md(text), styles["body"]))

    while i < len(lines):
        ln = lines[i]
        raw = ln.rstrip("\n")

        # fenced code
        if raw.strip().startswith("```"):
            if not in_code:
                flush_para()
                in_code = True
                code_buf = []
            else:
                in_code = False
                story.append(Preformatted("\n".join(code_buf), styles["code"]))
                code_buf = []
            i += 1
            continue
        if in_code:
            code_buf.append(raw)
            i += 1
            continue

        # blank
        if not raw.strip():
            flush_para()
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}$", raw.strip()):
            flush_para()
            story.append(Spacer(1, 4))
            i += 1
            continue

        # image
        m_img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", raw.strip())
        if m_img:
            flush_para()
            alt, path = m_img.group(1), m_img.group(2)
            fp = resolve_fig(path)
            if fp is not None:
                img = Image(str(fp))
                # fit width
                max_w = 160 * mm
                max_h = 90 * mm
                iw, ih = img.imageWidth, img.imageHeight
                scale = min(max_w / iw, max_h / ih, 1.0)
                img.drawWidth = iw * scale
                img.drawHeight = ih * scale
                img.hAlign = "CENTER"
                story.append(Spacer(1, 4))
                story.append(img)
            else:
                story.append(Paragraph(f"[missing figure: {esc(path)}]", styles["caption"]))
            i += 1
            # optional following caption line starting with **Figure
            if i < len(lines) and lines[i].strip().startswith("**Figure"):
                story.append(Paragraph(inline_md(lines[i].strip()), styles["caption"]))
                i += 1
            continue

        # headings
        if raw.startswith("# "):
            flush_para()
            text = raw[2:].strip()
            if not title_done:
                story.append(Paragraph(inline_md(text), styles["title"]))
                title_done = True
            else:
                story.append(Paragraph(inline_md(text), styles["h1"]))
            i += 1
            continue
        if raw.startswith("## "):
            flush_para()
            story.append(Paragraph(inline_md(raw[3:].strip()), styles["h1"]))
            i += 1
            continue
        if raw.startswith("### "):
            flush_para()
            story.append(Paragraph(inline_md(raw[4:].strip()), styles["h2"]))
            i += 1
            continue
        if raw.startswith("#### "):
            flush_para()
            story.append(Paragraph(inline_md(raw[5:].strip()), styles["h3"]))
            i += 1
            continue

        # table block
        if raw.strip().startswith("|"):
            flush_para()
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            rows = parse_table(tbl_lines)
            if rows:
                story.append(Spacer(1, 3))
                story.append(table_flowable(rows, styles))
                story.append(Spacer(1, 6))
            continue

        # bullets
        if re.match(r"^[-*]\s+", raw.strip()) or re.match(r"^\d+\.\s+", raw.strip()):
            flush_para()
            items = []
            while i < len(lines) and (
                re.match(r"^[-*]\s+", lines[i].strip())
                or re.match(r"^\d+\.\s+", lines[i].strip())
            ):
                t = re.sub(r"^([-*]|\d+\.)\s+", "", lines[i].strip())
                items.append(ListItem(Paragraph(inline_md(t), styles["bullet"]), leftIndent=8))
                i += 1
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=12))
            story.append(Spacer(1, 3))
            continue

        # meta bold lines under title (document type etc.)
        if raw.strip().startswith("**") and not title_done:
            story.append(Paragraph(inline_md(raw.strip()), styles["meta"]))
            i += 1
            continue

        # normal paragraph accumulation
        para_buf.append(raw.strip())
        i += 1

    flush_para()
    return story


def add_page_number(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(
        A4[0] / 2,
        12 * mm,
        f"CAD-CBA-v1 local-complete draft · PI venue polish · p. {page}",
    )
    canvas.restoreState()


def main() -> int:
    if not MD.is_file():
        print(f"ERROR: missing {MD}", file=sys.stderr)
        return 1
    md_text = MD.read_text(encoding="utf-8")
    styles = make_styles()
    story = build_story(md_text, styles)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="CAD-CBA-v1 Manuscript (local-complete, PI venue polish)",
        author="COLIDE project (PI to finalise)",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
