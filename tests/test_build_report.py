import pytest

import build_report
from common import ROOT


def test_parse_markdown_extracts_headings_subtitle_tables_figures_and_lists():
    md = "\n".join([
        "# Report Title",
        "",
        "**Subtitle line**",
        "",
        "## Section One",
        "",
        "Body with `code` and **bold** and *italic*.",
        "",
        "| Name | Score |",
        "| --- | ---: |",
        "| **Model A** | 0.800 |",
        "",
        "![Alt text](fig.png)",
        "",
        "*Figure 1. Caption text.*",
        "",
        "- First bullet",
        "- Second bullet",
        "",
        "---",
        "",
        "### Subsection",
        "",
        "1. Step one",
        "2. Step two",
    ])
    blocks = build_report.parse_markdown(md)
    assert [b["type"] for b in blocks] == [
        "h1", "subtitle", "h2", "paragraph", "table", "figure", "bullets", "pagebreak", "h3", "numbered",
    ]
    assert blocks[1]["text"] == "Subtitle line"
    assert blocks[4]["alignments"] == ["LEFT", "RIGHT"]
    assert blocks[5]["caption"] == "Figure 1. Caption text."
    assert blocks[6]["items"] == ["First bullet", "Second bullet"]
    assert blocks[9]["items"] == [("1.", "Step one"), ("2.", "Step two")]


def test_inline_markup_escapes_xml_and_preserves_code_spans():
    fonts = build_report.FontSet("Lato-Regular", "Lato-Bold", "Lato-Italic", "Lato-BoldItalic", "DejaVuSansMono")
    out = build_report.inline_markup("Use `a < b & *c*` with **bold** and [docs](https://example.com/?a=1&b=2).", fonts)
    assert '<font name="DejaVuSansMono" size="8.2">a &lt; b &amp; *c*</font>' in out
    assert "<b>bold</b>" in out
    assert '<link href="https://example.com/?a=1&amp;b=2" color="#23486b">docs</link>' in out


def test_build_pdf_compiles_the_real_report_within_two_pages(tmp_path):
    output = tmp_path / "report.pdf"
    result = build_report.build_pdf(
        input_path=ROOT / "report" / "report.md",
        output_path=output,
    )
    assert result["pages"] == 2
    data = output.read_bytes()
    assert data.startswith(b"%PDF-") and len(data) > 10_000


def test_build_pdf_rejects_overflow_without_overwriting_existing_pdf(tmp_path):
    md = tmp_path / "long.md"
    md.write_text("# Title\n\n" + "\n\n".join(f"Paragraph {i}. " + ("word " * 80) for i in range(20)), encoding="utf-8")
    output = tmp_path / "report.pdf"
    output.write_bytes(b"original-pdf")
    with pytest.raises(ValueError, match="exceeding --max-pages=1"):
        build_report.build_pdf(input_path=md, output_path=output, max_pages=1)
    assert output.read_bytes() == b"original-pdf"
    assert list(tmp_path.glob("*.tmp")) == []


def test_heading_followed_by_figure_is_grouped_in_one_keep_together():
    from reportlab.platypus import KeepTogether

    blocks = build_report.parse_markdown("## Failure analyses\n\n![Matrix](figures/question_model_matrix.png)\n\n*Caption.*\n")
    fonts = build_report.register_fonts()
    styles = build_report._build_styles(fonts)
    story = build_report._build_story(blocks, ROOT / "report", 500.0, fonts, styles)
    assert len(story) == 1 and isinstance(story[0], KeepTogether)


def test_build_pdf_fails_loudly_on_missing_figure(tmp_path):
    md = tmp_path / "bad.md"
    md.write_text("# Title\n\n![Missing](nope.png)\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="referenced figure not found"):
        build_report.build_pdf(input_path=md, output_path=tmp_path / "out.pdf")
