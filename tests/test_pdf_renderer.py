from pathlib import Path
import types


from logic import pdf_renderer


def test_normalize_output_path_creates_dir_and_pdf(tmp_path):
    raw = tmp_path / "subdir" / "file"
    normalized = pdf_renderer.normalize_output_path(str(raw))
    assert normalized.endswith("subdir/file.pdf")
    assert Path(normalized).is_absolute()
    assert Path(normalized).parent.exists()
    assert Path(normalized).suffix == ".pdf"


def test_render_html_to_pdf_writes_file(tmp_path, monkeypatch):
    def fake_config(**kwargs):
        return types.SimpleNamespace()

    written = {}

    def fake_from_string(html, path, configuration=None, options=None):
        Path(path).write_bytes(b"PDF")
        written["path"] = path

    monkeypatch.setattr(pdf_renderer.pdfkit, "configuration", fake_config)
    monkeypatch.setattr(pdf_renderer.pdfkit, "from_string", fake_from_string)

    output = tmp_path / "out"
    pdf_renderer.render_html_to_pdf("<p>hi</p>", str(output))
    assert written["path"].endswith(".pdf")
    assert Path(written["path"]).exists()
