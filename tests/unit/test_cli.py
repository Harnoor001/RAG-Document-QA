from app.ingest.__main__ import main


def test_cli_prints_card_line(monkeypatch, tmp_path, handbook, capsys):
    monkeypatch.setenv("EMBED_MODEL", "hash")
    assert main([str(handbook), "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out.strip()
    name, pages, chunks, doc = out.split(" · ")
    assert name == "handbook.pdf" and pages == "42 pages"
    assert chunks.endswith("chunks indexed") and len(doc) == 12


def test_cli_reports_bad_input_without_traceback(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EMBED_MODEL", "hash")
    bad = tmp_path / "x.pdf"
    bad.write_text("hello")
    assert main([str(bad), "--data-dir", str(tmp_path)]) == 2
    assert main([str(tmp_path / "missing.pdf"), "--data-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.count("error:") == 2 and "Traceback" not in err
