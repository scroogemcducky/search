"""End-to-end runs of the command-line tool with --offline (no models)."""

import json
import shutil
from pathlib import Path

import pytest

from bookqa.cli import main

DEMO = Path(__file__).parent.parent / "examples" / "demo"


@pytest.fixture
def project(tmp_path) -> Path:
    shutil.copytree(DEMO, tmp_path / "demo")
    return tmp_path / "demo"


def run(project: Path, *args: str) -> None:
    main([args[0], "--project", str(project), *args[1:]])


def test_init_creates_a_project(tmp_path, capsys):
    main(["init", str(tmp_path / "new"), "--prompts"])
    assert (tmp_path / "new" / "bookqa.toml").exists()
    assert (tmp_path / "new" / "prompts" / "answer.md").exists()
    assert "bookqa build" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["init", str(tmp_path / "new")])


def test_init_template_is_valid_config(tmp_path):
    from bookqa.config import Project

    main(["init", str(tmp_path / "p")])
    config = Project(tmp_path / "p").config
    assert config.embedding.model == "Qwen/Qwen3-Embedding-4B"


def test_full_offline_run(project, capsys):
    run(project, "build", "--offline")
    run(project, "status")
    run(project, "claims", "--evidence", "study", "--limit", "2")
    run(project, "ask", "--offline", "How long should a nap be?")
    out = capsys.readouterr().out
    assert "Read 3 books" in out and "claims from" in out
    assert "(verified)" in out
    answer = project / "output" / "how-long-should-a-nap-be" / "answer.md"
    assert "## Sources" in answer.read_text()

    run(project, "ask", "--offline", "--file", str(project / "questions.jsonl"))
    # The first question is also in questions.jsonl, so its folder is reused.
    assert len(list((project / "output").iterdir())) == 4

    run(project, "eval", "--offline")
    assert (project / ".bookqa" / "eval" / "report.md").exists()
    rows = json.loads((project / ".bookqa" / "eval" / "per_question.json").read_text())
    assert {r["method"] for r in rows} >= {"keyword only (BM25)"}


def test_stale_chunks_are_detected(project):
    run(project, "build", "--offline")
    (project / "bookqa.toml").write_text("[chunking]\ntarget_words = 40\n")
    run(project, "ingest")
    with pytest.raises(SystemExit, match="changed since"):
        run(project, "ask", "--offline", "anything")


def test_ask_before_build_explains_what_to_do(project):
    with pytest.raises(SystemExit, match="bookqa ingest"):
        run(project, "ask", "--offline", "anything")
