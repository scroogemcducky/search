import pytest

from bookqa.config import Project, load_config


def test_defaults_when_missing(tmp_path):
    config = load_config(tmp_path / "bookqa.toml")
    assert config.embedding.model == "Qwen/Qwen3-Embedding-8B"


def test_nested_overrides(tmp_path):
    (tmp_path / "bookqa.toml").write_text(
        '[embedding]\nbackend = "hashing"\ndimensions = 64\n[claims]\nevidence_types = ["a", "b"]\n'
    )
    project = Project(tmp_path)
    assert project.config.embedding.backend == "hashing"
    assert project.config.embedding.dimensions == 64
    assert project.config.claims.evidence_types == ["a", "b"]
    assert project.config.llm.model == "qwen3:14b"


def test_unknown_key_is_rejected(tmp_path):
    (tmp_path / "bookqa.toml").write_text("[embedding]\nmodle = 'x'\n")
    with pytest.raises(ValueError, match=r"embedding\.modle"):
        load_config(tmp_path / "bookqa.toml")
