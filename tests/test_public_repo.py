"""The repository is public. Three files are private working notes (instructions for the assistant, personal planning, machine details): they stay on
disk, are git-ignored and docker-ignored, and nothing that is published may depend on them or point to them."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ["CLAUDE.md", "ROADMAP.md", "docs/ENVIRONMENT_AND_HANDOFF.md"]
SCANNED_SUFFIXES = {".py", ".md", ".js", ".html", ".css", ".yml", ".yaml", ".toml", ".ini", ".txt", ".sh"}
SCANNED_DIRS = ["app", "explain", "scripts", "tests", "docs", "experiments", "migrations", ".github", "kaggle_upload"]
SKIP_NAMES = {"test_public_repo.py", "test_dashboard_data.py"}      # these two name the files on purpose


def lines(name):
    return {ln.strip() for ln in (ROOT / name).read_text(encoding="utf-8").splitlines()}


def test_the_private_notes_are_ignored_by_git_and_by_docker():
    git, docker = lines(".gitignore"), lines(".dockerignore")
    for f in PRIVATE:
        assert "/" + f in git, f"{f} is not in .gitignore"
        assert f in docker or "/" + f in docker, f"{f} is not in .dockerignore"


def test_nothing_published_refers_to_the_private_notes():
    hits = []
    candidates = [ROOT / "README.md", ROOT / "Dockerfile", ROOT / "docker-compose.yml"]
    for d in SCANNED_DIRS:
        base = ROOT / d
        if base.exists():
            candidates += [p for p in base.rglob("*") if p.is_file() and p.suffix in SCANNED_SUFFIXES]
    for p in candidates:
        rel = p.relative_to(ROOT).as_posix()
        if p.name in SKIP_NAMES or rel in PRIVATE or "_chunks" in rel:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name in ("CLAUDE.md", "ROADMAP.md", "ENVIRONMENT_AND_HANDOFF"):
            if name in text:
                hits.append(f"{rel} mentions {name}")
    assert not hits, hits


def test_the_committed_dashboard_data_does_not_name_the_private_notes():
    assert "ROADMAP" not in (ROOT / "app" / "data" / "dashboard.json").read_text(encoding="utf-8")


def test_the_readme_links_only_to_files_that_exist():
    import re

    import pytest
    if not (ROOT / "README.md").exists() or not (ROOT / "docs").exists():
        pytest.skip("the image build context leaves out README.md and docs/")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    missing = []
    for target in re.findall(r"\]\((?!https?://|#)([^)\s]+)\)", text):
        if not (ROOT / target.split("#")[0]).exists():
            missing.append(target)
    assert not missing, missing
