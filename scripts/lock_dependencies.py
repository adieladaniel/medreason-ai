"""Resolve requirements-api.txt and requirements-test.txt into fully pinned, hashed lock files for the container.

Why: the two .txt files pin only the packages we import. Everything those packages depend on (starlette, pydantic, scipy, sympy,
urllib3 and about 60 more) was left to whatever pip found on build day, so the same commit could build a different image next month,
and a package replaced on the index would be installed without anyone noticing. A lock file names every package at one version with
the SHA-256 of its files, and `pip install --require-hashes` refuses anything else. The same tests then run against the same bytes
on every machine and in CI.

How: `uv pip compile` (tool only, not a project dependency: `pip install uv`) resolves for the platform the container really is,
Linux x86-64 with Python 3.11, so it can be run from this Windows laptop. The CPU-only PyTorch index is searched as well as PyPI:
that is where torch 2.11.0+cpu comes from, and it is why no CUDA package (about 3 GB) is in the lock. `unsafe-best-match` lets
uv take each package's version from whichever index has it; the hashes in the lock are what make that safe.

Run from the repo root after changing a pin in requirements-api.txt or requirements-test.txt:  python scripts/lock_dependencies.py
Then rebuild the image (`docker build --target test .`). The lock is for the container and CI, not for the Windows laptop, which
keeps using requirements.txt.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMON = [
    "--python-version", "3.11", "--python-platform", "x86_64-unknown-linux-gnu", "--generate-hashes",
    "--index-url", "https://pypi.org/simple", "--extra-index-url", "https://download.pytorch.org/whl/cpu",
    "--index-strategy", "unsafe-best-match", "--quiet",
]


def uv() -> list[str]:
    exe = shutil.which("uv")
    if exe:
        return [exe]
    probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True)
    if probe.returncode == 0:
        return [sys.executable, "-m", "uv"]
    sys.exit("uv is not installed: pip install uv   (it is a tool for this script, not a dependency of the project)")


def main() -> None:
    base = uv() + ["pip", "compile"]
    # the service first, then the test tools constrained to the service's versions, so a shared package has one version
    subprocess.run(base + ["requirements-api.txt", "-o", "requirements-api.lock"] + COMMON, cwd=ROOT, check=True)
    constraints = ROOT / "requirements-api.constraints.tmp"
    names = [ln.split(" ")[0] for ln in (ROOT / "requirements-api.lock").read_text(encoding="utf-8").splitlines()
             if ln and not ln[0].isspace() and not ln.startswith("#") and "==" in ln]
    constraints.write_text("\n".join(names) + "\n", encoding="utf-8")                  # name==version only, no hashes
    try:
        subprocess.run(base + ["requirements-test.txt", "-c", constraints.name, "-o", "requirements-test.lock"] + COMMON, cwd=ROOT, check=True)
    finally:
        constraints.unlink(missing_ok=True)
    for lock in ("requirements-api.lock", "requirements-test.lock"):
        text = (ROOT / lock).read_text(encoding="utf-8")
        print(f"{lock}: {sum(1 for ln in text.splitlines() if ln and not ln[0].isspace() and '==' in ln and not ln.startswith('#'))} packages")


if __name__ == "__main__":
    main()
