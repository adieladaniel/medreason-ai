"""
Phase 7 step 6: the files that build and run the containers cannot be run by the normal test suite, so these check the
properties that would otherwise rot unnoticed. The image itself is exercised by `docker build --target test .`, which runs
this suite again inside a clean build, and by starting the stack (docs/DOCKER.md).
"""
import ast
import re
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from app import dashboard_data, prefetch
from app.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parent.parent


def pins(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?==([^\s#]+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(3)
    return out


FULL, RUNTIME = pins(ROOT / "requirements.txt"), pins(ROOT / "requirements-api.txt")
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


# ------------------------------------------------------------------ requirements
def test_every_runtime_pin_is_the_same_version_as_in_the_full_requirements():
    assert RUNTIME, "requirements-api.txt has no pins"
    wrong = {k: (v, FULL.get(k)) for k, v in RUNTIME.items() if FULL.get(k) != v}
    assert not wrong, f"requirements-api.txt differs from requirements.txt: {wrong}"


def test_the_dockerfile_installs_the_same_torch_and_test_tools_as_the_requirements():
    for name in ("torch", "torchvision", "pytest", "httpx"):
        assert f"{name}=={FULL[name]}" in DOCKERFILE, f"Dockerfile does not install {name}=={FULL[name]}"
    assert "download.pytorch.org/whl/cpu" in DOCKERFILE                  # the default wheel carries CUDA libraries


IMPORT_TO_DIST = {"PIL": "pillow", "sklearn": "scikit-learn", "sse_starlette": "sse-starlette", "qdrant_client": "qdrant-client",
                  "sentence_transformers": "sentence-transformers", "pydantic_settings": "pydantic-settings",
                  "sqlalchemy": "sqlalchemy", "xrv": "torchxrayvision", "torchxrayvision": "torchxrayvision"}
INSTALLED_AS_DEPENDENCY = {"pydantic", "starlette", "torch", "torchvision"}      # torch and torchvision: the Dockerfile's first RUN
LOCAL = {"app", "explain", "experiments", "scripts", "tests"}


def imported_modules(folder: str) -> set[str]:
    found = set()
    for path in (ROOT / folder).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_everything_the_service_imports_is_installed_by_the_runtime_requirements():
    modules = imported_modules("app") | imported_modules("explain")
    third_party = {m for m in modules if m not in sys.stdlib_module_names and m not in LOCAL and m != "__future__"}
    missing = sorted(m for m in third_party
                     if m not in INSTALLED_AS_DEPENDENCY and IMPORT_TO_DIST.get(m, m).lower().replace("_", "-") not in RUNTIME)
    assert not missing, f"imported by app/ or explain/ but not in requirements-api.txt: {missing}"


# ------------------------------------------------------------------ compose
def test_every_published_port_is_bound_to_the_loopback_address():
    ports = [p for svc in COMPOSE["services"].values() for p in svc.get("ports", [])]
    assert len(ports) == 3
    assert all(str(p).startswith("127.0.0.1:") for p in ports), ports       # the API has no login


def test_the_database_password_must_come_from_the_environment_and_has_no_default():
    password = COMPOSE["services"]["postgres"]["environment"]["POSTGRES_PASSWORD"]
    assert password.startswith("${POSTGRES_PASSWORD:?"), password            # ":?" refuses to start when it is unset
    assert "change-me" not in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_the_trained_models_are_mounted_read_only_and_not_baked_into_the_image():
    api = COMPOSE["services"]["api"]
    assert "./experiments:/app/experiments:ro" in api["volumes"]
    runtime = DOCKERFILE.split("AS runtime")[1]
    assert "COPY" not in runtime and "experiments" not in runtime
    base = DOCKERFILE.split("AS test")[0]
    assert "COPY experiments" not in base and "COPY data" not in DOCKERFILE


def test_the_demo_cases_are_mounted_read_only_and_kept_out_of_git_and_the_image():
    assert "./demo/cases:/app/demo/cases:ro" in COMPOSE["services"]["api"]["volumes"]
    assert "demo/cases/" in lines(".gitignore") and "/demo" in lines(".dockerignore")        # MIMIC images and clinical text


def test_the_api_runs_offline_against_the_services_in_the_stack_and_the_host_ollama():
    env = COMPOSE["services"]["api"]["environment"]
    assert env["HF_HUB_OFFLINE"] == "1" and env["MEDREASON_REBUILD_DASHBOARD"] == "false"
    assert env["MEDREASON_QDRANT_URL"] == "http://qdrant:6333"
    assert env["MEDREASON_OLLAMA_URL"] == "http://host.docker.internal:11434"
    assert env["MEDREASON_DATABASE_URL"].startswith("postgresql+psycopg://medreason:${POSTGRES_PASSWORD}@postgres:5432/")
    assert "host.docker.internal:host-gateway" in COMPOSE["services"]["api"]["extra_hosts"]


def test_the_setup_service_is_opt_in_publishes_nothing_and_mounts_only_open_access_data():
    setup = COMPOSE["services"]["setup"]
    assert setup["profiles"] == ["setup"]                                       # `docker compose up` never starts it
    assert "ports" not in setup
    assert "./data/corpus:/app/data/corpus:ro" in setup["volumes"]
    assert not any("data/processed" in v or "data/raw" in v for v in setup["volumes"])
    assert setup["image"] == COMPOSE["services"]["api"]["image"]


def test_the_api_waits_for_a_healthy_database():
    assert COMPOSE["services"]["api"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert "healthcheck" in COMPOSE["services"]["postgres"]


def test_the_container_does_not_run_as_root_and_has_a_health_check():
    runtime = DOCKERFILE.split("AS runtime")[1]
    assert re.search(r"^USER app$", runtime, re.M) and "HEALTHCHECK" in runtime
    assert "--no-access-log" in runtime                                        # the app's own request log replaces it


# ------------------------------------------------------------------ keeping secrets and data out
def lines(name: str) -> set[str]:
    return {x.strip() for x in (ROOT / name).read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}


def test_the_env_file_is_ignored_by_git_and_by_the_build_context():
    assert ".env" in lines(".gitignore") and ".env" in lines(".dockerignore")
    assert "POSTGRES_PASSWORD=change-me" in (ROOT / ".env.example").read_text(encoding="utf-8")      # a placeholder, never a real one


@pytest.mark.parametrize("pattern", ["/data", "experiments/**/*.joblib", "experiments/**/*.pth", "experiments/**/*.npy",
                                     "experiments/**/val_predictions.csv", "experiments/explain_samples"])
def test_the_build_context_leaves_out_the_clinical_data_and_what_is_derived_from_it(pattern):
    assert pattern in lines(".dockerignore")


def test_every_pattern_git_ignores_for_clinical_data_is_also_in_dockerignore():
    git, docker = lines(".gitignore"), lines(".dockerignore")
    for pattern in ("experiments/**/val_predictions.csv", "experiments/**/oof_train_predictions.csv", "experiments/**/*.npy",
                    "experiments/explain_samples/", "experiments/test_evaluation/predictions.csv", "experiments/other_pathology/studies.csv",
                    "experiments/e2e/**/cases.json"):
        assert pattern in git
        assert pattern.rstrip("/") in docker, f"{pattern} is ignored by git but not by docker"


# ------------------------------------------------------------------ code the packaging relies on
def test_prefetch_loads_each_download_through_the_function_the_app_uses():
    d = prefetch.downloads()
    assert set(d) == {"image backbone (torchxrayvision)", "text encoder (Bio_ClinicalBERT)", "passage embedder", "reranker",
                      "entailment checker"}
    assert all(callable(f) for f in d.values())


@pytest.mark.parametrize("rebuild,expected", [(True, 1), (False, 0)])
def test_the_dashboard_file_is_rebuilt_at_startup_only_when_the_setting_allows(tmp_path, monkeypatch, rebuild, expected):
    calls = []
    monkeypatch.setattr(dashboard_data, "ensure_built", lambda: calls.append(1))
    cfg = Settings(database_url=f"sqlite:///{(tmp_path / 'p.db').as_posix()}", upload_dir=tmp_path / "u", run_worker=False,
                   rebuild_dashboard=rebuild)
    with TestClient(create_app(cfg, build_dashboard=True)):
        pass
    assert len(calls) == expected
