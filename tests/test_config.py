"""Settings that differ between this laptop and docker compose: the Ollama address and model, and the
Qdrant mode. The Qdrant client and the Ollama call are replaced, so no service or model is needed."""
import pytest

from app.config import get_settings
from app.rag import index, writer
from tests.conftest import make_agent


@pytest.fixture
def env(monkeypatch):
    """Set MEDREASON_* variables and drop the cached settings so the next read sees them."""
    def set_(**kv):
        for k, v in kv.items():
            monkeypatch.setenv("MEDREASON_" + k.upper(), str(v))
        get_settings.cache_clear()
    get_settings.cache_clear()
    yield set_
    get_settings.cache_clear()


# ------------------------------------------------------------------ Ollama
def test_ollama_defaults_match_the_laptop_setup(env):
    # The address is the IPv4 number, not "localhost": on Windows, Python tries ::1 first and waits 2 s for it to be refused.
    assert writer.chat_url() == "http://127.0.0.1:11434/api/chat"
    assert get_settings().ollama_model == "llama3.2:latest"


def test_ollama_url_follows_the_setting_and_ignores_a_trailing_slash(env):
    env(ollama_url="http://host.docker.internal:11434/")
    assert writer.chat_url() == "http://host.docker.internal:11434/api/chat"


def test_the_writer_uses_the_configured_model_unless_one_is_passed(env, monkeypatch):
    used = []
    monkeypatch.setattr(writer, "_ask", lambda p, topic, model, timeout: used.append(model) or "Low albumin is associated with mortality.")
    ev = {"passages": [{"n": 1, "chunk_id": 1, "text": "t", "heading": "h", "title": "T", "citation": "C", "url": "u",
                        "drivers": ["d"], "queries": ["q"], "terms": None, "rerank_score": 9.0}], "unanswered": []}
    plan = {"queries": [{"text": "q", "modality": "blood", "driver": "d", "effect": 0.1, "terms": None}]}
    agent = make_agent()

    env(ollama_model="qwen2.5:3b")
    assert writer.write_explanation(agent, plan, ev)["model"] == "qwen2.5:3b"
    assert writer.write_explanation(agent, plan, ev, model="llama3.2:latest")["model"] == "llama3.2:latest"
    assert used == ["qwen2.5:3b", "llama3.2:latest"]


# ------------------------------------------------------------------ Qdrant
class FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        pass


@pytest.fixture
def fake_qdrant(monkeypatch):
    monkeypatch.setattr(index, "QdrantClient", FakeClient)
    monkeypatch.setattr(index, "_clients", {})


def test_embedded_mode_is_the_default_and_uses_one_folder_per_model(env, fake_qdrant, tmp_path):
    env(qdrant_path=tmp_path)
    c = index._client("NeuML/pubmedbert-base-embeddings")
    assert c.kwargs == {"path": str(tmp_path / "NeuML_pubmedbert_base_embeddings")}
    assert (tmp_path / "NeuML_pubmedbert_base_embeddings").is_dir()
    assert index.collection("NeuML/pubmedbert-base-embeddings") == "evidence"      # unchanged, so the saved index still works


def test_a_qdrant_url_switches_to_server_mode(env, fake_qdrant):
    env(qdrant_url="http://qdrant:6333")
    assert index._client("NeuML/pubmedbert-base-embeddings").kwargs == {"url": "http://qdrant:6333"}


def test_server_mode_keeps_models_in_separate_collections(env):
    env(qdrant_url="http://qdrant:6333")
    a, b = index.collection("NeuML/pubmedbert-base-embeddings"), index.collection("BAAI/bge-small-en-v1.5")
    assert a != b and a.startswith("evidence_") and b.startswith("evidence_")


def test_the_client_is_created_once_per_model(env, fake_qdrant):
    env(qdrant_url="http://qdrant:6333")
    assert index._client("m") is index._client("m")


def test_the_tests_never_use_the_projects_own_database_or_upload_folder(env):
    """conftest points the default database and upload folder at a throwaway folder, so a test that forgets to
    override them cannot create data/app.db or fill data/uploads."""
    from app.config import ROOT
    cfg = get_settings()
    assert not cfg.database_url.startswith(f"sqlite:///{(ROOT / 'data').as_posix()}")
    assert ROOT / "data" not in cfg.upload_dir.parents and cfg.upload_dir != ROOT / "data" / "uploads"
