import pytest
from unittest.mock import MagicMock, patch

from mullet.core.knowledge.manager import (
    KnowledgeContext,
    KnowledgeManager,
    knowledge_session,
    build_knowledge_environment,
)


@pytest.fixture
def patch_engine():
    client_mod = MagicMock()
    client_inst = MagicMock()
    client_inst.index_path = "/tmp/idx"
    client_inst.search_config = {"bm25_weight": 0.5, "invalid_key": "skip", "vector_weight": 0.3}
    client_inst.embedding_config = {"model": "emb-model"}
    client_mod.load_client.return_value = client_inst

    memory_mod = MagicMock()
    memory_mod.MemoryStore = MagicMock()

    workflow_mod = MagicMock()
    workflow_mod.WorkflowEngine = MagicMock()

    search_mod = MagicMock()
    search_mod.HybridSearcher = MagicMock()

    mods = {
        "client": client_mod,
        "memory": memory_mod,
        "workflow": workflow_mod,
        "search.searcher": search_mod,
    }

    def side_effect(name):
        return mods[name]

    with patch("mullet.core.knowledge.manager._get_engine_module", side_effect=side_effect):
        yield mods


def test_knowledge_context_defaults():
    ctx = KnowledgeContext()
    assert ctx.markdown == ""
    assert ctx.chunks_used == 0
    assert ctx.memories_used == 0
    assert ctx.tokens_estimated == 0


def test_searcher_params_filter_config(patch_engine):
    km = KnowledgeManager("test")
    _ = km.searcher
    HybridSearcher = patch_engine["search.searcher"].HybridSearcher
    call_kwargs = HybridSearcher.call_args.kwargs
    assert "bm25_weight" in call_kwargs
    assert "vector_weight" in call_kwargs
    assert "invalid_key" not in call_kwargs
    assert call_kwargs.get("embedding_model") == "emb-model"


def test_build_context_no_searcher(patch_engine):
    km = KnowledgeManager("test")
    km._searcher = None
    km._searcher_initialized = True
    km.memory.search.return_value = []
    ctx = km.build_context("query")
    assert ctx.markdown == ""
    assert ctx.chunks_used == 0
    assert ctx.memories_used == 0


def test_build_context_respects_token_budget(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    results = []
    for i in range(20):
        results.append({"content": "x" * 400, "source": f"src/{i}.py"})
    fake_searcher.search.return_value = results
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = []

    ctx = km.build_context("query", max_tokens=1000, top_k=20)
    assert ctx.chunks_used <= 7
    assert ctx.chunks_used >= 1
    assert "## Project Context" in ctx.markdown


def test_search_delegates(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = [{"content": "hi"}]
    km._searcher = fake_searcher
    km._searcher_initialized = True

    result = km.search("hello", top_k=5)
    fake_searcher.search.assert_called_once_with("hello", top_k=5)
    assert result == [{"content": "hi"}]


def test_remember_recall(patch_engine):
    km = KnowledgeManager("test")
    km.remember("key1", "content1", namespace="ns")
    km.memory.store.assert_called_once_with(namespace="ns", key="key1", content="content1", ttl=None)

    km.memory.search.return_value = [{"content": "found"}]
    result = km.recall("query", namespace="ns", top_k=3)
    km.memory.search.assert_called_once_with(query="query", namespace="ns", top_k=3)
    assert result == [{"content": "found"}]


def test_knowledge_session_yields_and_closes(patch_engine):
    registry = MagicMock()
    with patch("mullet.core.knowledge.tools.register") as mock_register:
        with patch.object(KnowledgeManager, "close") as mock_close:
            with knowledge_session("test", register_tools=registry) as km:
                assert isinstance(km, KnowledgeManager)
                mock_register.assert_called_once_with(registry, km)
            mock_close.assert_called_once()


def test_build_knowledge_environment(patch_engine):
    with patch("mullet.core.knowledge.context.build_system_prompt") as mock_build_sp:
        with patch("mullet.core.knowledge.tools.register") as mock_register:
            with patch.object(KnowledgeManager, "build_context", return_value=KnowledgeContext()):
                mock_build_sp.return_value = "augmented"
                registry = MagicMock()
                km, ctx, augmented = build_knowledge_environment("test", "task", registry, "sys")
                assert isinstance(km, KnowledgeManager)
                assert isinstance(ctx, KnowledgeContext)
                assert augmented == "augmented"
                mock_register.assert_called_once_with(registry, km)
                mock_build_sp.assert_called_once_with("sys", ctx, "test")


import os
from unittest.mock import patch, MagicMock

from mullet.core.knowledge.manager import _get_engine_module


def _fake_path_class(existing_dir=None, has_client_py=False):
    client_py_path = os.path.join(existing_dir, "core", "client.py") if existing_dir else None

    class FakePath:
        def __init__(self, p):
            self._p = str(p)

        def exists(self):
            p = self._p
            if p == existing_dir:
                return True
            if has_client_py and p == client_py_path:
                return True
            return False

        def __truediv__(self, other):
            return FakePath(os.path.join(self._p, str(other)))

        def resolve(self):
            return self

        @property
        def parents(self):
            return [self, self, self, self, FakePath("/nonexistent")]

        def __str__(self):
            return self._p

    return FakePath


@patch("builtins.__import__")
def test_get_engine_module_vendored_success(mock_import):
    mock_mod = MagicMock()
    mock_import.return_value = mock_mod
    result = _get_engine_module("client")
    mock_import.assert_called_once_with("mullet.knowledge.engine.client", fromlist=["client"])
    assert result == mock_mod


def test_get_engine_module_vendored_fail_external_success():
    fake_path = "/fake/trove"
    FakePath = _fake_path_class(existing_dir=fake_path, has_client_py=True)

    with patch.dict(os.environ, {"TRAWL_PATH": fake_path}):
        with patch("builtins.__import__") as mock_import:
            with patch("mullet.core.knowledge.manager.Path", FakePath):
                mock_mod = MagicMock()

                def side_effect(name, **kwargs):
                    if name == "mullet.knowledge.engine.client":
                        raise ImportError("no vendored")
                    if name == "core.client":
                        return mock_mod
                    raise ImportError(f"unexpected {name}")

                mock_import.side_effect = side_effect

                result = _get_engine_module("client")
                assert result == mock_mod


def test_get_engine_module_vendored_fail_external_import_error():
    fake_path = "/fake/trove"
    FakePath = _fake_path_class(existing_dir=fake_path, has_client_py=True)

    with patch.dict(os.environ, {"TRAWL_PATH": fake_path}):
        with patch("builtins.__import__") as mock_import:
            with patch("mullet.core.knowledge.manager.Path", FakePath):
                mock_import.side_effect = ImportError("all fail")
                with pytest.raises(ImportError, match="Knowledge engine module 'client' not found"):
                    _get_engine_module("client")


def test_get_engine_module_path_exists_no_client_py():
    fake_path = "/fake/trove"
    FakePath = _fake_path_class(existing_dir=fake_path, has_client_py=False)

    with patch.dict(os.environ, {"TRAWL_PATH": fake_path}):
        with patch("builtins.__import__") as mock_import:
            with patch("mullet.core.knowledge.manager.Path", FakePath):
                mock_import.side_effect = ImportError("no vendored")
                with pytest.raises(ImportError, match="Knowledge engine module 'client' not found"):
                    _get_engine_module("client")


def test_get_engine_module_all_fail_no_paths():
    FakePath = _fake_path_class(existing_dir=None, has_client_py=False)

    with patch.dict(os.environ, {"TRAWL_PATH": ""}):
        with patch("builtins.__import__") as mock_import:
            with patch("mullet.core.knowledge.manager.Path", FakePath):
                mock_import.side_effect = ImportError("no vendored")
                with pytest.raises(ImportError, match="Knowledge engine module"):
                    _get_engine_module("client")


def test_searcher_init_failure_sets_none(patch_engine):
    with patch("mullet.core.knowledge.manager._get_engine_module") as mock_gem:
        def side_effect(name):
            if name == "search.searcher":
                raise RuntimeError("no embeddings")
            return patch_engine[name]

        mock_gem.side_effect = side_effect

        km = KnowledgeManager("test")
        assert km.searcher is None
        assert km._searcher_initialized is True


def test_build_context_search_exception(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.side_effect = RuntimeError("search exploded")
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = []

    ctx = km.build_context("query")
    assert ctx.markdown == ""
    assert ctx.chunks_used == 0


def test_build_context_token_budget_break(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = [
        {"content": "small", "source": "a.py"},
        {"content": "x" * 4000, "source": "b.py"},
    ]
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = []

    ctx = km.build_context("query", max_tokens=100)
    assert ctx.chunks_used == 1
    assert "small" in ctx.markdown
    assert "b.py" not in ctx.markdown


def test_build_context_include_graph_success(patch_engine):
    graph_search_mod = MagicMock()
    graph_store_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.stats = {"entities": 5}
    graph_store_mod.KnowledgeGraph.return_value = mock_graph

    entity = {"name": "Entity1", "type": "class", "description": "A class"}
    graph_search_mod.local_search.return_value = [entity]

    def side_effect(name):
        if name == "graph.search":
            return graph_search_mod
        if name == "graph.store":
            return graph_store_mod
        return patch_engine[name]

    with patch("mullet.core.knowledge.manager._get_engine_module", side_effect=side_effect):
        km = KnowledgeManager("test")
        fake_searcher = MagicMock()
        fake_searcher.search.return_value = []
        km._searcher = fake_searcher
        km._searcher_initialized = True
        km.memory.search.return_value = []

        ctx = km.build_context("query", include_graph=True)
        assert "## Related Entities" in ctx.markdown
        assert "Entity1" in ctx.markdown
        graph_search_mod.local_search.assert_called_once_with("query", mock_graph, fake_searcher, top_k=5)


def test_build_context_include_graph_no_entities(patch_engine):
    graph_search_mod = MagicMock()
    graph_store_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.stats = {"entities": 0}
    graph_store_mod.KnowledgeGraph.return_value = mock_graph

    def side_effect(name):
        if name == "graph.search":
            return graph_search_mod
        if name == "graph.store":
            return graph_store_mod
        return patch_engine[name]

    with patch("mullet.core.knowledge.manager._get_engine_module", side_effect=side_effect):
        km = KnowledgeManager("test")
        fake_searcher = MagicMock()
        fake_searcher.search.return_value = []
        km._searcher = fake_searcher
        km._searcher_initialized = True
        km.memory.search.return_value = []

        ctx = km.build_context("query", include_graph=True)
        assert "## Related Entities" not in ctx.markdown


def test_build_context_include_graph_no_searcher(patch_engine):
    def side_effect(name):
        if name == "graph.search":
            return MagicMock()
        if name == "graph.store":
            return MagicMock()
        return patch_engine[name]

    with patch("mullet.core.knowledge.manager._get_engine_module", side_effect=side_effect):
        km = KnowledgeManager("test")
        km._searcher = None
        km._searcher_initialized = True
        km.memory.search.return_value = []

        ctx = km.build_context("query", include_graph=True)
        assert "## Related Entities" not in ctx.markdown


def test_build_context_include_graph_exception(patch_engine):
    def side_effect(name):
        if name.startswith("graph"):
            raise ImportError("no graph")
        return patch_engine[name]

    with patch("mullet.core.knowledge.manager._get_engine_module", side_effect=side_effect):
        km = KnowledgeManager("test")
        fake_searcher = MagicMock()
        fake_searcher.search.return_value = []
        km._searcher = fake_searcher
        km._searcher_initialized = True
        km.memory.search.return_value = []

        ctx = km.build_context("query", include_graph=True)
        assert ctx.markdown == ""


def test_build_context_graph_entity_token_budget(patch_engine):
    graph_search_mod = MagicMock()
    graph_store_mod = MagicMock()
    mock_graph = MagicMock()
    mock_graph.stats = {"entities": 5}
    graph_store_mod.KnowledgeGraph.return_value = mock_graph

    graph_search_mod.local_search.return_value = [
        {"name": "E1", "type": "class", "description": "x" * 4000},
    ]

    def side_effect(name):
        if name == "graph.search":
            return graph_search_mod
        if name == "graph.store":
            return graph_store_mod
        return patch_engine[name]

    with patch("mullet.core.knowledge.manager._get_engine_module", side_effect=side_effect):
        km = KnowledgeManager("test")
        fake_searcher = MagicMock()
        fake_searcher.search.return_value = []
        km._searcher = fake_searcher
        km._searcher_initialized = True
        km.memory.search.return_value = []

        ctx = km.build_context("query", include_graph=True, max_tokens=100)
        assert "## Related Entities" in ctx.markdown
        assert "E1" not in ctx.markdown


def test_build_context_include_memory_with_results(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = [{"content": "remember this"}, {"content": "and that"}]

    ctx = km.build_context("query", include_memory=True)
    assert "## Memories" in ctx.markdown
    assert "remember this" in ctx.markdown
    assert "and that" in ctx.markdown
    assert ctx.memories_used == 2


def test_build_context_include_memory_token_budget(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = [
        {"content": "x" * 400},
        {"content": "y" * 4000},
    ]

    ctx = km.build_context("query", include_memory=True, max_tokens=500)
    assert ctx.memories_used == 1
    assert "x" * 400 in ctx.markdown
    assert "y" * 4000 not in ctx.markdown


def test_build_context_include_memory_exception(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.side_effect = RuntimeError("memory fail")

    ctx = km.build_context("query", include_memory=True)
    assert "## Memories" not in ctx.markdown
    assert ctx.memories_used == 0


def test_build_context_memory_non_dict(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = []
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = ["plain string memory"]

    ctx = km.build_context("query", include_memory=True)
    assert "plain string memory" in ctx.markdown


def test_build_context_chunk_without_source(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.search.return_value = [{"content": "no source here"}]
    km._searcher = fake_searcher
    km._searcher_initialized = True
    km.memory.search.return_value = []

    ctx = km.build_context("query")
    assert "no source here" in ctx.markdown
    assert "**" not in ctx.markdown


def test_search_returns_empty_when_searcher_none(patch_engine):
    km = KnowledgeManager("test")
    km._searcher = None
    km._searcher_initialized = True

    result = km.search("hello")
    assert result == []


def test_recall_exception_falls_back_to_list(patch_engine):
    km = KnowledgeManager("test")
    km.memory.search.side_effect = RuntimeError("search fail")
    km.memory.list.return_value = [{"content": "listed"}]

    result = km.recall("query", namespace="ns")
    km.memory.list.assert_called_once_with(namespace="ns")
    assert result == [{"content": "listed"}]


def test_recall_exception_no_namespace_returns_empty(patch_engine):
    km = KnowledgeManager("test")
    km.memory.search.side_effect = RuntimeError("search fail")

    result = km.recall("query")
    km.memory.list.assert_not_called()
    assert result == []


def test_create_workflow(patch_engine):
    km = KnowledgeManager("test")
    km.workflows.create.return_value = "wf-123"

    result = km.create_workflow("myflow", [{"step": 1}])
    km.workflows.create.assert_called_once_with(name="myflow", steps=[{"step": 1}])
    assert result == "wf-123"


def test_close_closes_searcher(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    km._searcher = fake_searcher

    km.close()
    fake_searcher.close.assert_called_once()


def test_close_searcher_close_raises(patch_engine):
    km = KnowledgeManager("test")
    fake_searcher = MagicMock()
    fake_searcher.close.side_effect = RuntimeError("close fail")
    km._searcher = fake_searcher

    # Should not raise
    km.close()
    fake_searcher.close.assert_called_once()


def test_close_no_searcher(patch_engine):
    km = KnowledgeManager("test")
    km._searcher = None

    km.close()
    # No error


def test_knowledge_session_close_exception(monkeypatch):
    """Lines 274-275: knowledge_session swallows KM.close() exceptions."""
    from mullet.core.knowledge.manager import knowledge_session
    from unittest.mock import MagicMock, patch

    mock_km = MagicMock()
    mock_km.close.side_effect = RuntimeError("close boom")

    with patch("mullet.core.knowledge.manager.KnowledgeManager", return_value=mock_km):
        with knowledge_session("test") as km:
            assert km is mock_km
    # No exception propagated
    mock_km.close.assert_called_once()
