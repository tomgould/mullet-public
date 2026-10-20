from unittest.mock import patch, MagicMock
from mullet.core.providers.lmstudio_manager import LMStudioManager


class TestLMStudioManager:
    
    def test_get_models_success(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "_make_request") as mock_request:
            mock_request.return_value = {"data": [{"id": "m1", "type": "llm", "state": "loaded"}]}
            result = mgr.get_models()
            assert result == [{"id": "m1", "type": "llm", "state": "loaded"}]
    
    def test_get_models_none_returns_empty(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "_make_request") as mock_request:
            mock_request.return_value = None
            result = mgr.get_models()
            assert result == []
    
    def test_get_loaded_models_filters_type_and_state(self):
        mgr = LMStudioManager()
        models = [
            {"id": "m1", "type": "llm", "state": "loaded"},
            {"id": "m2", "type": "llm", "state": "not-loaded"},
            {"id": "m3", "type": "vlm", "state": "loaded"},
            {"id": "m4", "type": "embeddings", "state": "loaded"}
        ]
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = models
            result = mgr.get_loaded_models()
            assert result == [{"id": "m1", "type": "llm", "state": "loaded"}]
    
    def test_pick_best_model_prefers_loaded(self):
        mgr = LMStudioManager()
        models = [
            {"id": "m1", "type": "llm", "state": "not-loaded"},
            {"id": "m2", "type": "llm", "state": "loaded"}
        ]
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = models
            result = mgr.pick_best_model()
            assert result == {"id": "m2", "type": "llm", "state": "loaded"}
    
    def test_pick_best_model_skips_embedding_in_id(self):
        mgr = LMStudioManager()
        models = [
            {"id": "nomic-embedding", "type": "llm", "state": "not-loaded"},
            {"id": "qwen2.5-coder", "type": "llm", "state": "not-loaded"}
        ]
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = models
            result = mgr.pick_best_model()
            assert result == {"id": "qwen2.5-coder", "type": "llm", "state": "not-loaded"}
    
    def test_pick_best_model_any_llm_fallback(self):
        mgr = LMStudioManager()
        models = [
            {"id": "vlm-model", "type": "vlm", "state": "loaded"}
        ]
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = models
            result = mgr.pick_best_model()
            assert result is None
    
    def test_pick_best_model_empty(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_models") as mock_get:
            mock_get.return_value = []
            result = mgr.pick_best_model()
            assert result is None
    
    def test_run_lms_command_success(self):
        mgr = LMStudioManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = mgr._run_lms_command(["test"])
            assert result == (True, "ok")
    
    def test_run_lms_command_failure(self):
        mgr = LMStudioManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error msg")
            result = mgr._run_lms_command(["test"])
            assert result == (False, "error msg")
    
    def test_run_lms_command_not_found(self):
        mgr = LMStudioManager()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            result = mgr._run_lms_command(["test"])
            assert result[0] is False
            assert "not found" in result[1]
    
    def test_ensure_loaded_already_loaded(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "_run_lms_command") as mock_run:
            mock_loaded.return_value = [{"id": "test-model"}]
            result = mgr.ensure_loaded("test-model")
            assert result is True
            mock_run.assert_not_called()
    
    def test_ensure_loaded_loads_successfully(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "_run_lms_command") as mock_run, \
             patch("time.sleep"):
            mock_loaded.side_effect = [[], [{"id": "test-model"}]]
            mock_run.return_value = (True, "loaded")
            result = mgr.ensure_loaded("test-model")
            assert result is True
    
    def test_ensure_loaded_command_fails(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "_run_lms_command") as mock_run:
            mock_loaded.return_value = []
            mock_run.return_value = (False, "failed")
            result = mgr.ensure_loaded("test-model")
            assert result is False
    
    def test_unload_success(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "_run_lms_command") as mock_run:
            mock_run.return_value = (True, "unloaded")
            result = mgr.unload("test-model")
            assert result is True
    
    def test_unload_failure(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "_run_lms_command") as mock_run:
            mock_run.return_value = (False, "error")
            result = mgr.unload("test-model")
            assert result is False
    
    def test_get_or_load_provider_already_loaded(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_models") as mock_models, \
             patch.object(mgr, "get_loaded_models") as mock_loaded, \
             patch.object(mgr, "pick_best_model") as mock_pick:
            mock_models.return_value = [{"id": "best", "type": "llm", "state": "loaded"}]
            mock_loaded.return_value = [{"id": "best", "type": "llm", "state": "loaded"}]
            mock_pick.return_value = {"id": "best", "type": "llm", "state": "loaded"}
            result = mgr.get_or_load_provider()
            assert result == ("best", True)
    
    def test_get_or_load_provider_no_models(self):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_models") as mock_models, \
             patch.object(mgr, "pick_best_model") as mock_pick:
            mock_models.return_value = []
            mock_pick.return_value = None
            result = mgr.get_or_load_provider()
            assert result == (None, False)

    @patch("shutil.which")
    def test_find_lms_cli_returns_none_when_absent(self, mock_which):
        from mullet.core.providers.lmstudio_manager import _find_lms_cli
        mock_which.return_value = None
        with patch("os.path.isfile", return_value=False):
            result = _find_lms_cli()
            assert result == "lms" # last-resort if not found anywhere

    @patch("shutil.which")
    def test_find_lms_cli_returns_path_when_present(self, mock_which):
        from mullet.core.providers.lmstudio_manager import _find_lms_cli
        mock_which.return_value = "/usr/local/bin/lms"
        result = _find_lms_cli()
        assert result == "/usr/local/bin/lms"
        mock_which.assert_called_once_with("lms")

    def test_make_request_returns_none_on_connection_error(self):
        mgr = LMStudioManager()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection error") # Simulate URLError or any other exception
            result = mgr._make_request("/test")
            assert result is None
    
    def test_make_request_parses_json_on_success(self):
        mgr = LMStudioManager()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.read.return_value = b'{"data": [{"id": "x"}]}'
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            result = mgr._make_request("/test")
            assert result == {"data": [{"id": "x"}]}

    @patch("mullet.core.providers.lmstudio_manager._find_lms_cli", return_value="/bin/lms")
    def test_reload_current_model_calls_lms_cli(self, mock_find_lms_cli):
        mgr = LMStudioManager()
        with patch.object(mgr, "get_loaded_models") as mock_get_loaded, \
             patch.object(mgr, "_run_lms_command") as mock_run_lms, \
             patch("time.sleep"):

            # Simulate a model being loaded initially
            mock_get_loaded.side_effect = [
                [{"id": "qwen-model", "type": "llm", "state": "loaded"}], # First call to get_loaded_models (for reload_current_model)
                [], # After unload, model is not loaded (for ensure_loaded)
                [{"id": "qwen-model", "type": "llm", "state": "loaded"}] # After load, model is loaded (for ensure_loaded)
            ]
            mock_run_lms.side_effect = [
                (True, "unloaded"), # For unload()
                (True, "loaded")    # For ensure_loaded()
            ]

            result = mgr.reload_current_model(ttl=120)

            assert result is True
            
            # Assert unload was called
            mock_run_lms.call_args_list[0].assert_called_with(["unload", "qwen-model"])
            # Assert load was called
            mock_run_lms.call_args_list[1].assert_called_with(["load", "--ttl", "120", "--yes", "qwen-model"])