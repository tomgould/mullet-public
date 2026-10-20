from mullet.core.knowledge.health import HealthChecker, HealthStatus

# Mock check functions
def mock_storage_healthy():
    return True, "Storage is accessible"

def mock_storage_unhealthy():
    return False, "Storage not found"

def mock_index_exists():
    return True, "Index exists"

def mock_index_missing():
    return False, "Index is missing"

def mock_embedding_model_loadable():
    return True, "Embedding model loaded"

def mock_embedding_model_unloadable():
    return False, "Embedding model failed to load"

def mock_check_raises_exception():
    raise Exception("Simulated error during check")

def test_all_healthy():
    checker = HealthChecker(
        storage_check=mock_storage_healthy,
        index_check=mock_index_exists,
        embedding_model_check=mock_embedding_model_loadable
    )
    result = checker.check()
    assert result["status"] == HealthStatus.HEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is True
    assert result["details"]["index_exists"]["healthy"] is True
    assert result["details"]["embedding_model_loadable"]["healthy"] is True

def test_one_unhealthy():
    checker = HealthChecker(
        storage_check=mock_storage_healthy,
        index_check=mock_index_missing,
        embedding_model_check=mock_embedding_model_loadable
    )
    result = checker.check()
    assert result["status"] == HealthStatus.UNHEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is True
    assert result["details"]["index_exists"]["healthy"] is False
    assert result["details"]["embedding_model_loadable"]["healthy"] is True

def test_all_unhealthy():
    checker = HealthChecker(
        storage_check=mock_storage_unhealthy,
        index_check=mock_index_missing,
        embedding_model_check=mock_embedding_model_unloadable
    )
    result = checker.check()
    assert result["status"] == HealthStatus.UNHEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is False
    assert result["details"]["index_exists"]["healthy"] is False
    assert result["details"]["embedding_model_loadable"]["healthy"] is False

def test_check_raises_exception():
    checker = HealthChecker(
        storage_check=mock_check_raises_exception,
        index_check=mock_index_exists
    )
    result = checker.check()
    assert result["status"] == HealthStatus.UNHEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is False
    assert "Error during check" in result["details"]["storage_accessible"]["message"]
    assert result["details"]["index_exists"]["healthy"] is True

def test_no_checks_configured():
    checker = HealthChecker()
    result = checker.check()
    assert result["status"] == HealthStatus.HEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is True
    assert result["details"]["storage_accessible"]["message"] == "Check not configured"

def test_partial_checks_configured():
    checker = HealthChecker(
        storage_check=mock_storage_healthy
    )
    result = checker.check()
    assert result["status"] == HealthStatus.HEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is True
    assert result["details"]["index_exists"]["healthy"] is True # Default to healthy if not configured

def test_only_one_check_configured_and_unhealthy():
    checker = HealthChecker(
        embedding_model_check=mock_embedding_model_unloadable
    )
    result = checker.check()
    assert result["status"] == HealthStatus.UNHEALTHY.value
    assert result["details"]["embedding_model_loadable"]["healthy"] is False

def test_multiple_checks_with_one_exception():
    checker = HealthChecker(
        storage_check=mock_storage_healthy,
        index_check=mock_check_raises_exception,
        embedding_model_check=mock_embedding_model_loadable
    )
    result = checker.check()
    assert result["status"] == HealthStatus.UNHEALTHY.value
    assert result["details"]["storage_accessible"]["healthy"] is True
    assert result["details"]["index_exists"]["healthy"] is False
    assert "Error during check" in result["details"]["index_exists"]["message"]
    assert result["details"]["embedding_model_loadable"]["healthy"] is True

