"""Unit tests for workflow and agents handlers in mullet/server/mcp_knowledge.py."""

from unittest.mock import MagicMock, patch

from mullet.server.mcp_knowledge import (
    handle_agents_get,
    handle_agents_list,
    handle_workflow_create,
    handle_workflow_list,
    handle_workflow_status,
    handle_workflow_update,
)


class TestWorkflowCreate:
    def test_workflow_create_missing_name(self):
        result = handle_workflow_create({})
        assert "Error" in result

    def test_workflow_create_missing_steps(self):
        result = handle_workflow_create({"name": "test"})
        assert "Error" in result


class TestWorkflowStatus:
    def test_workflow_status_missing_id(self):
        result = handle_workflow_status({})
        assert "Error" in result


class TestWorkflowUpdate:
    def test_workflow_update_missing_fields(self):
        result = handle_workflow_update({"workflow_id": "x"})
        assert "Error" in result


class TestWorkflowList:
    def test_workflow_list_with_mock_engine(self):
        mock_engine = MagicMock()
        mock_engine.list.return_value = []
        with patch("mullet.server.mcp_knowledge._get_workflow_engine", return_value=mock_engine):
            result = handle_workflow_list({})
        assert "No workflows found." in result

    def test_workflow_list_with_results(self):
        mock_engine = MagicMock()
        mock_engine.list.return_value = [
            {
                "name": "Test",
                "status": "pending",
                "id": "wf1",
                "completed_steps": 0,
                "total_steps": 2,
            }
        ]
        with patch("mullet.server.mcp_knowledge._get_workflow_engine", return_value=mock_engine):
            result = handle_workflow_list({})
        assert "Test" in result
        assert "wf1" in result


class TestAgentsList:
    def test_agents_list_empty(self):
        mock_loader = MagicMock()
        mock_loader.list.return_value = []
        with patch("mullet.knowledge.engine.agents.AgentLoader", return_value=mock_loader):
            result = handle_agents_list({})
        assert "No agents found." in result


class TestAgentsGet:
    def test_agents_get_missing_name(self):
        result = handle_agents_get({})
        assert "Error" in result or "required" in result.lower()
