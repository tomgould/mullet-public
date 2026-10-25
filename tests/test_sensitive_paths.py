"""The file and docker tools refuse credential locations, and the list is the user's to change."""

import json
import os

import pytest

from mullet.core.tools.docker import _check_blocked_flags, _mount_sources
from mullet.core.tools.file_ops import file_ops_handler
from mullet.core.tools.sensitive_paths import DEFAULT_DENY, covers, denied_by, load_deny


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MULLET_CONFIG_DIR", str(tmp_path / "cfg"))
    (tmp_path / "cfg").mkdir()
    (tmp_path / ".ssh").mkdir()
    (tmp_path / ".ssh" / "id_rsa").write_text("PRIVATE")
    (tmp_path / "project").mkdir()
    (tmp_path / "project" / "main.py").write_text("print('hi')\n")
    (tmp_path / "project" / ".env").write_text("TOKEN=x\n")
    return tmp_path


def test_default_list_covers_ssh_cloud_and_mullet_state(home):
    for path in (".ssh/id_rsa", ".aws/credentials", ".config/mullet/.env", ".mullet/config.yaml"):
        assert denied_by(home / path, DEFAULT_DENY), path


def test_env_files_are_refused_at_any_depth(home):
    assert denied_by(home / "project" / ".env", DEFAULT_DENY) == ".env"
    assert denied_by(home / "project" / ".env.production", DEFAULT_DENY) == ".env.*"
    assert denied_by(home / "project" / "deploy" / "server.pem", DEFAULT_DENY) == "*.pem"


def test_ordinary_files_are_not_refused(home):
    assert denied_by(home / "project" / "main.py", DEFAULT_DENY) is None
    assert denied_by(home / "project" / "environment.md", DEFAULT_DENY) is None


def test_a_symlink_cannot_hide_a_denied_target(home):
    link = home / "project" / "innocent.txt"
    link.symlink_to(home / ".ssh" / "id_rsa")
    assert denied_by(link, DEFAULT_DENY)


def test_covers_finds_a_denied_location_inside_a_parent(home):
    assert covers(home, DEFAULT_DENY)
    assert covers(home / ".ssh", DEFAULT_DENY)
    assert covers(home / "project", DEFAULT_DENY) is None


def test_the_user_can_replace_the_list(home):
    (home / "cfg" / "config.yaml").write_text("delegation:\n  file_deny: ['~/private']\n")
    assert load_deny() == ("~/private",)
    assert denied_by(home / ".ssh" / "id_rsa", load_deny()) is None
    assert denied_by(home / "private" / "x", load_deny()) == "~/private"


def test_a_bad_setting_falls_back_to_the_default(home):
    (home / "cfg" / "config.yaml").write_text("delegation:\n  file_deny: 5\n")
    assert load_deny() == DEFAULT_DENY


def test_the_file_tool_refuses_a_denied_read_and_write_but_not_a_normal_one(home):
    refused = json.loads(file_ops_handler(action="read_file", path=str(home / ".ssh" / "id_rsa")))
    assert "deny list" in refused["error"]
    refused = json.loads(file_ops_handler(action="write_file", path=str(home / ".ssh" / "authorized_keys"),
                                          content="x"))
    assert "deny list" in refused["error"] and not (home / ".ssh" / "authorized_keys").exists()
    env = json.loads(file_ops_handler(action="read_file", path=str(home / "project" / ".env")))
    assert "deny list" in env["error"]
    ok = json.loads(file_ops_handler(action="read_file", path=str(home / "project" / "main.py")))
    assert ok["lines"] == ["print('hi')"]


@pytest.mark.parametrize("options", [
    "-v /:/host", "-v /:/host:ro", "--volume=/:/h", "--volume /:/h", "-v/:/h",
    "--mount type=bind,source=/,target=/h", "--mount=type=bind,src=/,dst=/h",
    "-v /var/run/docker.sock:/var/run/docker.sock", "-v ~/.ssh:/root/.ssh",
    "-v ~:/home", "-v $HOME:/home", "--device /dev/sda", "--ipc=host", "--userns=host",
])
def test_docker_refuses_dangerous_options(home, options):
    assert _check_blocked_flags(options)


@pytest.mark.parametrize("options", ["-d --rm", "-v myvolume:/data", "-e A=b -p 8080:80"])
def test_docker_allows_ordinary_options(home, options):
    assert _check_blocked_flags(options) is None


def test_docker_allows_a_project_mount(home):
    assert _check_blocked_flags(f"-v {home / 'project'}:/work") is None


def test_mount_sources_reads_every_spelling():
    assert _mount_sources("-v /a:/x --volume /b:/y --volume=/c:/z -v/d:/w") == ["/a", "/b", "/c", "/d"]
    assert _mount_sources("--mount type=bind,source=/e,target=/x") == ["/e"]
    assert _mount_sources("-v named:/data") == []
