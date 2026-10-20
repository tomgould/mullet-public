import pytest
from mullet.core.skills.discovery import SkillDiscovery

@pytest.fixture
def setup_skill_dirs(tmp_path):
    # Skill 1: basic skill with triggers
    skill1_dir = tmp_path / "skill1"
    skill1_dir.mkdir()
    with open(skill1_dir / "SKILL.md", "w") as f:
        f.write("Description for skill1.\n")
        f.write("Triggers: trigger_one, trigger_two\n")
    
    # Skill 2: skill in a nested directory, no triggers
    nested_dir = tmp_path / "category" / "skill2"
    nested_dir.mkdir(parents=True)
    with open(nested_dir / "SKILL.md", "w") as f:
        f.write("Description for skill2.\n")

    # Skill 3: another skill with multiple triggers and extra spaces
    skill3_dir = tmp_path / "skill3"
    skill3_dir.mkdir()
    with open(skill3_dir / "SKILL.md", "w") as f:
        f.write("Triggers: trigger_alpha ,  trigger_beta,trigger_gamma  \n")

    # Directory without SKILL.md
    no_skill_dir = tmp_path / "no_skill"
    no_skill_dir.mkdir()
    with open(no_skill_dir / "README.md", "w") as f:
        f.write("Just a readme.")

    return tmp_path

def test_discover_basic_skills(setup_skill_dirs):
    root_dir = setup_skill_dirs
    discovery = SkillDiscovery()
    skills = discovery.discover(str(root_dir))
    
    skill_names = sorted([s["name"] for s in skills])
    assert skill_names == ["skill1", "skill2", "skill3"]
    assert len(skills) == 3

def test_skill1_details(setup_skill_dirs):
    root_dir = setup_skill_dirs
    discovery = SkillDiscovery()
    skills = discovery.discover(str(root_dir))
    skill1 = next(s for s in skills if s["name"] == "skill1")
    
    assert skill1["name"] == "skill1"
    assert skill1["path"] == str(root_dir / "skill1" / "SKILL.md")
    assert sorted(skill1["triggers"]) == ["trigger_one", "trigger_two"]

def test_skill2_details(setup_skill_dirs):
    root_dir = setup_skill_dirs
    discovery = SkillDiscovery()
    skills = discovery.discover(str(root_dir))
    skill2 = next(s for s in skills if s["name"] == "skill2")
    
    assert skill2["name"] == "skill2"
    assert skill2["path"] == str(root_dir / "category" / "skill2" / "SKILL.md")
    assert skill2["triggers"] == []

def test_skill3_details_with_multiple_spaced_triggers(setup_skill_dirs):
    root_dir = setup_skill_dirs
    discovery = SkillDiscovery()
    skills = discovery.discover(str(root_dir))
    skill3 = next(s for s in skills if s["name"] == "skill3")
    
    assert skill3["name"] == "skill3"
    assert skill3["path"] == str(root_dir / "skill3" / "SKILL.md")
    assert sorted(skill3["triggers"]) == ["trigger_alpha", "trigger_beta", "trigger_gamma"]

def test_discover_empty_directory(tmp_path):
    discovery = SkillDiscovery()
    skills = discovery.discover(str(tmp_path))
    assert skills == []

def test_discover_directory_without_skill_md(tmp_path):
    no_skill_dir = tmp_path / "no_skill"
    no_skill_dir.mkdir()
    with open(no_skill_dir / "README.md", "w") as f:
        f.write("Just a readme.")
    
    discovery = SkillDiscovery()
    skills = discovery.discover(str(tmp_path))
    assert skills == []

def test_discover_non_existent_directory():
    discovery = SkillDiscovery()
    skills = discovery.discover("/non_existent_path_12345")
    assert skills == []

def test_discover_with_mixed_content(setup_skill_dirs):
    root_dir = setup_skill_dirs
    # Add another skill with no triggers
    skill4_dir = root_dir / "another_category" / "skill4"
    skill4_dir.mkdir(parents=True)
    with open(skill4_dir / "SKILL.md", "w") as f:
        f.write("Description for skill4.")

    discovery = SkillDiscovery()
    skills = discovery.discover(str(root_dir))
    
    skill_names = sorted([s["name"] for s in skills])
    assert skill_names == ["skill1", "skill2", "skill3", "skill4"]
    assert len(skills) == 4

    skill4 = next(s for s in skills if s["name"] == "skill4")
    assert skill4["triggers"] == []
