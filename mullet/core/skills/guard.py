"""Security scanner for user-created skills.

Called by SkillLoader._load_skill() for non-builtin skills.
"""

import re
from pathlib import Path
from typing import Dict, Any

def scan_skill(skill_dir: Path) -> Dict[str, Any]:
    """
    Security scanner for user-created skills.
    Scans SKILL.md and any .py/.sh files for:
    - shell injection patterns
    - env var exfiltration
    - destructive commands (rm -rf /)
    - network calls to suspicious domains
    """
    threats = []
    warnings = []
    
    # Define threat patterns (regex, description, severity)
    # This is a simplified set based on the example, more comprehensive patterns would be needed.
    THREAT_PATTERNS = [
        (r'rm\s+-rf\s+/', "Destructive command: rm -rf /", "critical"),
        (r'curl\s+[^\n]*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "Potential env var exfiltration via curl", "high"),
        (r'wget\s+[^\n]*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "Potential env var exfiltration via wget", "high"),
        (r'os\.environ', "Access to os.environ (potential env dump)", "medium"),
        (r'subprocess\.run', "Subprocess execution (potential shell injection)", "medium"),
        (r'eval\(', "Use of eval() (potential code injection)", "high"),
        (r'exec\(', "Use of exec() (potential code injection)", "high"),
        (r'import\s+os', "Import of os module", "low"),
        (r'import\s+shutil', "Import of shutil module", "low"),
        (r'http[s]?://(bad-domain\.com|suspicious\.org)', "Network call to suspicious domain", "high"), # Example suspicious domains
    ]

    def _scan_file(filepath: Path):
        with open(filepath) as f:
            for i, line in enumerate(f.readlines()):
                for pattern, description, severity in THREAT_PATTERNS:
                    if re.search(pattern, line):
                        threats.append({
                            "file": str(filepath),
                            "line": i + 1,
                            "match": line.strip(),
                            "description": description,
                            "severity": severity
                        })

    # Scan SKILL.md
    skill_md_path = skill_dir / "SKILL.md"
    if skill_md_path.exists():
        _scan_file(skill_md_path)
    else:
        warnings.append(f"SKILL.md not found in {skill_dir}")

    # Scan .py and .sh files
    for filepath in skill_dir.rglob("*.py"):
        _scan_file(filepath)
    for filepath in skill_dir.rglob("*.sh"):
        _scan_file(filepath)

    safe = not bool(threats)
    return {
        "threats": threats,
        "safe": safe,
        "warnings": warnings
    }
