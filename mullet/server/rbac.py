"""Role-Based Access Control for project access."""

import sqlite3


ACCESS_LEVELS = {"read": 0, "write": 1, "owner": 2}


def check_project_access(
    user: dict,
    project_name: str,
    required_level: str = "read",
    db: sqlite3.Connection | None = None,
) -> bool:
    """Check if user has required access level to a project.

    Admins bypass all checks.
    """
    if user.get("role") == "admin":
        return True

    if not db:
        return False

    # Look up project by ID, name, or pool_name
    project_row = db.execute(
        "SELECT id FROM projects WHERE id = ? OR name = ? OR pool_name = ?",
        (project_name, project_name, project_name),
    ).fetchone()
    if not project_row:
        return False

    row = db.execute(
        "SELECT access_level FROM user_projects WHERE user_id = ? AND project_id = ?",
        (user["id"], project_row[0]),
    ).fetchone()

    if not row:
        return False

    return ACCESS_LEVELS.get(row[0], 0) >= ACCESS_LEVELS.get(required_level, 0)


def get_user_projects(user_id: str, db: sqlite3.Connection) -> list[dict]:
    """Get all projects a user has access to."""
    rows = db.execute(
        """SELECT p.name, p.pool_name, p.description, up.access_level
           FROM user_projects up
           JOIN projects p ON up.project_id = p.id
           WHERE up.user_id = ?""",
        (user_id,),
    ).fetchall()
    return [{"name": r[0], "client": r[1], "description": r[2], "access": r[3]} for r in rows]
