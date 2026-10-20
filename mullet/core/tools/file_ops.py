"""Batch file-ops tool: rename, copy, move with path-traversal safety."""
import glob
import os
import shutil
from pathlib import Path

def batch_rename(directory, pattern, replacement):
    """
    Renames files in a directory that match a given pattern.
    Args:
        directory (str): The directory to perform the rename operation in.
        pattern (str): The pattern to search for in filenames.
        replacement (str): The string to replace the pattern with.
    Returns:
        list: A list of tuples, where each tuple contains (old_path, new_path)
              for successfully renamed files.
    """
    renamed_files = []
    for filepath in Path(directory).glob("*"):
        if filepath.is_file():
            old_name = filepath.name
            if pattern in old_name:
                new_name = old_name.replace(pattern, replacement)
                new_filepath = filepath.parent / new_name
                shutil.move(filepath, new_filepath)
                renamed_files.append((str(filepath), str(new_filepath)))
    return renamed_files

def batch_copy(src_glob, dest_dir):
    """
    Copies files matching a glob pattern to a destination directory.
    Args:
        src_glob (str): The glob pattern for source files.
        dest_dir (str): The destination directory.
    Returns:
        list: A list of tuples, where each tuple contains (src_path, dest_path)
              for successfully copied files.
    """
    copied_files = []
    os.makedirs(dest_dir, exist_ok=True)
    for src_path in glob.glob(src_glob):
        if os.path.isfile(src_path):
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
            copied_files.append((src_path, dest_path))
    return copied_files

def batch_move(src_glob, dest_dir):
    """
    Moves files matching a glob pattern to a destination directory.
    Args:
        src_glob (str): The glob pattern for source files.
        dest_dir (str): The destination directory.
    Returns:
        list: A list of tuples, where each tuple contains (src_path, dest_path)
              for successfully moved files.
    """
    moved_files = []
    os.makedirs(dest_dir, exist_ok=True)
    for src_path in glob.glob(src_glob):
        if os.path.isfile(src_path):
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.move(src_path, dest_path)
            moved_files.append((src_path, dest_path))
    return moved_files

FILE_OPS_SCHEMA = {
    "name": "file_ops",
    "description": "File system operations: read, write, patch, list, and search files. PREFER 'patch' over 'write_file' when modifying existing files — patch finds old_text and replaces with new_text without rewriting the whole file.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read_file", "write_file", "patch", "list_directory", "search_files"],
                "description": "The file operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path.",
            },
            "content": {
                "type": "string",
                "description": "Content to write (for write_file).",
            },
            "old_text": {
                "type": "string",
                "description": "Text to find in file (for patch action).",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text (for patch action).",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern for search_files (e.g. '*.py').",
            },
            "offset": {
                "type": "integer",
                "description": "Line offset for read_file (0-indexed). Default 0.",
            },
            "limit": {
                "type": "integer",
                "description": "Max lines to read for read_file. Default 200.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Recurse into subdirectories for list_directory/search_files. Default false.",
            },
        },
        "required": ["action", "path"],
    },
}


def file_ops_handler(**kwargs) -> str:
    """Handle file operations tool calls."""
    import json

    action = kwargs.get("action")
    path_str = kwargs.get("path", "")

    if not path_str:
        return json.dumps({"error": "path is required"})

    path = Path(os.path.abspath(path_str)).resolve()

    # Security: block path traversal outside home/cwd
    allowed_roots = [Path.home(), Path.cwd(), Path("/tmp")]
    if not any(path.resolve().is_relative_to(r.resolve()) for r in allowed_roots):
        return json.dumps({"error": f"Path traversal blocked: {path} is outside allowed directories"})

    if action == "read_file":
        if not path.is_file():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            lines = path.read_text(errors="replace").splitlines()
            offset = kwargs.get("offset", 0)
            limit = kwargs.get("limit", 200)
            selected = lines[offset:offset + limit]
            return json.dumps({
                "path": str(path),
                "total_lines": len(lines),
                "offset": offset,
                "lines": selected,
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif action == "write_file":
        content = kwargs.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return json.dumps({"path": str(path), "bytes_written": len(content)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif action == "patch":
        old_text = kwargs.get("old_text", "")
        new_text = kwargs.get("new_text", "")
        if not path.is_file():
            return json.dumps({"error": f"File not found: {path}"})
        if not old_text:
            return json.dumps({"error": "old_text is required for patch action"})
        try:
            content = path.read_text(errors="replace")
            if old_text not in content:
                return json.dumps({"error": "old_text not found in file", "old_text_preview": old_text[:100]})
            new_content = content.replace(old_text, new_text, 1)
            path.write_text(new_content)
            return json.dumps({"path": str(path), "status": "patched"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif action == "list_directory":
        if not path.is_dir():
            return json.dumps({"error": f"Directory not found: {path}"})
        recursive = kwargs.get("recursive", False)
        try:
            entries = []
            if recursive:
                for item in sorted(path.rglob("*")):
                    entries.append({
                        "path": str(item.relative_to(path)),
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                    })
            else:
                for item in sorted(path.iterdir()):
                    entries.append({
                        "path": item.name,
                        "type": "dir" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else 0,
                    })
            return json.dumps({"path": str(path), "count": len(entries), "entries": entries[:500]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif action == "search_files":
        pattern = kwargs.get("pattern", "*")
        recursive = kwargs.get("recursive", False)
        try:
            if recursive:
                matches = sorted(path.rglob(pattern))
            else:
                matches = sorted(path.glob(pattern))
            results = [str(m.relative_to(path)) for m in matches if m.is_file()]
            return json.dumps({"path": str(path), "pattern": pattern, "count": len(results), "matches": results[:500]})
        except Exception as e:
            return json.dumps({"error": str(e)})

    return json.dumps({"error": f"Unknown action: {action}"})


def register(registry):
    """Register file operations tools."""
    registry.register(
        name="file_ops",
        func=file_ops_handler,
        schema=FILE_OPS_SCHEMA,
        toolset="file",
    )

