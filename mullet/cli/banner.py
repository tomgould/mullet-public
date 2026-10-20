"""ASCII-art banner for Mullet CLI."""

from __future__ import annotations

import mullet


_LOGO_LINES = (
    r"   __  __       _ _      _   ",
    r"  |  \/  |_   _| | | ___| |_ ",
    r"  | |\/| | | | | | |/ _ \ __|",
    r"  | |  | | |_| | | |  __/ |_ ",
    r"  |_|  |_|\__,_|_|_|\___|\__|",
)


def render_banner(*, version: str | None = None, no_color: bool = False) -> str:
    """ASCII art Mullet logo + version + tagline.

    Args:
        version: Version string to display. Defaults to ``mullet.__version__``.
        no_color: When ``True``, return plain text; otherwise use Rich cyan.

    Returns:
        Multi-line banner string.
    """
    version = version or mullet.__version__
    tagline = "Cloud in the front, Local GPU in the back."
    logo = "\n".join(_LOGO_LINES)
    banner = f"{logo}\n  v{version} — {tagline}"
    if no_color:
        return banner
    try:
        from rich.text import Text
        from rich.console import Console
        text = Text(banner, style="bold cyan")
        console = Console(force_terminal=True, no_color=False)
        with console.capture() as capture:
            console.print(text)
        return capture.get().rstrip("\n")
    except Exception:
        return banner
