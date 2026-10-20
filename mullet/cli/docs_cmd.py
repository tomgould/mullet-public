"""CLI commands for library pool documentation."""

import json
import sys
import urllib.request
from datetime import datetime
from typing import Callable

import click

from mullet.knowledge.library_pool import LibraryPoolRegistry


@click.group()
def docs():
    """Manage library pool documentation."""
    pass


@docs.command(name="list")
def list_pools():
    """List all registered library pools."""
    registry = LibraryPoolRegistry()
    pools = registry.list()
    if not pools:
        click.echo("No library pools registered.")
        return
    click.echo(f"{'Name':<20} {'Language':<12} {'Version':<12} {'Last Synced'}")
    click.echo("-" * 60)
    for pool in pools:
        synced = pool.last_synced.strftime("%Y-%m-%d %H:%M") if pool.last_synced else "never"
        click.echo(f"{pool.name:<20} {pool.language:<12} {pool.version:<12} {synced}")


@docs.command(name="sync")
@click.option("--dry-run", is_flag=True, help="Show what would change without updating.")
def sync_pools(dry_run: bool):
    """Sync library pool versions against upstream registries.

    Checks PyPI, npm, and crates.io for newer versions of registered
    libraries and updates the local registry when a drift is found.
    """
    registry = LibraryPoolRegistry()
    pools = registry.list()
    if not pools:
        click.echo("No library pools registered.")
        return

    updated = 0
    current = 0
    failed = 0

    for pool in pools:
        fetcher = _resolve_fetcher(pool.source_url, pool.language)
        if fetcher is None:
            click.echo(f"  · {pool.name:<20} unknown registry ({pool.source_url})")
            continue

        latest = fetcher(pool.name)
        if latest is None:
            click.echo(f"  ! {pool.name:<20} could not check version")
            failed += 1
            continue

        if latest == pool.version:
            click.echo(f"  ✓ {pool.name:<20} {pool.version} (up to date)")
            current += 1
        else:
            click.echo(f"  ↑ {pool.name:<20} {pool.version} → {latest}")
            updated += 1
            if not dry_run:
                pool.version = latest
                pool.last_synced = datetime.utcnow()
                registry.register(pool)

    click.echo()
    if dry_run:
        click.echo(f"Dry-run: {current} up to date, {updated} would update, {failed} failed.")
    else:
        click.echo(f"Sync complete: {current} up to date, {updated} updated, {failed} failed.")


@docs.command(name="remove")
@click.argument("name")
def remove_pool(name: str):
    """Remove a registered library pool."""
    registry = LibraryPoolRegistry()
    if registry.get(name) is None:
        click.echo(f"Library pool '{name}' not found.", err=True)
        sys.exit(1)
    registry.unregister(name)
    click.echo(f"Removed library pool '{name}'.")


def _resolve_fetcher(source_url: str, language: str) -> Callable[[str], str | None] | None:
    """Return a version-fetcher for the given source URL or language hint."""
    url = source_url.lower()
    if "pypi" in url or language.lower() == "python":
        return _fetch_pypi_latest
    if "npm" in url or language.lower() in ("javascript", "typescript", "js", "ts"):
        return _fetch_npm_latest
    if "crates" in url or language.lower() == "rust":
        return _fetch_crates_latest
    return None


def _fetch_pypi_latest(name: str) -> str | None:
    try:
        req = urllib.request.Request(
            f"https://pypi.org/pypi/{name}/json",
            headers={"Accept": "application/json", "User-Agent": "mullet-docs-sync/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("info", {}).get("version")
    except Exception:
        return None


def _fetch_npm_latest(name: str) -> str | None:
    try:
        req = urllib.request.Request(
            f"https://registry.npmjs.org/{name}/latest",
            headers={"Accept": "application/json", "User-Agent": "mullet-docs-sync/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("version")
    except Exception:
        return None


def _fetch_crates_latest(name: str) -> str | None:
    try:
        req = urllib.request.Request(
            f"https://crates.io/api/v1/crates/{name}",
            headers={"Accept": "application/json", "User-Agent": "mullet-docs-sync/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("crate", {}).get("max_version")
    except Exception:
        return None
