#!/usr/bin/env python3
"""
Vector Memory — one-time setup tool.
Run once per bot. Safe to re-run — it will refuse if already configured.
"""
import sys
import os
import argparse
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
SKILL_REFS = Path(__file__).parent
BOT_CONFIG_DIR = Path(os.getenv("HERMES_BOT_CONFIG_DIR", "~/.hermes/bots"))
LOCK_FILE = SKILL_REFS / ".vector_memory_configured"
COLLECTION_INIT = SKILL_REFS / "init_collection.py"


def lock_path(bot_id: str) -> Path:
    return BOT_CONFIG_DIR / bot_id / ".vector_memory.lock"


def eprint(msg: str):
    print(msg, file=sys.stderr)


def fail(msg: str) -> None:
    eprint(f"ERROR: {msg}")
    sys.exit(1)


def check_lock(bot_id: str) -> bool:
    """Return True if already configured (lock exists)."""
    return lock_path(bot_id).exists()


def write_lock(bot_id: str) -> None:
    """Atomically create the lock file so this setup can't run again."""
    p = lock_path(bot_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"vector_memory configured\n"
        f"bot_id={bot_id}\n"
        f"setup_time={os.popen('date -Iseconds').read().strip()}\n"
    )
    eprint(f"[+] Lock written → {p}")


def validate_env() -> tuple[str, str]:
    """Check required env vars, return (host, bot_id)."""
    host = os.getenv("QDRANT_HOST", "").strip()
    bot_id = os.getenv("BOT_ID", "").strip()
    if not host:
        fail("QDRANT_HOST env var is not set. Set it before running setup.")
    if not bot_id:
        fail("BOT_ID env var is not set. Set it before running setup.")
    return host, bot_id


def run_collection_init(bot_id: str) -> None:
    """Run the existing init_collection.py with the correct --bot-id."""
    import subprocess
    eprint(f"[*] Running collection initialisation for bot_id={bot_id} …")
    result = subprocess.run(
        [sys.executable, str(COLLECTION_INIT), "--bot-id", bot_id],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(
            f"init_collection.py failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout}\n"
            f"  stderr: {result.stderr}"
        )
    eprint(f"[+] Collection ready: {result.stdout}")


def check_qdrant_connection(host: str) -> None:
    """Ping Qdrant to verify it's reachable."""
    import urllib.request
    import urllib.error

    try:
        url = f"http://{host}:6333/collections"
        with urllib.request.urlopen(url, timeout=5) as r:
            if r.status != 200:
                fail(f"Qdrant responded with HTTP {r.status}")
    except urllib.error.URLError as e:
        fail(f"Cannot reach Qdrant at http://{host}:6333 — {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time Vector Memory setup for a Hermes bot."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Override the lock and re-run (use when migrating Qdrant hosts).",
    )
    args = parser.parse_args()

    eprint("=" * 60)
    eprint("  Vector Memory — one-time setup")
    eprint("=" * 60)

    host, bot_id = validate_env()
    eprint(f"[*] QDRANT_HOST = {host}")
    eprint(f"[*] BOT_ID      = {bot_id}")

    # ── Idempotency guard ──────────────────────────────────────────────────
    if check_lock(bot_id):
        if args.force:
            eprint("[!] --force given; overriding lock and proceeding …")
        else:
            fail(
                "Already configured — lock file exists.\n"
                "  This bot already ran setup. To re-run anyway (e.g. after a\n"
                "  Qdrant migration), pass --force.\n"
                f"  Lock: {lock_path(bot_id)}"
            )

    # ── Connectivity check ─────────────────────────────────────────────────
    eprint("[*] Checking Qdrant connectivity …")
    check_qdrant_connection(host)
    eprint(f"[+] Qdrant reachable at http://{host}:6333")

    # ── Collection init ────────────────────────────────────────────────────
    run_collection_init(bot_id)

    # ── Write lock ─────────────────────────────────────────────────────────
    write_lock(bot_id)

    eprint("")
    eprint("=" * 60)
    eprint("  ✓ Vector Memory is configured for this bot.")
    eprint("")
    eprint("  Next: activate the plugin in your bot config or run:")
    eprint(f"    hermes memory setup  (then select 'vector-memory')")
    eprint("=" * 60)


if __name__ == "__main__":
    main()
