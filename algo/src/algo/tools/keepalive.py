"""
keepalive.py
------------
Background keep-alive thread that pings LM Studio's /v1/models endpoint
every 60 seconds to prevent model unloading during long tool operations
(backtest, WFO, indicator calculation, etc.).

Usage:
    from algo.tools.keepalive import start_keepalive, stop_keepalive

    start_keepalive()       # Call once at pipeline startup
    stop_keepalive()        # Call at shutdown (optional, daemon thread auto-dies)
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
import urllib.error


_PING_INTERVAL = 60  # seconds between pings
_stop_event = threading.Event()
_thread: threading.Thread | None = None


def _keepalive_loop() -> None:
    """Runs in a daemon thread; sends GET /v1/models every _PING_INTERVAL seconds."""
    base_url = os.getenv("OPENAI_API_BASE", os.getenv("LM_STUDIO_BASE_URL", "http://192.168.18.10:1234/v1"))
    # Ensure we hit the /models endpoint
    if base_url.endswith("/v1"):
        url = base_url + "/models"
    elif base_url.endswith("/v1/"):
        url = base_url + "models"
    else:
        url = base_url.rstrip("/") + "/v1/models"

    while not _stop_event.is_set():
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                _ = resp.read()
        except Exception:
            # Swallow all errors — this is best-effort
            pass
        # Wait for the interval, but wake up quickly if stop is requested
        _stop_event.wait(timeout=_PING_INTERVAL)


def start_keepalive() -> None:
    """Start the keep-alive background thread (idempotent)."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return  # Already running
    _stop_event.clear()
    _thread = threading.Thread(target=_keepalive_loop, daemon=True, name="lm-studio-keepalive")
    _thread.start()
    print("[keepalive] Started LM Studio keep-alive ping (every 60s)")


def stop_keepalive() -> None:
    """Signal the keep-alive thread to stop."""
    global _thread
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None
    print("[keepalive] Stopped LM Studio keep-alive ping")
