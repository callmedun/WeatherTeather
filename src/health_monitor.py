"""
health_monitor.py -- Phase 6: Circuit Breaker & Health Monitoring
===================================================================
Implements:
  1. Circuit Breaker  — auto-pause trading after N consecutive API failures
  2. Health Monitor   — tracks uptime, error rates, API latencies
  3. Retry Decorator  — exponential backoff for all external API calls
  4. GRIB Cleanup     — removes stale Herbie GRIB files
  5. Status Report    — unified bot health dashboard for Telegram

Circuit Breaker states:
  CLOSED  → normal operation
  OPEN    → paused due to failures (auto-recover after cooldown)
  HALF    → test mode after cooldown (one request allowed through)

Usage:
    from src.health_monitor import circuit_breaker, health_monitor

    # Protect an API call:
    @circuit_breaker.guard("gamma_api")
    async def fetch_markets():
        ...

    # Retry with backoff:
    from src.health_monitor import retry_with_backoff
    result = await retry_with_backoff(fetch_markets, max_attempts=3)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CBState(Enum):
    CLOSED   = "CLOSED"    # Normal
    OPEN     = "OPEN"      # Tripped — no requests
    HALF     = "HALF_OPEN" # Recovery test


@dataclass
class CircuitStats:
    """Per-circuit statistics."""
    name: str
    state: CBState = CBState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None
    tripped_at: Optional[float] = None
    total_calls: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    _latency_samples: list[float] = field(default_factory=list)

    def record_latency(self, ms: float) -> None:
        self._latency_samples.append(ms)
        if len(self._latency_samples) > 50:
            self._latency_samples = self._latency_samples[-50:]
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)


class CircuitBreaker:
    """
    Multi-circuit breaker with configurable thresholds.

    Each named circuit (e.g. "gamma_api", "weather_api") tracks
    failures independently. When a circuit trips, requests are blocked
    until the cooldown expires and one test request succeeds.
    """

    DEFAULT_FAILURE_THRESHOLD  = 3    # trips after N consecutive failures
    DEFAULT_COOLDOWN_SECONDS   = 120  # wait before half-open
    DEFAULT_SUCCESS_THRESHOLD  = 2    # successes needed to close after half-open

    def __init__(self):
        self._circuits: dict[str, CircuitStats] = {}
        self._lock = asyncio.Lock()
        self._global_pause_callback: Optional[Callable] = None

    def register_pause_callback(self, cb: Callable) -> None:
        """Register a callback invoked when any circuit trips (e.g. pause trading)."""
        self._global_pause_callback = cb

    def _get_or_create(self, name: str) -> CircuitStats:
        if name not in self._circuits:
            self._circuits[name] = CircuitStats(name=name)
        return self._circuits[name]

    def is_open(self, name: str) -> bool:
        """Returns True if circuit is OPEN (blocking requests)."""
        stats = self._circuits.get(name)
        if not stats or stats.state == CBState.CLOSED:
            return False
        if stats.state == CBState.HALF:
            return False
        # Check if cooldown expired
        if stats.tripped_at and (time.time() - stats.tripped_at) >= self.DEFAULT_COOLDOWN_SECONDS:
            stats.state = CBState.HALF
            logger.info(f"[CB] Circuit '{name}' → HALF_OPEN (cooldown expired)")
            return False
        return True

    async def record_success(self, name: str, latency_ms: float = 0.0) -> None:
        async with self._lock:
            stats = self._get_or_create(name)
            stats.failures = 0
            stats.successes += 1
            stats.total_calls += 1
            stats.last_success_at = time.time()
            stats.record_latency(latency_ms)
            if stats.state == CBState.HALF:
                if stats.successes >= self.DEFAULT_SUCCESS_THRESHOLD:
                    stats.state = CBState.CLOSED
                    stats.tripped_at = None
                    logger.info(f"[CB] Circuit '{name}' → CLOSED (recovered)")

    async def record_failure(self, name: str, error: str = "") -> None:
        async with self._lock:
            stats = self._get_or_create(name)
            stats.failures += 1
            stats.total_errors += 1
            stats.total_calls += 1
            stats.last_failure_at = time.time()
            if stats.failures >= self.DEFAULT_FAILURE_THRESHOLD and stats.state != CBState.OPEN:
                stats.state = CBState.OPEN
                stats.tripped_at = time.time()
                logger.error(
                    f"[CB] Circuit '{name}' TRIPPED after {stats.failures} failures. "
                    f"Last error: {error}"
                )
                if self._global_pause_callback:
                    try:
                        await self._global_pause_callback(name, error)
                    except Exception as cb_e:
                        logger.debug(f"[CB] Pause callback error: {cb_e}")

    def guard(self, circuit_name: str):
        """Decorator that wraps an async function with circuit breaker protection."""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                if self.is_open(circuit_name):
                    raise CircuitOpenError(f"Circuit '{circuit_name}' is OPEN — request blocked")
                t0 = time.time()
                try:
                    result = await func(*args, **kwargs)
                    latency = (time.time() - t0) * 1000
                    await self.record_success(circuit_name, latency)
                    return result
                except CircuitOpenError:
                    raise
                except Exception as e:
                    await self.record_failure(circuit_name, str(e)[:100])
                    raise
            return wrapper
        return decorator

    def get_status_text(self) -> str:
        """Returns formatted circuit status for Telegram."""
        if not self._circuits:
            return "  (no circuits registered yet)"
        lines = []
        for name, stats in sorted(self._circuits.items()):
            icon = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}.get(stats.state.value, "⚪")
            cooldown_str = ""
            if stats.state == CBState.OPEN and stats.tripped_at:
                remaining = max(0, self.DEFAULT_COOLDOWN_SECONDS - (time.time() - stats.tripped_at))
                cooldown_str = f" (recover in {remaining:.0f}s)"
            lines.append(
                f"  {icon} {name}: {stats.state.value}{cooldown_str} | "
                f"errors={stats.total_errors}/{stats.total_calls} | "
                f"latency={stats.avg_latency_ms:.0f}ms"
            )
        return "\n".join(lines)

    def reset(self, name: Optional[str] = None) -> None:
        """Manually reset a circuit (or all circuits)."""
        if name:
            if name in self._circuits:
                self._circuits[name] = CircuitStats(name=name)
        else:
            self._circuits.clear()


class CircuitOpenError(Exception):
    """Raised when a guarded function is called with an OPEN circuit."""
    pass


# ---------------------------------------------------------------------------
# Retry with Exponential Backoff
# ---------------------------------------------------------------------------

async def retry_with_backoff(
    func: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 30.0,
    circuit_name: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Call an async function with exponential backoff retry.
    Logs each failure and raises the last exception after max_attempts.

    Args:
        func          : async callable
        max_attempts  : total attempts before giving up
        base_delay    : initial delay in seconds
        max_delay     : maximum delay cap in seconds
        circuit_name  : if provided, records failures to circuit breaker
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)
        except CircuitOpenError:
            raise  # don't retry when circuit is open
        except Exception as e:
            last_exc = e
            if attempt == max_attempts:
                if circuit_name:
                    await circuit_breaker.record_failure(circuit_name, str(e)[:100])
                logger.error(f"[Retry] {func.__name__} failed after {max_attempts} attempts: {e}")
                raise

            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            # Add jitter ±20%
            jitter = delay * 0.2 * (0.5 - hash(str(e)) % 100 / 100)
            delay = max(0.5, delay + jitter)
            logger.warning(
                f"[Retry] {func.__name__} attempt {attempt}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)

    raise last_exc  # type: ignore


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------

@dataclass
class BotHealthSnapshot:
    """Point-in-time health snapshot."""
    uptime_seconds: float
    scan_count: int
    last_scan_at: Optional[str]
    last_monitor_at: Optional[str]
    consecutive_errors: int
    circuit_status: str
    paper_signals_today: int
    skill_records: int
    open_positions: int
    is_paused: bool
    mode: str


class HealthMonitor:
    """
    Tracks bot operational health metrics over time.
    Provides unified status report combining all Phase 2-5 data.
    """

    def __init__(self):
        self._start_time = time.time()
        self._scan_count = 0
        self._monitor_count = 0
        self._error_log: list[dict] = []  # last 50 errors

    def record_scan(self) -> None:
        self._scan_count += 1

    def record_error(self, source: str, message: str) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "msg": message[:200],
        }
        self._error_log.append(entry)
        if len(self._error_log) > 50:
            self._error_log = self._error_log[-50:]

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def _fmt_uptime(self) -> str:
        s = int(self.uptime_seconds)
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m}m {sec}s"

    def get_full_status(
        self,
        scheduler=None,
        paper_trader_ref=None,
        skill_tracker_ref=None,
    ) -> str:
        """
        Returns a comprehensive Telegram-formatted status message.
        Aggregates data from all phases.
        """
        from config.settings import config

        mode = "🧪 DRY RUN" if config.dry_run else "💵 LIVE"
        paused = "⏸ ПАУЗA" if config.is_paused else "▶️ АКТИВЕН"

        # Scheduler times
        last_scan = "—"
        last_mon  = "—"
        if scheduler:
            if scheduler.last_scan_time:
                last_scan = scheduler.last_scan_time.strftime("%H:%M:%S UTC")
            if scheduler.last_monitor_time:
                last_mon = scheduler.last_monitor_time.strftime("%H:%M:%S UTC")

        # Circuit breaker
        cb_text = circuit_breaker.get_status_text()

        # Phase 4: Skill tracker
        skill_text = "—"
        if skill_tracker_ref:
            try:
                import sqlite3
                conn = sqlite3.connect(skill_tracker_ref.db_path, timeout=3)
                skill_n = conn.execute("SELECT COUNT(*) FROM model_skill").fetchone()[0]
                forecasts_n = conn.execute("SELECT COUNT(*) FROM model_forecasts").fetchone()[0]
                conn.close()
                skill_text = f"{skill_n} skill records | {forecasts_n} forecasts logged"
            except Exception:
                skill_text = "available (no data yet)"

        # Phase 5: Paper trader
        paper_text = "—"
        if paper_trader_ref:
            try:
                import sqlite3
                conn = sqlite3.connect(paper_trader_ref.db_path, timeout=3)
                p_total = conn.execute("SELECT COUNT(*) FROM paper_signals").fetchone()[0]
                p_open  = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE status='OPEN'").fetchone()[0]
                p_wins  = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE status='WIN'").fetchone()[0]
                p_loss  = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE status='LOSS'").fetchone()[0]
                p_pnl   = conn.execute("SELECT COALESCE(SUM(pnl),0) FROM paper_signals").fetchone()[0]
                conn.close()
                wr = p_wins / (p_wins + p_loss) if (p_wins + p_loss) > 0 else 0.0
                paper_text = (
                    f"{p_total} signals | {p_open} open | "
                    f"{p_wins}W/{p_loss}L ({wr:.0%}) | PnL=${p_pnl:+.2f}"
                )
            except Exception:
                paper_text = "available (no signals yet)"

        # Recent errors
        recent_errors = self._error_log[-5:] if self._error_log else []
        error_text = ""
        if recent_errors:
            error_lines = [f"  ⚠️ {e['source']}: {e['msg'][:80]}" for e in recent_errors]
            error_text = "\n🔴 Последние ошибки:\n" + "\n".join(error_lines)

        return (
            f"🤖 *POLYMARKET WEATHER BOT v2* — Phase 6\n\n"
            f"📊 *Режим:* {mode} | {paused}\n"
            f"⏱ *Аптайм:* {self._fmt_uptime()}\n"
            f"🔄 *Сканов:* {self._scan_count}\n\n"
            f"⏱ *Последний скан:* {last_scan}\n"
            f"⏱ *Последний монитор:* {last_mon}\n\n"
            f"⚡ *Circuit Breakers:*\n{cb_text}\n\n"
            f"🧠 *Phase 4 Skill Tracker:* {skill_text}\n"
            f"📋 *Phase 5 Paper Trader:* {paper_text}"
            f"{error_text}"
        )

    def get_recent_errors(self, n: int = 10) -> str:
        if not self._error_log:
            return "✅ Нет зафиксированных ошибок."
        lines = [f"🔴 Последние {n} ошибок:\n"]
        for e in self._error_log[-n:]:
            lines.append(f"  [{e['ts'][:16]}] {e['source']}: {e['msg'][:100]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# GRIB File Cleanup
# ---------------------------------------------------------------------------

class GribCleaner:
    """
    Removes stale Herbie GRIB files older than max_age_days.
    Runs as a daily scheduled task.
    """

    DEFAULT_MAX_AGE_DAYS = 7
    DEFAULT_GRIB_DIRS = [
        "~/.cache/herbie",
        "~/herbie",
        "/tmp/herbie",
        "data/grib",
    ]

    def __init__(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS):
        self.max_age_days = max_age_days
        self._total_freed_mb = 0.0

    def cleanup(self, extra_dirs: Optional[list[str]] = None) -> dict:
        """
        Remove GRIB files older than max_age_days.
        Returns stats dict with files_removed, mb_freed.
        """
        dirs_to_check = self.DEFAULT_GRIB_DIRS[:]
        if extra_dirs:
            dirs_to_check.extend(extra_dirs)

        files_removed = 0
        mb_freed = 0.0
        max_age_sec = self.max_age_days * 86400
        now = time.time()

        for dir_str in dirs_to_check:
            dir_path = Path(os.path.expanduser(dir_str))
            if not dir_path.exists():
                continue
            try:
                for f in dir_path.rglob("*.grib2"):
                    try:
                        age = now - f.stat().st_mtime
                        if age > max_age_sec:
                            size_mb = f.stat().st_size / (1024 * 1024)
                            f.unlink()
                            files_removed += 1
                            mb_freed += size_mb
                    except Exception as e:
                        logger.debug(f"[GribCleaner] Could not remove {f}: {e}")

                # Also remove empty subdirs
                for d in sorted(dir_path.rglob("*"), reverse=True):
                    if d.is_dir():
                        try:
                            d.rmdir()  # only removes if empty
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"[GribCleaner] Error scanning {dir_path}: {e}")

        self._total_freed_mb += mb_freed
        if files_removed > 0:
            logger.info(
                f"[GribCleaner] Removed {files_removed} GRIB files, "
                f"freed {mb_freed:.1f}MB (total freed: {self._total_freed_mb:.1f}MB)"
            )
        return {"files_removed": files_removed, "mb_freed": mb_freed}

    def get_stats(self) -> str:
        return f"🗑️ GRIB Cleaner: {self._total_freed_mb:.1f}MB freed total"


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

circuit_breaker = CircuitBreaker()
health_monitor  = HealthMonitor()
grib_cleaner    = GribCleaner()
