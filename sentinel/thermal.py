"""Thermal monitoring for Jetson — pause experiments when SoC overheats.

Reads the tj-thermal zone (junction temperature) and provides async-friendly
guards for the experiment runtime loop. Stdlib-only, no dependencies.
"""

import asyncio
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Defaults (Jetson Orin Nano)
_DEFAULT_ZONE = "/sys/class/thermal/thermal_zone8"  # tj-thermal
_WARN_C = 70
_CRIT_C = 82
_RESUME_C = 68
_POLL_S = 10
_MAX_WAIT_S = 600


class ThermalGuard:
    """Monitor junction temperature and throttle/pause as needed."""

    def __init__(
        self,
        zone_path: str = _DEFAULT_ZONE,
        warn_c: int = _WARN_C,
        crit_c: int = _CRIT_C,
        resume_c: int = _RESUME_C,
        poll_s: int = _POLL_S,
        max_wait_s: int = _MAX_WAIT_S,
    ):
        self.zone_path = Path(zone_path)
        self.warn_c = warn_c
        self.crit_c = crit_c
        self.resume_c = resume_c
        self.poll_s = poll_s
        self.max_wait_s = max_wait_s
        self._pause_count = 0
        self._total_pause_s = 0.0
        self._max_temp = 0.0
        self._check_count = 0
        self._log_interval = 10  # Log temp at INFO every N checks

    def read_temp(self) -> float | None:
        """Read current temperature in degrees C, or None if unavailable."""
        try:
            raw = (self.zone_path / "temp").read_text().strip()
            return int(raw) / 1000.0
        except (OSError, ValueError):
            return None

    @property
    def stats(self) -> dict:
        return {
            "pause_count": self._pause_count,
            "total_pause_seconds": round(self._total_pause_s, 1),
            "max_temp_c": round(self._max_temp, 1),
            "checks": self._check_count,
        }

    async def check(self, context: str = "") -> float:
        """Check thermal state and return extra delay to add (seconds).

        Args:
            context: Optional label for log messages (e.g. "pre-inference Aria").

        Returns:
            0.0  — temperature OK
            >0   — warm, caller should add this many extra seconds of delay
            After a critical pause, returns 0.0 (already waited)
        """
        temp = self.read_temp()
        if temp is None:
            return 0.0  # Don't block on sensor failure

        # Track high-water mark and check count
        if temp > self._max_temp:
            self._max_temp = temp
        self._check_count += 1

        ctx = f" [{context}]" if context else ""

        # Periodic INFO log so we always have a thermal trace
        if self._check_count % self._log_interval == 0:
            log.info("Thermal:%s %.1f°C (max=%.1f°C, checks=%d)", ctx, temp, self._max_temp, self._check_count)

        if temp >= self.crit_c:
            await self._wait_cooldown(temp)
            return 0.0
        elif temp >= self.warn_c:
            log.warning("Thermal:%s %.1f°C (warm) — adding extra delay", ctx, temp)
            return 10.0
        else:
            log.debug("Thermal:%s %.1f°C (OK)", ctx, temp)
        return 0.0

    async def _wait_cooldown(self, initial_temp: float) -> None:
        """Pause until temperature drops below resume threshold."""
        self._pause_count += 1
        log.warning(
            "THERMAL PAUSE #%d: %.1f°C >= %d°C — waiting for cooldown",
            self._pause_count, initial_temp, self.crit_c,
        )

        start = time.monotonic()
        while True:
            await asyncio.sleep(self.poll_s)
            elapsed = time.monotonic() - start

            temp = self.read_temp()
            if temp is None:
                log.warning("Thermal: sensor read failed, resuming")
                break

            if temp <= self.resume_c:
                log.info(
                    "Thermal: cooled to %.1f°C (<= %d°C) after %.0fs — resuming",
                    temp, self.resume_c, elapsed,
                )
                break

            if elapsed >= self.max_wait_s:
                log.warning(
                    "Thermal: still %.1f°C after %ds max wait — resuming anyway",
                    temp, self.max_wait_s,
                )
                break

            if int(elapsed) % 60 < self.poll_s:
                log.info(
                    "Thermal: waiting... %.1f°C (target <= %d°C, %.0fs elapsed)",
                    temp, self.resume_c, elapsed,
                )

        self._total_pause_s += time.monotonic() - start
