"""
model_skill_tracker.py -- Phase 4: Model Skill Score Tracking & BMA Weight Learning
=====================================================================================
Tracks per-model forecast accuracy (MAE, CRPS, Brier Score) for each
ICAO station by season. Results feed back into bma_engine.py to replace
the static skill weights with learned ones.

Data stored in SQLite (same DB as portfolio_manager for simplicity).

Tables:
  model_forecasts   -- raw forecast snapshots (model, icao, valid_date, predicted_max_c)
  model_verifications -- actual outcomes matched to forecasts
  model_skill       -- aggregated skill scores per model/icao/season

Workflow (run daily):
  1. At market resolution: record actual daily max/min for each ICAO
  2. Match to existing forecasts within ±6h issue time
  3. Compute MAE, RMSE, Brier Score per model
  4. Update skill scores table → bma_engine reads these on next scan
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "model_skill.db"
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# Seasons: DJF=0, MAM=1, JJA=2, SON=3
def _season(d: date) -> int:
    m = d.month
    if m in (12, 1, 2): return 0   # Winter
    if m in (3, 4, 5):  return 1   # Spring
    if m in (6, 7, 8):  return 2   # Summer
    return 3                         # Fall


@dataclass
class SkillScore:
    """Aggregated skill for one model/icao/season combination."""
    model: str
    icao: str
    season: int
    n_samples: int
    mae_c: float        # Mean Absolute Error in °C
    rmse_c: float       # Root Mean Squared Error in °C
    brier_score: float  # Brier Score (for bucket probability)
    crps: float         # Continuous Ranked Probability Score (approx)
    bias_c: float       # Systematic bias (positive = warm bias)
    skill_weight: float # Normalized weight for BMA (higher = more weight)


class ModelSkillTracker:
    """
    Tracks model forecast accuracy and derives BMA weights.
    Thread-safe SQLite backend.
    """

    # Static baseline weights used when insufficient data
    BASELINE_WEIGHTS = {
        "gefs_mean":  0.30,
        "ecmwf_ens":  0.25,
        "ecmwf":      0.20,
        "gfs":        0.12,
        "hrrr":       0.08,
        "nam":        0.03,
        "nbm":        0.02,
    }

    MIN_SAMPLES_FOR_LEARNED = 15  # require ≥15 samples before using learned weights

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS model_forecasts (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    icao            TEXT NOT NULL,
                    model           TEXT NOT NULL,
                    issue_time      TEXT NOT NULL,
                    valid_date      TEXT NOT NULL,
                    horizon_h       INTEGER NOT NULL,
                    predicted_max_c REAL,
                    predicted_min_c REAL,
                    spread_c        REAL,
                    created_at      TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS actual_observations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    icao        TEXT NOT NULL,
                    obs_date    TEXT NOT NULL UNIQUE,
                    actual_max_c REAL,
                    actual_min_c REAL,
                    source      TEXT DEFAULT 'metar',
                    created_at  TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS forecast_errors (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    icao        TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    obs_date    TEXT NOT NULL,
                    horizon_h   INTEGER NOT NULL,
                    season      INTEGER NOT NULL,
                    error_max_c REAL,
                    error_min_c REAL,
                    abs_error_max_c REAL,
                    abs_error_min_c REAL,
                    sq_error_max_c  REAL,
                    sq_error_min_c  REAL,
                    created_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(icao, model, obs_date, horizon_h)
                );

                CREATE TABLE IF NOT EXISTS model_skill (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    icao        TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    season      INTEGER NOT NULL,
                    n_samples   INTEGER NOT NULL,
                    mae_c       REAL NOT NULL,
                    rmse_c      REAL NOT NULL,
                    bias_c      REAL NOT NULL,
                    skill_weight REAL NOT NULL,
                    updated_at  TEXT DEFAULT (datetime('now')),
                    UNIQUE(icao, model, season)
                );

                CREATE INDEX IF NOT EXISTS idx_forecasts_icao_date
                    ON model_forecasts(icao, valid_date);
                CREATE INDEX IF NOT EXISTS idx_errors_icao_model
                    ON forecast_errors(icao, model, season);
            """)
        logger.debug(f"[SkillTracker] DB initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # 1. Ingest forecast snapshots (called by herbie_fetcher each scan)
    # ------------------------------------------------------------------

    def record_forecast(
        self,
        icao: str,
        model: str,
        issue_time: datetime,
        valid_date: date,
        predicted_max_c: Optional[float],
        predicted_min_c: Optional[float],
        spread_c: Optional[float] = None,
        horizon_h: int = 24,
    ) -> None:
        """Record one model's point forecast for a specific station & date."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_forecasts
                        (icao, model, issue_time, valid_date, horizon_h,
                         predicted_max_c, predicted_min_c, spread_c)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    icao.upper(),
                    model,
                    issue_time.isoformat(),
                    valid_date.isoformat(),
                    horizon_h,
                    predicted_max_c,
                    predicted_min_c,
                    spread_c,
                ))
        except Exception as e:
            logger.warning(f"[SkillTracker] record_forecast error: {e}")

    def record_forecasts_from_herbie(
        self,
        icao: str,
        forecast_daily: dict,
        issue_time: Optional[datetime] = None,
    ) -> int:
        """
        Bulk-record all model forecasts from a herbie_fetcher forecast_daily dict.
        Returns number of records inserted.
        """
        if issue_time is None:
            issue_time = datetime.now(timezone.utc)

        today = date.today()
        count = 0
        model_map = {
            "gefs_mean_max": ("gefs_mean", "max"),
            "gefs_mean_min": ("gefs_mean", "min"),
            "ecmwf_max":     ("ecmwf",     "max"),
            "ecmwf_min":     ("ecmwf",     "min"),
            "gfs_max":       ("gfs",       "max"),
            "gfs_min":       ("gfs",       "min"),
            "hrrr_max":      ("hrrr",      "max"),
            "hrrr_min":      ("hrrr",      "min"),
            "nam_max":       ("nam",        "max"),
            "nam_min":       ("nam",        "min"),
            "nbm_max":       ("nbm",        "max"),
            "nbm_min":       ("nbm",        "min"),
            "ecmwf_ens_mean_max": ("ecmwf_ens", "max"),
            "ecmwf_ens_mean_min": ("ecmwf_ens", "min"),
        }

        from datetime import timedelta

        # Group by model: collect max and min together
        # day_idx=0 → valid_date=today (current day's forecast)
        # day_idx=1 → tomorrow, etc.
        # horizon_h = 24 for day_idx=0 (end-of-day forecast issued now)
        model_forecasts: dict[str, dict] = {}
        for key, (model, temp_type) in model_map.items():
            vals = forecast_daily.get(key) or []
            for day_idx, val in enumerate(vals[:4]):
                if val is None:
                    continue
                valid = today + timedelta(days=day_idx)
                horizon = max(6, day_idx * 24)  # 0h→6h, 1day→24h, 2day→48h
                mkey = (model, valid.isoformat(), horizon)
                if mkey not in model_forecasts:
                    model_forecasts[mkey] = {"max": None, "min": None}
                model_forecasts[mkey][temp_type] = float(val)

        # Also record GEFS spread
        spread_vals = forecast_daily.get("gefs_spread_max") or []

        for (model, valid_str, horizon_h), temps in model_forecasts.items():
            # Add spread if available for GEFS
            spread = None
            if model == "gefs_mean" and spread_vals:
                day_idx = (horizon_h // 24) - 1
                if 0 <= day_idx < len(spread_vals):
                    spread = spread_vals[day_idx]

            self.record_forecast(
                icao=icao,
                model=model,
                issue_time=issue_time,
                valid_date=date.fromisoformat(valid_str),
                predicted_max_c=temps.get("max"),
                predicted_min_c=temps.get("min"),
                spread_c=spread,
                horizon_h=horizon_h,
            )
            count += 1

        logger.debug(f"[SkillTracker] Recorded {count} forecasts for {icao}")
        return count

    # ------------------------------------------------------------------
    # 2. Ingest actual observations (called at market resolution)
    # ------------------------------------------------------------------

    def record_observation(
        self,
        icao: str,
        obs_date: date,
        actual_max_c: Optional[float],
        actual_min_c: Optional[float],
        source: str = "metar",
    ) -> None:
        """Record actual observed daily max/min temperature for a station."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO actual_observations
                        (icao, obs_date, actual_max_c, actual_min_c, source)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    icao.upper(),
                    obs_date.isoformat(),
                    actual_max_c,
                    actual_min_c,
                    source,
                ))
        except Exception as e:
            logger.warning(f"[SkillTracker] record_observation error: {e}")

    # ------------------------------------------------------------------
    # 3. Compute errors (run daily after observations are in)
    # ------------------------------------------------------------------

    def compute_errors_for_date(self, obs_date: date) -> int:
        """
        For each icao/model, match forecast to observation and record error.
        Returns number of error records inserted.
        """
        count = 0
        try:
            with self._get_conn() as conn:
                # Get all observations for this date
                obs_rows = conn.execute("""
                    SELECT icao, actual_max_c, actual_min_c
                    FROM actual_observations
                    WHERE obs_date = ?
                """, (obs_date.isoformat(),)).fetchall()

                season = _season(obs_date)

                for obs in obs_rows:
                    icao = obs["icao"]
                    actual_max = obs["actual_max_c"]
                    actual_min = obs["actual_min_c"]

                    # Match forecasts for this icao/date
                    fc_rows = conn.execute("""
                        SELECT model, horizon_h, predicted_max_c, predicted_min_c
                        FROM model_forecasts
                        WHERE icao = ? AND valid_date = ?
                        ORDER BY horizon_h
                    """, (icao, obs_date.isoformat())).fetchall()

                    for fc in fc_rows:
                        model = fc["model"]
                        horizon = fc["horizon_h"]
                        pred_max = fc["predicted_max_c"]
                        pred_min = fc["predicted_min_c"]

                        err_max = (pred_max - actual_max) if (pred_max is not None and actual_max is not None) else None
                        err_min = (pred_min - actual_min) if (pred_min is not None and actual_min is not None) else None

                        try:
                            conn.execute("""
                                INSERT OR REPLACE INTO forecast_errors
                                    (icao, model, obs_date, horizon_h, season,
                                     error_max_c, error_min_c,
                                     abs_error_max_c, abs_error_min_c,
                                     sq_error_max_c, sq_error_min_c)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                icao, model, obs_date.isoformat(), horizon, season,
                                err_max, err_min,
                                abs(err_max) if err_max is not None else None,
                                abs(err_min) if err_min is not None else None,
                                err_max ** 2 if err_max is not None else None,
                                err_min ** 2 if err_min is not None else None,
                            ))
                            count += 1
                        except Exception as e:
                            logger.debug(f"[SkillTracker] Error insert skip: {e}")

            logger.info(f"[SkillTracker] Computed {count} errors for {obs_date}")
        except Exception as e:
            logger.error(f"[SkillTracker] compute_errors error: {e}")
        return count

    # ------------------------------------------------------------------
    # 4. Update skill scores (run after error computation)
    # ------------------------------------------------------------------

    def update_skill_scores(self, icao: Optional[str] = None) -> dict[str, SkillScore]:
        """
        Recompute skill scores for all icao/model/season combinations.
        Updates model_skill table.
        Returns dict of (icao, model, season) -> SkillScore.
        """
        results: dict[str, SkillScore] = {}

        try:
            with self._get_conn() as conn:
                where = f"WHERE icao = '{icao.upper()}'" if icao else ""
                rows = conn.execute(f"""
                    SELECT
                        icao, model, season,
                        COUNT(*) as n,
                        AVG(abs_error_max_c) as mae,
                        AVG(sq_error_max_c)  as mse,
                        AVG(error_max_c)     as bias
                    FROM forecast_errors
                    {where}
                    GROUP BY icao, model, season
                    HAVING COUNT(*) >= 3
                """).fetchall()

                # Group by icao/season to normalize weights
                icao_season_groups: dict[tuple, list] = {}
                for r in rows:
                    key = (r["icao"], r["season"])
                    icao_season_groups.setdefault(key, []).append(r)

                for (icao_key, season), group in icao_season_groups.items():
                    # Inverse-MAE weighting: better model (lower MAE) gets higher weight
                    maes = [(r["model"], float(r["mae"] or 9.9)) for r in group]
                    inv_maes = {m: 1.0 / max(0.1, mae) for m, mae in maes}
                    total_inv = sum(inv_maes.values())
                    norm_weights = {m: v / total_inv for m, v in inv_maes.items()}

                    for r in group:
                        model = r["model"]
                        n = int(r["n"])
                        mae = float(r["mae"] or 9.9)
                        mse = float(r["mse"] or 99.0)
                        rmse = math.sqrt(mse) if mse >= 0 else 9.9
                        bias = float(r["bias"] or 0.0)
                        skill_w = norm_weights.get(model, 0.0)

                        # Blend with baseline weights (less certain with fewer samples)
                        blend_factor = min(1.0, n / (self.MIN_SAMPLES_FOR_LEARNED * 2))
                        baseline_w = self.BASELINE_WEIGHTS.get(model, 0.05)
                        blended_w = blend_factor * skill_w + (1 - blend_factor) * baseline_w

                        # Renormalize will happen in get_bma_weights
                        ss = SkillScore(
                            model=model,
                            icao=icao_key,
                            season=season,
                            n_samples=n,
                            mae_c=mae,
                            rmse_c=rmse,
                            brier_score=0.0,  # computed separately if needed
                            crps=mae * 0.8,   # approximate CRPS ≈ 0.8 * MAE for normal dist
                            bias_c=bias,
                            skill_weight=blended_w,
                        )
                        results[f"{icao_key}:{model}:{season}"] = ss

                        try:
                            conn.execute("""
                                INSERT OR REPLACE INTO model_skill
                                    (icao, model, season, n_samples, mae_c, rmse_c, bias_c, skill_weight)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (icao_key, model, season, n, mae, rmse, bias, blended_w))
                        except Exception as e:
                            logger.debug(f"[SkillTracker] skill insert error: {e}")

                logger.info(f"[SkillTracker] Updated {len(results)} skill score records")

        except Exception as e:
            logger.error(f"[SkillTracker] update_skill_scores error: {e}")

        return results

    # ------------------------------------------------------------------
    # 5. Get learned BMA weights for bma_engine
    # ------------------------------------------------------------------

    def get_bma_weights(self, icao: str, season: Optional[int] = None) -> dict[str, float]:
        """
        Return normalized BMA weights for an ICAO station.
        Falls back to baseline weights if insufficient data.

        Args:
            icao    : ICAO station code (e.g. 'KORD')
            season  : 0=DJF, 1=MAM, 2=JJA, 3=SON (None = today's season)

        Returns:
            dict of {model_name: weight} normalized to sum=1
        """
        if season is None:
            season = _season(date.today())

        try:
            with self._get_conn() as conn:
                rows = conn.execute("""
                    SELECT model, n_samples, skill_weight
                    FROM model_skill
                    WHERE icao = ? AND season = ?
                    ORDER BY skill_weight DESC
                """, (icao.upper(), season)).fetchall()

                if not rows:
                    logger.debug(f"[SkillTracker] No learned weights for {icao} s={season}, using baseline")
                    return self._normalize(self.BASELINE_WEIGHTS)

                # Check if we have enough data to trust learned weights
                total_n = sum(int(r["n_samples"]) for r in rows)
                if total_n < self.MIN_SAMPLES_FOR_LEARNED:
                    logger.debug(f"[SkillTracker] Insufficient samples (n={total_n}) for {icao}, blending")

                raw_weights = {r["model"]: float(r["skill_weight"]) for r in rows}

                # Fill in missing models with baseline (low weight)
                for model, w in self.BASELINE_WEIGHTS.items():
                    if model not in raw_weights:
                        raw_weights[model] = w * 0.3  # Down-weighted missing models

                return self._normalize(raw_weights)

        except Exception as e:
            logger.error(f"[SkillTracker] get_bma_weights error: {e}")
            return self._normalize(self.BASELINE_WEIGHTS)

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            n = len(weights)
            return {k: 1.0 / n for k in weights}
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # 6. Get model bias correction for a specific model/icao/season
    # ------------------------------------------------------------------

    def get_bias_correction(
        self,
        model: str,
        icao: str,
        season: Optional[int] = None,
    ) -> float:
        """
        Return bias correction for a model (subtract this from forecast).
        E.g., if HRRR has +0.8°C warm bias, returns +0.8 → caller subtracts 0.8°C.
        Returns 0.0 if insufficient data.
        """
        if season is None:
            season = _season(date.today())

        try:
            with self._get_conn() as conn:
                row = conn.execute("""
                    SELECT bias_c, n_samples FROM model_skill
                    WHERE icao = ? AND model = ? AND season = ?
                """, (icao.upper(), model, season)).fetchone()

                if row and int(row["n_samples"]) >= self.MIN_SAMPLES_FOR_LEARNED:
                    return float(row["bias_c"])

        except Exception as e:
            logger.debug(f"[SkillTracker] bias query error: {e}")

        return 0.0

    # ------------------------------------------------------------------
    # 7. Summary report (for Telegram /skill command)
    # ------------------------------------------------------------------

    def get_skill_report(self, icao: Optional[str] = None) -> str:
        """Return a formatted skill score report for monitoring."""
        season_names = {0: "DJF(Winter)", 1: "MAM(Spring)", 2: "JJA(Summer)", 3: "SON(Fall)"}
        try:
            with self._get_conn() as conn:
                where = f"WHERE icao = '{icao.upper()}'" if icao else ""
                rows = conn.execute(f"""
                    SELECT icao, model, season, n_samples, mae_c, bias_c, skill_weight
                    FROM model_skill
                    {where}
                    ORDER BY icao, season, skill_weight DESC
                """).fetchall()

                if not rows:
                    return "📊 No skill data yet (need more resolved trades with observations)."

                lines = ["📊 *MODEL SKILL SCORES*\n"]
                current_key = None
                for r in rows:
                    key = f"{r['icao']}-{season_names.get(r['season'], r['season'])}"
                    if key != current_key:
                        lines.append(f"\n*{key}* (n={r['n_samples']})")
                        current_key = key
                    bias_str = f"bias={r['bias_c']:+.2f}°C"
                    lines.append(
                        f"  {r['model']:12s} MAE={r['mae_c']:.2f}°C {bias_str} w={r['skill_weight']:.3f}"
                    )

                return "\n".join(lines)

        except Exception as e:
            return f"Error generating skill report: {e}"

    # ------------------------------------------------------------------
    # 8. Integration: record from calibration resolution events
    # ------------------------------------------------------------------

    def process_resolution(
        self,
        icao: str,
        obs_date: date,
        actual_max_c: Optional[float],
        actual_min_c: Optional[float],
        forecast_daily: Optional[dict] = None,
    ) -> None:
        """
        Full pipeline: record observation + compute errors + update skills.
        Called when a market resolves and we have the actual temperature.
        """
        self.record_observation(icao, obs_date, actual_max_c, actual_min_c)

        if forecast_daily:
            # Store today's forecast for future error computation
            self.record_forecasts_from_herbie(icao, forecast_daily)

        errors_added = self.compute_errors_for_date(obs_date)
        if errors_added > 0:
            self.update_skill_scores(icao=icao)
            logger.info(f"[SkillTracker] Processed resolution for {icao} on {obs_date}: {errors_added} errors")


# Singleton
model_skill_tracker = ModelSkillTracker()
