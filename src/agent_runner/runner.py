"""Research loop runner.

Implements the AutoResearch keep-or-revert loop:

  1. Load registered hypotheses from strategies/hypotheses/
  2. For each hypothesis with status < validated_candidate:
     a. Run backtest over train window
     b. Score result (composite_score)
     c. Check gates
     d. If passes: update manifest status → validated_candidate
     e. If fails:  update manifest status → quarantined (but keep the record)
  3. Write a research report to reports/

The agent may call run_once() after creating a new hypothesis file.
This runner is in the agent's allow_write list for reports/ only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
import hashlib
import importlib
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.backtester.engine import run_backtest
from src.backtester.walk_forward import walk_forward_splits
from src.evaluator.scoring import check_gates, score
from src.hypothesis_engine.registry import load_all
from src.types import EvalResult, HypothesisStatus

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path("reports")
_FEATURES_DIR = Path("data/features")
_RISK_CONFIG_PATH = Path("configs/risk.yaml")
_COSTS_CONFIG_PATH = Path("configs/costs.yaml")
_UNIVERSE_CONFIG_PATH = Path("configs/universe.yaml")
_DEFAULT_PHASE_TIMEOUT_SECONDS = 900


def _instantiate_hypothesis(module_name: str, class_name: str) -> Any:
    """Instantiate hypothesis class in worker process from module/class spec."""
    if "." not in module_name:
        hypotheses_path = str(Path("strategies/hypotheses").resolve())
        if hypotheses_path not in sys.path:
            sys.path.append(hypotheses_path)

    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)
    return klass()


def _prepare_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a stable, event-time-ordered frame with contiguous index."""
    if frame.empty:
        return frame
    if "event_time" in frame.columns and not frame["event_time"].is_monotonic_increasing:
        frame = frame.sort_values("event_time", kind="mergesort")
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.start != 0:
        frame = frame.reset_index(drop=True)
    return frame


def _load_symbol_frame(path: Path) -> pd.DataFrame:
    """Load and normalize one symbol parquet frame."""
    frame = pd.read_parquet(path)
    return _prepare_symbol_frame(frame)


def _run_symbol_backtest_task(
    symbol: str,
    frame_path: Path,
    module_name: str,
    class_name: str,
) -> tuple[str, EvalResult | None, str | None]:
    """Run one symbol-level full backtest task in a worker process."""
    try:
        symbol_df = _load_symbol_frame(frame_path)
        if symbol_df.empty:
            return symbol, None, None
        hypothesis = _instantiate_hypothesis(module_name, class_name)
        return symbol, run_backtest(symbol_df, hypothesis), None
    except Exception as exc:  # noqa: BLE001
        return symbol, None, str(exc)


def _run_symbol_walk_forward_task(
    symbol: str,
    frame_path: Path,
    module_name: str,
    class_name: str,
    config: dict[str, int],
) -> tuple[str, list[float], str | None]:
    """Run all walk-forward folds for one symbol in a worker process."""
    try:
        symbol_df = _load_symbol_frame(frame_path)
        if symbol_df.empty:
            return symbol, [], None

        hypothesis = _instantiate_hypothesis(module_name, class_name)
        net_pnls: list[float] = []
        for _train_df, _val_df, test_df in walk_forward_splits(
            symbol_df,
            train_days=config["train_days"],
            validate_days=config["validate_days"],
            test_days=config["test_days"],
        ):
            if test_df.empty:
                continue
            net_pnls.append(run_backtest(test_df, hypothesis).net_pnl)

        return symbol, net_pnls, None
    except Exception as exc:  # noqa: BLE001
        return symbol, [], str(exc)


class ResearchRunner:
    """Executes one research cycle: load → backtest → gate → keep/revert.

    Parameters
    ----------
    features_dir:
        Directory containing precomputed <symbol>_5m.parquet feature files.
    reports_dir:
        Directory to write per-run JSON reports.
    hypotheses_dir:
        Directory to scan for hypothesis .py files.
    """

    def __init__(
        self,
        features_dir: Path = _FEATURES_DIR,
        reports_dir: Path = _REPORTS_DIR,
        hypotheses_dir: Path = Path("strategies/hypotheses"),
        risk_config_path: Path = _RISK_CONFIG_PATH,
        workers: int | None = None,
    ) -> None:
        self.features_dir = features_dir
        self.reports_dir = reports_dir
        self.hypotheses_dir = hypotheses_dir
        self.risk_config_path = risk_config_path
        if workers is None:
            env_workers = os.getenv("AUTORESEARCH_WORKERS")
            self.workers = max(1, int(env_workers)) if env_workers else 1
        else:
            self.workers = max(1, int(workers))

        env_timeout = os.getenv("AUTORESEARCH_PHASE_TIMEOUT_SECONDS")
        self.phase_timeout_seconds = (
            max(60, int(env_timeout)) if env_timeout else _DEFAULT_PHASE_TIMEOUT_SECONDS
        )
        self._mp_context = mp.get_context("spawn")

    def run_once(self) -> dict[str, Any]:
        """Execute one research cycle.

        Returns a summary dict describing outcomes for all evaluated hypotheses.
        """
        run_start = time.perf_counter()
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        run_at = datetime.now(timezone.utc)
        timings: dict[str, Any] = {
            "load_features_seconds": 0.0,
            "hypotheses": {},
            "report_write_seconds": 0.0,
            "total_seconds": 0.0,
            "workers": self.workers,
        }

        registry = load_all(self.hypotheses_dir)
        if not registry:
            logger.warning("No hypotheses found in %s", self.hypotheses_dir)
            return {"hypotheses_evaluated": 0}

        # Load all available feature files
        load_start = time.perf_counter()
        feature_paths = self._discover_feature_paths()
        feature_frames = self._load_feature_frames(feature_paths)
        timings["load_features_seconds"] = round(time.perf_counter() - load_start, 4)
        if not feature_frames:
            logger.warning("No feature data found in %s; run data_ingest first", self.features_dir)
            return {"hypotheses_evaluated": 0, "warning": "no_feature_data"}

        wf_config = self._load_walk_forward_config()
        results = {}
        for hyp_id, hyp in registry.items():
            logger.info("Evaluating %s ...", hyp_id)
            hyp_start = time.perf_counter()
            full_backtest_seconds = 0.0
            walk_forward_seconds = 0.0
            symbol_timing: dict[str, float] = {}
            hyp_module = hyp.__class__.__module__
            hyp_class = hyp.__class__.__name__
            try:
                symbol_results, symbol_timing, full_backtest_seconds = self._run_symbol_backtests(
                    feature_frames,
                    feature_paths,
                    hyp,
                    hyp_module,
                    hyp_class,
                )

                if not symbol_results:
                    raise ValueError("no_symbol_results")

                raw_result = self._aggregate_symbol_results(hyp, symbol_results)
                scored_result = score(raw_result)
                gate = check_gates(scored_result)
                wf_start = time.perf_counter()
                wf_summary = self._run_walk_forward(
                    feature_frames,
                    feature_paths,
                    hyp,
                    wf_config,
                    hyp_module,
                    hyp_class,
                )
                walk_forward_seconds = time.perf_counter() - wf_start

                failed_gates = list(gate.failed_gates)
                if not wf_summary["passed"]:
                    if wf_summary["folds"] == 0:
                        failed_gates.append("walk_forward: insufficient_folds")
                    else:
                        if wf_summary["positive_fold_ratio"] < wf_summary["min_positive_fold_ratio"]:
                            failed_gates.append(
                                "walk_forward_positive_fold_ratio: "
                                f"{wf_summary['positive_fold_ratio']:.3f} < "
                                f"{wf_summary['min_positive_fold_ratio']:.3f}"
                            )
                        if wf_summary["median_oos_net_pnl"] <= wf_summary["min_median_oos_net_pnl"]:
                            failed_gates.append(
                                "walk_forward_median_oos_net_pnl: "
                                f"{wf_summary['median_oos_net_pnl']:.3f} <= "
                                f"{wf_summary['min_median_oos_net_pnl']:.3f}"
                            )

                outcome = {
                    "hypothesis_id": hyp_id,
                    "n_trades": scored_result.total_trades,
                    "net_pnl": scored_result.net_pnl,
                    "composite_score": scored_result.composite_score,
                    "gate_passed": gate.passed and wf_summary["passed"],
                    "failed_gates": failed_gates,
                    "walk_forward": wf_summary,
                    "recommended_status": (
                        HypothesisStatus.VALIDATED_CANDIDATE.value if (gate.passed and wf_summary["passed"])
                        else HypothesisStatus.QUARANTINED.value
                    ),
                }
                results[hyp_id] = outcome
                timings["hypotheses"][hyp_id] = {
                    "full_backtest_seconds": round(full_backtest_seconds, 4),
                    "walk_forward_seconds": round(walk_forward_seconds, 4),
                    "total_seconds": round(time.perf_counter() - hyp_start, 4),
                    "symbols": dict(sorted(symbol_timing.items())),
                }
                logger.info(
                    "%s: score=%.3f gates=%s trades=%d",
                    hyp_id, scored_result.composite_score,
                    "PASS" if gate.passed else "FAIL",
                    scored_result.total_trades,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Error evaluating %s: %s", hyp_id, exc)
                results[hyp_id] = {"hypothesis_id": hyp_id, "error": str(exc)}
                timings["hypotheses"][hyp_id] = {
                    "full_backtest_seconds": round(full_backtest_seconds, 4),
                    "walk_forward_seconds": round(walk_forward_seconds, 4),
                    "total_seconds": round(time.perf_counter() - hyp_start, 4),
                    "symbols": symbol_timing,
                    "error": str(exc),
                }

        report = {
            "run_at": run_at.isoformat(),
            "hypotheses_evaluated": len(results),
            "provenance": self._build_provenance(feature_frames, wf_config),
            "results": {k: results[k] for k in sorted(results)},
            "timings": timings,
        }
        write_start = time.perf_counter()
        self._save_report(report, run_at)
        timings["report_write_seconds"] = round(time.perf_counter() - write_start, 4)
        timings["total_seconds"] = round(time.perf_counter() - run_start, 4)
        # Persist finalized timing fields (set after first write).
        self._save_report(report, run_at)
        return report

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _discover_feature_paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for path in sorted(self.features_dir.glob("*_5m.parquet")):
            symbol = path.name.replace("_5m.parquet", "")
            paths[symbol] = path
        return paths

    def _load_feature_frames(self, feature_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
        frames = {}
        for symbol, path in sorted(feature_paths.items()):
            try:
                frames[symbol] = _load_symbol_frame(path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load features for %s: %s", symbol, exc)
        return frames

    def _save_report(self, report: dict, run_at: datetime | None = None) -> None:
        ts = (run_at or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        path = self.reports_dir / f"research_run_{ts}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.info("Report saved to %s", path)

    def _run_walk_forward(
        self,
        feature_frames: dict[str, pd.DataFrame],
        feature_paths: dict[str, Path],
        hypothesis: Any,
        config: dict[str, int] | None = None,
        module_name: str | None = None,
        class_name: str | None = None,
    ) -> dict[str, Any]:
        config = config or self._load_walk_forward_config()
        min_positive_fold_ratio = 0.60
        min_median_oos_net_pnl = 0.0
        hyp_id = getattr(hypothesis, "hypothesis_id", "unknown")

        net_pnls: list[float] = []
        symbols = sorted(feature_frames)
        pending_symbols = set(symbols)

        use_parallel = self.workers > 1 and len(feature_frames) > 1
        if use_parallel:
            ex: ProcessPoolExecutor | None = None
            try:
                ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=self._mp_context)
                futures = {
                    ex.submit(
                        _run_symbol_walk_forward_task,
                        symbol,
                        feature_paths[symbol],
                        module_name or hypothesis.__class__.__module__,
                        class_name or hypothesis.__class__.__name__,
                        config,
                    ): symbol
                    for symbol in symbols
                }
                try:
                    for fut in as_completed(futures, timeout=self.phase_timeout_seconds):
                        symbol = futures[fut]
                        pending_symbols.discard(symbol)
                        task_symbol, symbol_net_pnls, error = fut.result()
                        if error:
                            logger.warning("Walk-forward task failed for %s: %s", symbol, error)
                            continue
                        if task_symbol != symbol:
                            logger.warning("Walk-forward task symbol mismatch: expected %s got %s", symbol, task_symbol)
                        net_pnls.extend(symbol_net_pnls)
                except TimeoutError:
                    logger.error(
                        "Walk-forward timeout for %s after %ss; pending symbols: %s",
                        hyp_id,
                        self.phase_timeout_seconds,
                        ",".join(sorted(pending_symbols)),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Parallel walk-forward failed; falling back to sequential mode: %s", exc)
                use_parallel = False
                pending_symbols = set(symbols)
            finally:
                if ex is not None:
                    ex.shutdown(wait=False, cancel_futures=True)

        sequential_symbols = symbols if not use_parallel else sorted(pending_symbols)
        if sequential_symbols:
            if use_parallel and pending_symbols:
                logger.info(
                    "Walk-forward fallback for %s on pending symbols: %s",
                    hyp_id,
                    ",".join(sequential_symbols),
                )
            for symbol in sequential_symbols:
                symbol_df = feature_frames[symbol]
                if symbol_df.empty:
                    continue
                for _train_df, _val_df, test_df in walk_forward_splits(
                    symbol_df,
                    train_days=config["train_days"],
                    validate_days=config["validate_days"],
                    test_days=config["test_days"],
                ):
                    if test_df.empty:
                        continue
                    net_pnls.append(run_backtest(test_df, hypothesis).net_pnl)

        folds = len(net_pnls)
        if folds == 0:
            return {
                "folds": 0,
                "positive_folds": 0,
                "positive_fold_ratio": 0.0,
                "median_oos_net_pnl": 0.0,
                "min_positive_fold_ratio": min_positive_fold_ratio,
                "min_median_oos_net_pnl": min_median_oos_net_pnl,
                "passed": False,
            }

        net_pnls = sorted(net_pnls)
        positive_folds = sum(1 for p in net_pnls if p > 0)
        positive_fold_ratio = positive_folds / folds
        median_idx = folds // 2
        if folds % 2 == 1:
            median_oos_net_pnl = net_pnls[median_idx]
        else:
            median_oos_net_pnl = (net_pnls[median_idx - 1] + net_pnls[median_idx]) / 2

        passed = (
            positive_fold_ratio >= min_positive_fold_ratio
            and median_oos_net_pnl > min_median_oos_net_pnl
        )
        return {
            "folds": folds,
            "positive_folds": positive_folds,
            "positive_fold_ratio": round(positive_fold_ratio, 4),
            "median_oos_net_pnl": round(median_oos_net_pnl, 4),
            "min_positive_fold_ratio": min_positive_fold_ratio,
            "min_median_oos_net_pnl": min_median_oos_net_pnl,
            "passed": passed,
        }

    def _run_symbol_backtests(
        self,
        feature_frames: dict[str, pd.DataFrame],
        feature_paths: dict[str, Path],
        hypothesis: Any,
        module_name: str,
        class_name: str,
    ) -> tuple[list[EvalResult], dict[str, float], float]:
        """Run full backtests per symbol, in parallel when workers > 1."""
        symbol_results: list[EvalResult] = []
        symbol_timing: dict[str, float] = {}
        full_backtest_seconds = 0.0
        hyp_id = getattr(hypothesis, "hypothesis_id", "unknown")
        symbols = sorted(feature_frames)
        pending_symbols = set(symbols)

        use_parallel = self.workers > 1 and len(feature_frames) > 1
        if use_parallel:
            ex: ProcessPoolExecutor | None = None
            try:
                ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=self._mp_context)
                futures = {}
                for symbol in symbols:
                    symbol_start = time.perf_counter()
                    fut = ex.submit(
                        _run_symbol_backtest_task,
                        symbol,
                        feature_paths[symbol],
                        module_name,
                        class_name,
                    )
                    futures[fut] = (symbol, symbol_start)

                try:
                    for fut in as_completed(futures, timeout=self.phase_timeout_seconds):
                        symbol, symbol_start = futures[fut]
                        pending_symbols.discard(symbol)
                        elapsed = time.perf_counter() - symbol_start
                        symbol_timing[symbol] = round(elapsed, 4)
                        full_backtest_seconds += elapsed

                        task_symbol, result, error = fut.result()
                        if error:
                            logger.warning("Backtest task failed for %s: %s", symbol, error)
                            continue
                        if task_symbol != symbol:
                            logger.warning("Backtest task symbol mismatch: expected %s got %s", symbol, task_symbol)
                        if result is not None:
                            symbol_results.append(result)
                except TimeoutError:
                    logger.error(
                        "Full-backtest timeout for %s after %ss; pending symbols: %s",
                        hyp_id,
                        self.phase_timeout_seconds,
                        ",".join(sorted(pending_symbols)),
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Parallel full backtest failed; falling back to sequential mode: %s", exc)
                use_parallel = False
                pending_symbols = set(symbols)
            finally:
                if ex is not None:
                    ex.shutdown(wait=False, cancel_futures=True)

        sequential_symbols = symbols if not use_parallel else sorted(pending_symbols)
        if use_parallel and pending_symbols:
            logger.info(
                "Full-backtest fallback for %s on pending symbols: %s",
                hyp_id,
                ",".join(sequential_symbols),
            )

        for symbol in sequential_symbols:
            symbol_df = feature_frames[symbol]
            symbol_start = time.perf_counter()
            if symbol_df.empty:
                symbol_timing[symbol] = round(time.perf_counter() - symbol_start, 4)
                continue
            symbol_result = run_backtest(symbol_df, hypothesis)
            symbol_results.append(symbol_result)
            symbol_elapsed = time.perf_counter() - symbol_start
            full_backtest_seconds += symbol_elapsed
            symbol_timing[symbol] = round(symbol_elapsed, 4)

        return symbol_results, symbol_timing, full_backtest_seconds

    def _load_walk_forward_config(self) -> dict[str, int]:
        defaults = {"train_days": 60, "validate_days": 20, "test_days": 20}
        if not self.risk_config_path.exists():
            return defaults

        try:
            data = yaml.safe_load(self.risk_config_path.read_text(encoding="utf-8")) or {}
            wf = data.get("walk_forward", {})
            return {
                "train_days": int(wf.get("train_days", defaults["train_days"])),
                "validate_days": int(wf.get("validate_days", defaults["validate_days"])),
                "test_days": int(wf.get("test_days", defaults["test_days"])),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse walk-forward config from %s: %s", self.risk_config_path, exc)
            return defaults

    def _build_provenance(
        self,
        feature_frames: dict[str, pd.DataFrame],
        wf_config: dict[str, int],
    ) -> dict[str, Any]:
        data_summary = self._summarize_feature_coverage(feature_frames)
        return {
            "git": {
                "commit": self._git_commit_hash(),
                "is_dirty": self._git_is_dirty(),
            },
            "configs": {
                "risk": self._config_descriptor(self.risk_config_path),
                "costs": self._config_descriptor(_COSTS_CONFIG_PATH),
                "universe": self._config_descriptor(_UNIVERSE_CONFIG_PATH),
            },
            "walk_forward": wf_config,
            "data": {
                "features_dir": str(self.features_dir),
                **data_summary,
            },
        }

    def _summarize_feature_coverage(self, feature_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        coverage: dict[str, Any] = {}
        total_bars = 0
        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []

        for symbol, frame in sorted(feature_frames.items()):
            bars = int(len(frame))
            total_bars += bars
            start_iso, end_iso, start_ts, end_ts = self._event_time_bounds(frame)
            if start_ts is not None:
                starts.append(start_ts)
            if end_ts is not None:
                ends.append(end_ts)
            coverage[symbol] = {
                "bars": bars,
                "event_time_start": start_iso,
                "event_time_end": end_iso,
            }

        return {
            "symbol_count": len(coverage),
            "total_bars": total_bars,
            "event_time_start": min(starts).isoformat() if starts else None,
            "event_time_end": max(ends).isoformat() if ends else None,
            "symbol_coverage": coverage,
        }

    def _event_time_bounds(
        self,
        frame: pd.DataFrame,
    ) -> tuple[str | None, str | None, pd.Timestamp | None, pd.Timestamp | None]:
        if "event_time" not in frame.columns or frame.empty:
            return None, None, None, None

        times = pd.to_datetime(frame["event_time"], utc=True, errors="coerce").dropna()
        if times.empty:
            return None, None, None, None

        start_ts = times.min()
        end_ts = times.max()
        return start_ts.isoformat(), end_ts.isoformat(), start_ts, end_ts

    def _config_descriptor(self, path: Path) -> dict[str, str | None]:
        return {
            "path": str(path),
            "sha256": self._file_sha256(path),
        }

    def _file_sha256(self, path: Path) -> str | None:
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _git_commit_hash(self) -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            return out.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    def _git_is_dirty(self) -> bool:
        try:
            out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
            return bool(out.strip())
        except Exception:  # noqa: BLE001
            return False

    def _aggregate_symbol_results(self, hypothesis: Any, results: list[EvalResult]) -> EvalResult:
        total_trades = sum(r.total_trades for r in results)
        if total_trades == 0:
            return EvalResult(
                hypothesis_id=getattr(hypothesis, "hypothesis_id", "unknown"),
                run_id="",
                status=HypothesisStatus.DRAFT,
                net_pnl=0.0,
                profit_factor=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_drawdown=0.0,
                max_intraday_drawdown=0.0,
                worst_day=0.0,
                longest_losing_streak=0,
                trades_per_day=0.0,
                exposure_time=0.0,
                total_trades=0,
                slippage_sensitivity=0.0,
                composite_score=0.0,
            )

        total_net_pnl = sum(r.net_pnl for r in results)
        total_wins = sum(round(r.win_rate * r.total_trades) for r in results)
        total_losses = max(0, total_trades - total_wins)

        gross_wins = sum(max(0, round(r.win_rate * r.total_trades)) * r.avg_win for r in results)
        gross_losses = sum(max(0, r.total_trades - round(r.win_rate * r.total_trades)) * abs(r.avg_loss) for r in results)

        avg_win = gross_wins / total_wins if total_wins > 0 else 0.0
        avg_loss = -(gross_losses / total_losses) if total_losses > 0 else 0.0
        profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 999.0

        weighted = lambda attr: sum(getattr(r, attr) * r.total_trades for r in results) / total_trades

        return EvalResult(
            hypothesis_id=getattr(hypothesis, "hypothesis_id", "unknown"),
            run_id="",
            status=HypothesisStatus.DRAFT,
            net_pnl=round(total_net_pnl, 4),
            profit_factor=round(profit_factor, 4),
            win_rate=round(total_wins / total_trades, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            max_drawdown=round(max(r.max_drawdown for r in results), 4),
            max_intraday_drawdown=round(max(r.max_intraday_drawdown for r in results), 4),
            worst_day=round(min(r.worst_day for r in results), 4),
            longest_losing_streak=max(r.longest_losing_streak for r in results),
            trades_per_day=round(weighted("trades_per_day"), 4),
            exposure_time=round(weighted("exposure_time"), 4),
            total_trades=total_trades,
            slippage_sensitivity=round(weighted("slippage_sensitivity"), 4),
            composite_score=0.0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one autoresearch cycle.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of process workers for symbol/fold parallelism (default: 1 unless AUTORESEARCH_WORKERS is set).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    report = ResearchRunner(workers=args.workers).run_once()
    print(json.dumps({
        "run_at": report.get("run_at"),
        "hypotheses_evaluated": report.get("hypotheses_evaluated"),
        "total_seconds": report.get("timings", {}).get("total_seconds"),
        "workers": report.get("timings", {}).get("workers"),
    }, indent=2))


if __name__ == "__main__":
    main()
