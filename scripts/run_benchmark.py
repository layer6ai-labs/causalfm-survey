#!/usr/bin/env python3
"""Benchmark causal models on RealCause Lalonde CPS and PSID realizations."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOPFN_DIR = PROJECT_ROOT / "notebooks" / "Do-PFN"
CAUSALFM_DIR = PROJECT_ROOT / "notebooks" / "CausalFM-toolkit"
CAUSALFM_CHECKPOINT = CAUSALFM_DIR / "checkpoints" / "checkpoints_standard" / "best_model.pth"
COHORTS = ("cps", "psid")
BASE_SEED = 82718
SPLIT_SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    label: str
    name: str
    wrapper: type
    foundation: bool
    supports_cate: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RealCause Lalonde benchmark from "
            "notebooks/RealCause_with_hpo_benchmark.ipynb."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument(
        "--cpu",
        dest="device",
        action="store_const",
        const="cpu",
        help="Run every model on CPU.",
    )
    devices.add_argument(
        "--gpu",
        dest="device",
        action="store_const",
        const="cuda",
        help="Run foundation models on CUDA; metalearners remain on CPU.",
    )
    parser.set_defaults(device="cpu")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run IPW without HPO on one realization per cohort as a lightweight "
            "end-to-end check."
        ),
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=10,
        help="RealCause realizations per cohort.",
    )
    parser.add_argument(
        "--hpo-time-budget",
        type=float,
        default=900.0,
        help="FLAML budget in seconds for each nuisance-model search.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for result CSVs.",
    )
    args = parser.parse_args()
    if not 1 <= args.n_realizations <= 100:
        parser.error("--n-realizations must be between 1 and 100")
    if args.hpo_time_budget <= 0:
        parser.error("--hpo-time-budget must be positive")
    return args


def configure_logging(timestamp: str) -> tuple[logging.Logger, Path]:
    log_path = PROJECT_ROOT / "logs" / f"run_benchmark_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.captureWarnings(True)
    return logging.getLogger("run_benchmark"), log_path


def seed_everything(seed: int, np: Any, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup(torch: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_available(spec: ModelSpec) -> bool:
    if spec.name == "Do-PFN":
        return spec.wrapper.is_available(repo_dir=str(DOPFN_DIR))
    if spec.name == "CausalFM":
        return spec.wrapper.is_available(checkpoint_path=str(CAUSALFM_CHECKPOINT))
    return spec.wrapper.is_available()


def make_model(
    spec: ModelSpec,
    foundation_device: str,
    hpo: bool,
    hpo_config: Any,
) -> Any:
    if spec.name == "CausalPFN":
        return spec.wrapper(device=foundation_device, cap_num_neighbours=True)
    if spec.name == "Do-PFN":
        return spec.wrapper(repo_dir=str(DOPFN_DIR), device=foundation_device)
    if spec.name == "CausalFM":
        return spec.wrapper(
            checkpoint_path=str(CAUSALFM_CHECKPOINT),
            device=foundation_device,
        )
    return spec.wrapper(device="cpu", hpo=hpo, hpo_config=hpo_config)


def task_seed(model_index: int, cohort_index: int, realization: int, stage: str) -> int:
    return (
        BASE_SEED
        + model_index * 100_000
        + cohort_index * 10_000
        + realization * 10
        + (1 if stage == "cate" else 2)
    )


def best_params(model: Any) -> str:
    return json.dumps(
        getattr(model, "best_params_", {}) or {},
        sort_keys=True,
        default=repr,
    )


def save_results(records: list[dict[str, Any]], output_path: Path, pd: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    os.replace(temporary, output_path)


def run(args: argparse.Namespace, timestamp: str, logger: logging.Logger) -> Path:
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for path in reversed((PROJECT_ROOT, DOPFN_DIR, CAUSALFM_DIR)):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import numpy as np
    import pandas as pd
    import torch

    from causal_bench import (
        CausalFMWrapper,
        CausalPFNWrapper,
        DRWrapper,
        DebiasedMLWrapper,
        DoPFNWrapper,
        HPOConfig,
        IPWWrapper,
        SLearnerWrapper,
        TLearnerWrapper,
        XLearnerWrapper,
        ate_abs_error,
        ate_rel_error,
        evaluate_cate,
        load_lalonde_realcause,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--gpu was requested, but CUDA is not available")

    seed_everything(BASE_SEED, np, torch)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    specs = [
        ModelSpec("CausalPFN (Foundation)", "CausalPFN", CausalPFNWrapper, True),
        ModelSpec("Do-PFN (Foundation)", "Do-PFN", DoPFNWrapper, True),
        ModelSpec("CausalFM (Foundation)", "CausalFM", CausalFMWrapper, True),
        ModelSpec("S-learner", "S-learner", SLearnerWrapper, False),
        ModelSpec("T-learner", "T-learner", TLearnerWrapper, False),
        ModelSpec("X-learner", "X-learner", XLearnerWrapper, False),
        ModelSpec("Debiased ML", "Debiased ML", DebiasedMLWrapper, False),
        ModelSpec("IPW", "IPW", IPWWrapper, False, supports_cate=False),
        ModelSpec("DR (Doubly Robust)", "DR", DRWrapper, False),
    ]
    n_realizations = 1 if args.smoke else args.n_realizations
    hpo = not args.smoke
    if args.smoke:
        specs = [spec for spec in specs if spec.name == "IPW"]

    missing = [spec.label for spec in specs if not model_available(spec)]
    missing_ate = [spec.label for spec in specs if not hasattr(spec.wrapper, "estimate_ate")]
    if missing or missing_ate:
        raise RuntimeError(
            f"Cannot start: unavailable={missing}, without estimate_ate={missing_ate}. "
            "See CLAUDE.md for optional dependency and vendor setup."
        )

    hpo_config = HPOConfig(
        time_budget=args.hpo_time_budget,
        cv=3,
        verbose=0,
        early_stop=True,
    )
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    mode = "smoke" if args.smoke else "full"
    output_path = results_dir / f"realcause_lalonde_{mode}_{timestamp}.csv"

    logger.info("Mode: %s; realizations per cohort: %d", mode, n_realizations)
    logger.info("Foundation device: %s; metalearner device: cpu", args.device)
    logger.info(
        "HPO: %s",
        f"{args.hpo_time_budget:g}s per search" if hpo else "disabled",
    )
    logger.info("Models: %s", ", ".join(spec.label for spec in specs))
    logger.info("Results: %s", output_path)

    datasets = {}
    for cohort in COHORTS:
        logger.info("Loading %s (%d realization(s))", cohort.upper(), n_realizations)
        datasets[cohort] = load_lalonde_realcause(
            cohort,
            n_realizations=n_realizations,
            seed=SPLIT_SEED,
            test_ratio=0.1,
        )

    records: list[dict[str, Any]] = []
    total = len(specs) * len(COHORTS) * n_realizations
    completed = 0

    for model_index, spec in enumerate(specs):
        for cohort_index, cohort in enumerate(COHORTS):
            for realization in datasets[cohort]:
                prefix = (
                    f"[{completed + 1}/{total}] {spec.label} {cohort.upper()} "
                    f"realization {realization.realization}"
                )
                cate_model = ate_model = None
                tau_hat = lower = upper = None
                cate_runtime = None
                cate_params = None
                try:
                    if spec.supports_cate:
                        seed = task_seed(
                            model_index,
                            cohort_index,
                            realization.realization,
                            "cate",
                        )
                        seed_everything(seed, np, torch)
                        logger.info("%s: CATE fit", prefix)
                        start = time.perf_counter()
                        cate_model = make_model(spec, args.device, hpo, hpo_config)
                        cate_model.fit(
                            realization.X_train,
                            realization.T_train,
                            realization.Y_train,
                        )
                        tau_hat, lower, upper = cate_model.predict(realization.X_test)
                        tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
                        if len(tau_hat) != len(realization.X_test):
                            raise RuntimeError("CATE output has the wrong length")
                        cate_runtime = time.perf_counter() - start
                        cate_params = best_params(cate_model)
                        logger.info("%s: CATE done in %.2fs", prefix, cate_runtime)
                        cate_model = None
                        cleanup(torch)
                    else:
                        seed = None
                        logger.info("%s: CATE unsupported; skipped", prefix)

                    ate_seed = task_seed(
                        model_index,
                        cohort_index,
                        realization.realization,
                        "ate",
                    )
                    seed_everything(ate_seed, np, torch)
                    logger.info("%s: fresh full-data ATE fit", prefix)
                    start = time.perf_counter()
                    ate_model = make_model(spec, args.device, hpo, hpo_config)
                    ate_model.fit(
                        realization.X_full,
                        realization.T_full,
                        realization.Y_full,
                    )
                    ate_hat = np.asarray(
                        ate_model.estimate_ate(
                            realization.X_full,
                            realization.T_full,
                            realization.Y_full,
                        ),
                        dtype=float,
                    )
                    if ate_hat.size != 1 or not np.isfinite(ate_hat).all():
                        raise RuntimeError(f"ATE must be one finite value, got {ate_hat}")
                    ate_hat = float(ate_hat.reshape(-1)[0])
                    ate_runtime = time.perf_counter() - start
                    ate_params = best_params(ate_model)
                    logger.info("%s: ATE done in %.2fs", prefix, ate_runtime)

                    if spec.supports_cate:
                        metrics = evaluate_cate(
                            tau_hat,
                            realization.tau_true_test,
                            ate_hat=ate_hat,
                            ate_true=realization.ate_true,
                            lower=lower,
                            upper=upper,
                            runtime_s=cate_runtime + ate_runtime,
                        )
                    else:
                        metrics = {
                            "pehe": None,
                            "ate_hat": ate_hat,
                            "ate_true": float(realization.ate_true),
                            "ate_abs_error": ate_abs_error(
                                ate_hat,
                                realization.ate_true,
                            ),
                            "ate_rel_error": ate_rel_error(
                                ate_hat,
                                realization.ate_true,
                            ),
                            "bias": None,
                            "coverage_95": None,
                            "runtime_s": ate_runtime,
                        }

                    records.append(
                        {
                            **metrics,
                            "model": spec.label,
                            "cohort": cohort,
                            "realization": int(realization.realization),
                            "execution_device": args.device if spec.foundation else "cpu",
                            "cate_runtime_s": cate_runtime,
                            "ate_runtime_s": ate_runtime,
                            "cate_best_params": cate_params,
                            "ate_best_params": ate_params,
                            "cate_seed": seed,
                            "ate_seed": ate_seed,
                            "hpo_time_budget_s": args.hpo_time_budget if hpo else None,
                            "ate_method": f"{spec.wrapper.__name__}.estimate_ate",
                        }
                    )
                    save_results(records, output_path, pd)
                    completed += 1
                    logger.info("%s: saved", prefix)
                except Exception:
                    logger.exception("%s: failed", prefix)
                    raise
                finally:
                    cate_model = ate_model = None
                    cleanup(torch)

        clear_cache = getattr(spec.wrapper, "clear_model_cache", None)
        if clear_cache is not None:
            clear_cache()
        cleanup(torch)

    frame = pd.DataFrame(records)
    summary = frame.groupby(["model", "cohort"], sort=False).agg(
        completed=("realization", "count"),
        pehe_mean=("pehe", "mean"),
        pehe_sem=("pehe", "sem"),
        ate_rel_error_mean=("ate_rel_error", "mean"),
        ate_rel_error_sem=("ate_rel_error", "sem"),
    )
    summary[["pehe_mean", "pehe_sem"]] /= 1000.0
    logger.info("Summary (PEHE columns are x1e3):\n%s", summary.to_string())
    logger.info("SUCCESS: completed %d/%d tasks", completed, total)
    return output_path


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logger, log_path = configure_logging(timestamp)
    logger.info("Log file: %s", log_path)
    try:
        output_path = run(args, timestamp, logger)
    except Exception:
        logger.exception("Benchmark terminated")
        return 1
    logger.info("Raw results: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
