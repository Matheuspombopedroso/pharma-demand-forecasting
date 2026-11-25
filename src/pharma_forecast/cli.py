"""Command-line interface: ``pharma-forecast <command>``.

Each pipeline step is a sub-command so it can be run, inspected and
re-run on its own (the Makefile and Docker image just call these):

=================  ======================================================
generate-data      write the synthetic CSV
train              walk-forward CV + hold-out + fit + log to MLflow
tune               Ray Tune search, then train with the best params
register           register a run's model; it becomes @challenger
promote            move @champion to the challenger if it is better
rollback           move @champion back to @previous
status             list registry versions and aliases
predict            forecast next 7/15/30 days with @champion
=================  ======================================================

Small state files in ``outputs/`` (last run id, best params) let the steps
chain without copy-pasting ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from pharma_forecast.config import Settings, get_settings
from pharma_forecast.data import (
    DEFAULT_INGREDIENTS,
    generate_consumption,
    load_consumption,
    save_consumption,
)

OUTPUTS = Path("outputs")
LAST_RUN_FILE = OUTPUTS / "last_run_id.txt"
BEST_PARAMS_FILE = OUTPUTS / "best_params.json"


def _print_df(df: pd.DataFrame) -> None:
    """Print a DataFrame fully, without pandas' column truncation."""
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))


def _save_run_id(run_id: str) -> None:
    OUTPUTS.mkdir(exist_ok=True)
    LAST_RUN_FILE.write_text(run_id)


def cmd_generate(args: argparse.Namespace, s: Settings) -> None:
    """Generate and save the synthetic dataset."""
    df = generate_consumption(
        n_days=args.days, ingredients=DEFAULT_INGREDIENTS[: args.ingredients], seed=s.seed
    )
    path = save_consumption(df, Path(args.out) if args.out else s.data_path)
    print(f"Wrote {len(df)} rows ({df['ingredient'].nunique()} ingredients) to {path}")


def cmd_train(args: argparse.Namespace, s: Settings) -> None:
    """Train with default params or a JSON params file."""
    from pharma_forecast.train import run_training  # lazy: heavy imports

    params = json.loads(Path(args.params).read_text()) if args.params else None
    res = run_training(load_consumption(s.data_path), s, params, args.n_folds, args.test_days)
    _save_run_id(res.run_id)
    print(f"MLflow run: {res.run_id}  cv_wape={res.cv_metrics['cv_wape']:.4f}")
    print("Hold-out comparison (lower is better):")
    _print_df(res.holdout.round(4))


def cmd_tune(args: argparse.Namespace, s: Settings) -> None:
    """Ray Tune search, then (unless --no-train) train with the best params."""
    from pharma_forecast.train import run_training
    from pharma_forecast.tune import run_tuning, save_best_params

    raw = load_consumption(s.data_path)
    res = run_tuning(raw, s, args.num_samples, args.max_concurrent, args.n_folds, args.test_days)
    OUTPUTS.mkdir(exist_ok=True)
    save_best_params(res, BEST_PARAMS_FILE)
    print(
        f"Tuned {len(res.trials)} trials in {res.wall_seconds:.1f}s "
        f"(max_concurrent={args.max_concurrent}); best cv_wape={res.best_cv_wape:.4f}"
    )
    print(f"Best params -> {BEST_PARAMS_FILE}: {json.dumps(res.best_params)}")
    if args.no_train:
        return
    tr = run_training(
        raw,
        s,
        res.best_params,
        args.n_folds,
        args.test_days,
        run_name="train-tuned",
        extra_tags={"source": "tune", "tune_run_id": res.parent_run_id},
    )
    _save_run_id(tr.run_id)
    print(f"MLflow run (tuned params): {tr.run_id}")
    _print_df(tr.holdout.round(4))


def cmd_register(args: argparse.Namespace, s: Settings) -> None:
    """Register a run's model (defaults to the last trained run)."""
    from pharma_forecast.registry import register

    run_id = args.run_id or LAST_RUN_FILE.read_text().strip()
    version = register(run_id, s.mlflow_tracking_uri)
    print(f"Registered run {run_id} as version {version}")


def cmd_promote(args: argparse.Namespace, s: Settings) -> None:
    """Promote challenger to champion when it wins on hold-out WAPE."""
    from pharma_forecast.registry import promote

    try:
        d = promote(s.mlflow_tracking_uri, args.version, args.force)
    except ValueError as exc:
        # Normal right after the very first `register`: that version was made
        # champion directly, so there is no challenger waiting.
        print(f"Nothing to promote: {exc}")
        return
    print(d.reason)
    print(
        f"candidate v{d.candidate_version} wape={d.candidate_metric:.4f}; "
        f"previous champion v{d.previous_champion} wape={d.champion_metric}"
    )


def cmd_rollback(args: argparse.Namespace, s: Settings) -> None:
    """Restore the previous champion."""
    from pharma_forecast.registry import rollback

    old, new = rollback(s.mlflow_tracking_uri, args.version)
    print(f"Rolled back: champion v{old} -> v{new}")


def cmd_status(args: argparse.Namespace, s: Settings) -> None:
    """Show registry versions and aliases."""
    from pharma_forecast.registry import status

    _print_df(pd.DataFrame(status(s.mlflow_tracking_uri)))


def cmd_predict(args: argparse.Namespace, s: Settings) -> None:
    """Forecast from the latest data with the chosen alias."""
    from pharma_forecast.predict import predict_latest

    out = predict_latest(load_consumption(s.data_path), s, alias=args.alias)
    _print_df(out)
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    """Define every sub-command and its options."""
    p = argparse.ArgumentParser(prog="pharma-forecast", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate-data", help="write synthetic consumption CSV")
    g.add_argument("--days", type=int, default=730)
    g.add_argument("--ingredients", type=int, default=len(DEFAULT_INGREDIENTS))
    g.add_argument("--out", default=None)
    g.set_defaults(func=cmd_generate)

    for name, func in (("train", cmd_train), ("tune", cmd_tune)):
        sp = sub.add_parser(name)
        sp.add_argument("--n-folds", type=int, default=5, help="folds incl. hold-out")
        sp.add_argument("--test-days", type=int, default=28)
        sp.set_defaults(func=func)
        if name == "train":
            sp.add_argument("--params", default=None, help="JSON file of XGBoost params")
        else:
            sp.add_argument("--num-samples", type=int, default=20)
            sp.add_argument("--max-concurrent", type=int, default=4)
            sp.add_argument("--no-train", action="store_true")

    r = sub.add_parser("register")
    r.add_argument("--run-id", default=None)
    r.set_defaults(func=cmd_register)

    pr = sub.add_parser("promote")
    pr.add_argument("--version", default=None)
    pr.add_argument("--force", action="store_true")
    pr.set_defaults(func=cmd_promote)

    rb = sub.add_parser("rollback")
    rb.add_argument("--version", default=None)
    rb.set_defaults(func=cmd_rollback)

    sub.add_parser("status").set_defaults(func=cmd_status)

    pd_ = sub.add_parser("predict")
    pd_.add_argument("--alias", default="champion")
    pd_.add_argument("--out", default=None)
    pd_.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point used by the ``pharma-forecast`` console script."""
    args = build_parser().parse_args(argv)
    args.func(args, get_settings())
    return 0


if __name__ == "__main__":
    sys.exit(main())
