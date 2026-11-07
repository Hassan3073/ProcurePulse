"""Tune the CPU LSTM on validation AUC, with resumable progress and no test scoring."""
import argparse
import contextlib
import io
import itertools
import json
import math
import time
from pathlib import Path

import pandas as pd

from src.deep.dataset import get_feature_columns
from src.deep.train_lstm import CONFIG_PATH, DEFAULTS, ROOT, SEED, resolve_params, train

OUTPUT_PATH = ROOT / "results" / "tuning" / "lstm_tuning.csv"
COLUMNS = ["model", "feature_set", "params", "score_name", "mean_score", "std_score", "selected"]
TUNED_KEYS = ("batch_size", "dropout", "hidden", "lr", "window")


def build_grid(grid="full", feature_set="stationary"):
    """Return 64 full or 16 reduced configurations in deterministic product order."""
    if grid not in ("full", "reduced"):
        raise ValueError("grid must be 'full' or 'reduced'")
    if feature_set not in ("all", "stationary"):
        raise ValueError("feature_set must be 'all' or 'stationary'")
    keys = ("lr", "batch_size", "dropout", "hidden", "window", "feature_set")
    values = ([1e-3, 3e-4], [32, 64], [0.1, 0.3], [32, 64],
              [20, 60] if grid == "full" else [20],
              ["all", "stationary"] if grid == "full" else [feature_set])
    fixed = {key: DEFAULTS[key] for key in ("layers", "epochs", "patience")}
    return [resolve_params({**fixed, **dict(zip(keys, combination))}, None)
            for combination in itertools.product(*values)]


def _key(params):
    return params["feature_set"], json.dumps({key: params[key] for key in TUNED_KEYS}, sort_keys=True)


def _row_key(row):
    return row["feature_set"], row["params"]


def _write_if_changed(path, content):
    """Avoid rewriting unchanged resume results, including their modification time."""
    payload = content.encode("utf-8")
    if path.is_file() and path.read_bytes() == payload:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _save_table(rows, path):
    _write_if_changed(path, pd.DataFrame(rows, columns=COLUMNS).to_csv(index=False, na_rep=""))


def _read_rows(path):
    table = pd.read_csv(path, converters={"mean_score": lambda value: float(value) if value else float("nan")})
    if table.columns.tolist() != COLUMNS:
        raise ValueError(f"Resume table must have columns in this order: {COLUMNS}")
    rows, seen = [], set()
    for row in table.to_dict("records"):
        if row["model"] != "lstm" or row["score_name"] != "val_auc":
            raise ValueError("Resume table must contain only LSTM validation AUC runs")
        tuned = json.loads(row["params"])
        if not isinstance(tuned, dict) or set(tuned) != set(TUNED_KEYS):
            raise ValueError(f"Each params JSON must contain exactly {TUNED_KEYS}")
        params = resolve_params({**tuned, "feature_set": row["feature_set"]}, None)
        key = _key(params)
        if key in seen:
            raise ValueError("Resume table contains duplicate feature_set/params pairs")
        seen.add(key)
        row.update(params=key[1], mean_score=float(row["mean_score"]),
                   std_score=float("nan"), selected=int(row["selected"]))
        rows.append(row)
    return rows


def _read_winner_metadata(path, rows):
    """Recover optional metadata for a previous winner; CSV scores remain authoritative."""
    if not path.is_file():
        return {}
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        key = _key(resolve_params({}, str(path)))
        if any(_row_key(row) == key and row["mean_score"] == config.get("val_auc") for row in rows):
            return {key: config}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Note: previous winner metadata unavailable: {exc}")
    return {}


def _rank_rows(rows, metadata):
    """Rank finite AUCs, using loss only when every tied run has that metadata."""
    grid_order = {_key(params): index for index, params in enumerate(build_grid())}
    score_groups = {}
    for index, row in enumerate(rows):
        if math.isfinite(row["mean_score"]):
            score_groups.setdefault(row["mean_score"], []).append((index, row))
    ranked = []
    for score in sorted(score_groups, reverse=True):
        group = score_groups[score]
        losses = [metadata.get(_row_key(row), {}).get("val_loss") for _, row in group]
        use_loss = all(loss is not None and math.isfinite(loss) for loss in losses)
        group.sort(key=lambda item: (
            metadata[_row_key(item[1])]["val_loss"] if use_loss else 0,
            grid_order.get(_row_key(item[1]), len(grid_order) + item[0]),
        ))
        ranked.extend(row for _, row in group)
    return ranked


def _save_winner(row, metadata, path):
    params = resolve_params({**json.loads(row["params"]), "feature_set": row["feature_set"]}, None)
    details = metadata.get(_row_key(row), {})
    n_features = details.get("n_features")
    if n_features is None:
        n_features = len(get_feature_columns(row["feature_set"]))
    config = {**params, "n_features": int(n_features), "val_auc": row["mean_score"], "seed": SEED}
    for key in ("best_epoch", "val_loss"):
        if details.get(key) is not None and math.isfinite(details[key]):
            config[key] = details[key]
    _write_if_changed(path, json.dumps(config, indent=2, allow_nan=False) + "\n")
    return config


def run_tuning(configurations, output, config_out, resume=False):
    """Run the requested prefix, saving only the table and winner config paths."""
    output, config_out = Path(output), Path(config_out)
    if output.resolve() == config_out.resolve():
        raise ValueError("--output and --config-out must be different files")
    rows = _read_rows(output) if resume and output.is_file() else []
    metadata = _read_winner_metadata(config_out, rows) if rows else {}
    seen = {_row_key(row) for row in rows}
    pending = [(index, params) for index, params in enumerate(configurations, start=1)
               if _key(params) not in seen]
    skipped = len(configurations) - len(pending)
    print(f"Requested {len(configurations)} runs; skipped {skipped} existing; {len(pending)} to train.", flush=True)
    durations = []
    interrupted = False
    try:
        for index, params in pending:
            started = time.perf_counter()
            result, error = None, None
            try:
                # B3 prints per epoch; tuning reports one compact line per finished run.
                with contextlib.redirect_stdout(io.StringIO()):
                    result = train(params, write_outputs=False)
                score = float(result["val_auc"])
                if not math.isfinite(score):
                    score = float("nan")
                metadata[_key(params)] = {key: result[key] for key in ("n_features", "best_epoch", "val_loss")}
            except Exception as exc:
                score = float("nan")
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            durations.append(elapsed)
            feature_set, serialized = _key(params)
            rows.append({"model": "lstm", "feature_set": feature_set, "params": serialized,
                         "score_name": "val_auc", "mean_score": score,
                         "std_score": float("nan"), "selected": 0})
            for row in rows:
                row["selected"] = 0
            _save_table(rows, output)
            remaining = len(pending) - len(durations)
            eta = sum(durations) / len(durations) * remaining
            val_loss = result["val_loss"] if result is not None else float("nan")
            best_epoch = result["best_epoch"] if result is not None else "n/a"
            message = (f"Run {index}/{len(configurations)}: params={json.dumps(params, sort_keys=True)} "
                       f"val_auc={score:.6f} val_loss={val_loss:.6f} best_epoch={best_epoch} "
                       f"seconds={elapsed:.1f} ETA={eta:.1f}s")
            print(message + (f" ERROR: {error}" if error else ""), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; saving finished runs. Continue with --resume.", flush=True)

    ranked = _rank_rows(rows, metadata)
    for row in rows:
        row["selected"] = int(not interrupted and bool(ranked) and row is ranked[0])
    _save_table(rows, output)
    if not ranked:
        print("No valid validation AUC is available; no winner config was written.")
        return None
    winner = _save_winner(ranked[0], metadata, config_out)
    print(f"Winner{' so far' if interrupted else ''}: {json.dumps(winner, sort_keys=True)}")
    top = pd.DataFrame(ranked[:5], columns=COLUMNS)
    top.insert(0, "rank", range(1, len(top) + 1))
    print("Ranked top 5 (or all valid runs if fewer):")
    print(top.to_string(index=False))
    next_path = ("models/deep/lstm_config.json" if config_out.resolve() == CONFIG_PATH.resolve()
                 else f'"{config_out}"')
    print(f"Next command: .\\.venv\\Scripts\\python -m src.deep.train_lstm --config {next_path}")
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=("full", "reduced"), default="full")
    parser.add_argument("--feature-set", choices=("all", "stationary"), default="stationary",
                        help="Feature set for the reduced grid only")
    parser.add_argument("--limit", type=int, help="Run only the first N grid combinations")
    parser.add_argument("--resume", action="store_true", help="Keep existing scores and skip completed runs")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--config-out", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    configurations = build_grid(args.grid, args.feature_set)
    if args.limit is not None:
        configurations = configurations[:args.limit]
    run_tuning(configurations, args.output, args.config_out, args.resume)


if __name__ == "__main__":
    main()
