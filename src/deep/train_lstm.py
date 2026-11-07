"""Train the CPU LSTM with early stopping and export its best checkpoint's predictions."""
import argparse
import copy
import json
import math
import random
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.deep.dataset import make_dataset, make_loader, make_loaders
from src.deep.model import LSTMClassifier, make_loss, make_optimizer

DEFAULTS = {"window": 20, "hidden": 64, "layers": 2, "dropout": 0.2, "lr": 1e-3,
            "batch_size": 64, "epochs": 100, "patience": 10, "feature_set": "all"}
SEED = 42
MODEL_NAME = "lstm"
CLIP_NORM = 1.0
ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT / "results" / "predictions" / "lstm.csv"
HISTORY_PATH = ROOT / "results" / "history" / "lstm_history.csv"
WEIGHTS_PATH = ROOT / "models" / "deep" / "lstm_best.pt"
CONFIG_PATH = ROOT / "models" / "deep" / "lstm_config.json"
HISTORY_COLUMNS = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
PREDICTION_COLUMNS = ["date", "split", "model", "y_true", "y_prob", "y_pred"]


def resolve_params(cli: dict, config_path: str | None) -> dict:
    """Merge defaults, an explicitly named JSON config, and non-None CLI values.

    Only DEFAULTS keys are accepted from either mapping; checkpoint metadata is
    ignored. The input mapping and DEFAULTS are left unchanged, and no files are
    written. No config is read when config_path is None.
    """
    params = DEFAULTS.copy()
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise ValueError(f"Config file does not exist: {path}")
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {path}: {exc.msg}") from exc
        if not isinstance(config, dict):
            raise ValueError("Config JSON must contain an object of parameter values")
        params.update({key: value for key, value in config.items() if key in DEFAULTS})
    params.update({key: value for key, value in cli.items()
                   if key in DEFAULTS and value is not None})

    for key in ("window", "hidden", "layers", "batch_size", "epochs", "patience"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{key} must be an integer >= 1")
        params[key] = int(value)
    for key in ("dropout", "lr"):
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
        params[key] = float(value)
    if not 0 <= params["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if params["lr"] <= 0:
        raise ValueError("lr must be greater than 0")
    if params["feature_set"] not in ("all", "stationary"):
        raise ValueError("feature_set must be 'all' or 'stationary'")
    return params


def _collect_logits(model, loader):
    """Collect CPU logits and labels in loader order with dropout disabled."""
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits.append(model(xb.cpu()))
            labels.append(yb.cpu())
    return torch.cat(logits), torch.cat(labels)


def _evaluate(model, loader, loss_fn):
    """Compute one loss over the full dataset, rather than averaging batch means."""
    logits, labels = _collect_logits(model, loader)
    loss = loss_fn(logits, labels).item()
    accuracy = ((torch.sigmoid(logits) >= 0.5) == labels).float().mean().item()
    return loss, accuracy


def _prediction_frame(model, params):
    """Predict unpurged splits chronologically using past-only B1 context."""
    frames = []
    for split in ("train", "val", "test"):
        dataset, dates = make_dataset(split, params["window"], params["feature_set"], purge=False)
        loader = make_loader(dataset, batch_size=512, shuffle=False)
        logits, labels = _collect_logits(model, loader)
        probabilities = torch.sigmoid(logits).numpy()
        frames.append(pd.DataFrame({
            "date": dates.strftime("%Y-%m-%d"),
            "split": split,
            "model": MODEL_NAME,
            "y_true": labels.numpy().astype("int64"),
            "y_prob": probabilities,
            "y_pred": (probabilities >= 0.5).astype("int64"),
        }, columns=PREDICTION_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def train(params, write_outputs=True) -> dict:
    """Fit with validation-loss early stopping and return the best epoch's metrics.

    Partial params inherit DEFAULTS. Metrics use evaluation mode and all samples
    in the purged training and unpurged validation datasets. Test metrics are
    never computed here. With write_outputs=False, tuning runs write no files.
    """
    params = resolve_params(params, None)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    loaders = make_loaders(params["window"], params["batch_size"], params["feature_set"])
    if not len(loaders["train"].dataset) or not len(loaders["val"].dataset):
        raise ValueError("Training and validation datasets must both contain samples")
    # A separate ordered loader keeps evaluation from advancing the training shuffle.
    train_eval_loader = make_loader(loaders["train"].dataset, params["batch_size"], shuffle=False)
    y_train = loaders["train"].dataset.tensors[1].numpy()
    model = LSTMClassifier(loaders["n_features"], params["hidden"], params["layers"],
                           params["dropout"]).cpu()
    loss_fn = make_loss(y_train).cpu()
    optimizer = make_optimizer(model, params["lr"])
    history = []
    best_loss, best_state, best_row = float("inf"), None, None
    wait = 0

    for epoch in range(1, params["epochs"] + 1):
        model.train()
        for xb, yb in loaders["train"]:
            optimizer.zero_grad()
            logits = model(xb.cpu())
            loss = loss_fn(logits, yb.cpu())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()

        train_loss, train_acc = _evaluate(model, train_eval_loader, loss_fn)
        val_loss, val_acc = _evaluate(model, loaders["val"], loss_fn)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError("Training produced a non-finite loss; check inputs and parameters")
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_acc": train_acc, "val_acc": val_acc}
        history.append(row)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_row = row.copy()
            wait = 0
        else:
            wait += 1
            if wait >= params["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    val_logits, val_labels = _collect_logits(model, loaders["val"])
    val_probabilities = torch.sigmoid(val_logits).numpy()
    val_targets = val_labels.numpy()
    val_auc = (float(roc_auc_score(val_targets, val_probabilities))
               if len(np.unique(val_targets)) > 1 else float("nan"))
    result = {"params": params, "n_features": loaders["n_features"],
              "best_epoch": best_row["epoch"], "epochs_run": len(history),
              "val_loss": best_loss, "val_acc": best_row["val_acc"], "val_auc": val_auc,
              "train_loss": best_row["train_loss"], "train_acc": best_row["train_acc"],
              "history": history}

    if write_outputs:
        predictions = _prediction_frame(model, params)
        for path in (HISTORY_PATH, WEIGHTS_PATH, CONFIG_PATH, PREDICTIONS_PATH):
            path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history, columns=HISTORY_COLUMNS).to_csv(HISTORY_PATH, index=False)
        torch.save(best_state, WEIGHTS_PATH)
        config = {**params, "n_features": result["n_features"], "best_epoch": result["best_epoch"],
                  "val_loss": best_loss, "val_auc": val_auc if math.isfinite(val_auc) else None,
                  "seed": SEED}
        CONFIG_PATH.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        predictions.to_csv(PREDICTIONS_PATH, index=False)
    return result


def main():
    """Train from explicitly merged options and report validation and prediction rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("window", "hidden", "layers", "batch-size", "epochs", "patience"):
        parser.add_argument(f"--{flag}", type=int, default=None)
    for flag in ("dropout", "lr"):
        parser.add_argument(f"--{flag}", type=float, default=None)
    parser.add_argument("--feature-set", choices=("all", "stationary"), default=None)
    parser.add_argument("--config", default=None, help="Optional JSON config; explicit flags override it")
    cli = vars(parser.parse_args())
    config_path = cli.pop("config")
    try:
        params = resolve_params(cli, config_path)
    except ValueError as exc:
        parser.error(str(exc))
    result = train(params, write_outputs=True)
    print(f"Final params: {json.dumps(result['params'])}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Epochs run: {result['epochs_run']}")
    print(f"Best val_loss: {result['val_loss']:.6f}")
    print(f"Best val_acc: {result['val_acc']:.6f}")
    print(f"Best val_auc: {result['val_auc']:.6f}")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    for split in ("val", "test"):
        rows = predictions[predictions["split"] == split]
        print(f"{split} up-rate: predicted={rows['y_pred'].mean():.6f} "
              f"true={rows['y_true'].mean():.6f}")


if __name__ == "__main__":
    main()
