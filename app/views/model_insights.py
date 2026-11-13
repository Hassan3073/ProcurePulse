"""Read-only Deliverable 2 results for Deliverable 3 B3."""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAMES = {"majority": "Majority baseline", "logreg": "Logistic regression",
         "random_forest": "Random forest", "hist_gb": "Gradient boosting", "lstm": "LSTM"}
GREEN, AMBER, RED, GREY = "#4CAF50", "#D4A017", "#E5534B", "#969BA3"


def friendly(model):
    return NAMES.get(str(model), str(model))


def unavailable(name):
    st.info(f"{Path(name).name} is not available.")


@st.cache_data(persist=False, show_spinner=False)
def load_csv(name, columns, numeric=()):
    """Read and validate a saved table, keeping all caching in memory."""
    try:
        frame = pd.read_csv(ROOT / name)
        if frame.empty or not set(columns).issubset(frame.columns):
            return None
        frame = frame.loc[:, list(columns)].copy()
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if frame[list(columns)].isna().any().any():
            return None
        return frame
    except Exception:
        return None


def table(name, columns, numeric=()):
    frame = load_csv(name, tuple(columns), tuple(numeric))
    if frame is None:
        unavailable(name)
    return frame


def show_image(name, caption):
    try:
        content = (ROOT / "report/figures" / name).read_bytes()
        st.image(content, use_container_width=True, alt=caption)
        st.caption(caption)
    except Exception:
        unavailable(name)


def styled(chart):
    return (chart.properties(height=350, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor="#D8DCE2", titleColor="#D8DCE2",
                            gridColor="#30343A", domainColor="#555B63")
            .configure_legend(labelColor="#D8DCE2", titleColor="#D8DCE2"))


def winners(frame, metric):
    if frame.empty:
        return "not available"
    best = frame[metric].max()
    names = ", ".join(frame.loc[frame[metric].eq(best), "model"].map(friendly))
    return f"{names} ({best:.3f})"


def tuning(name):
    frame = table(name, ["model", "feature_set", "params", "score_name", "mean_score", "selected"],
                  ["mean_score", "selected"])
    if frame is None:
        return
    parsed = []
    try:
        for value in frame["params"]:
            params = json.loads(value)
            if not isinstance(params, dict):
                raise ValueError("Parameters must be a JSON object")
            parsed.append(params)
    except (ValueError, TypeError):
        unavailable(name)
        return
    st.caption(f"{len(frame)} configurations evaluated. Score: {', '.join(frame['score_name'].astype(str).unique())}.")
    if name.endswith("lstm_tuning.csv"):
        if len(frame) < 64:
            st.caption("The planned full LSTM grid was 64 runs; it was only partially evaluated.")
        rates = pd.to_numeric(pd.Series([p.get("lr") for p in parsed]), errors="coerce")
        if rates.notna().all() and not np.isclose(rates, 0.0003).any():
            st.caption("Learning rate 3e-4 was not tested.")
    elif frame["score_name"].eq("cv_roc_auc").all():
        st.caption("The saved grid-search scores are cross-validation AUC, not held-out validation AUC.")
    display = frame[["model", "feature_set", "mean_score", "selected"]].rename(
        columns={"model": "Model", "feature_set": "Feature set", "mean_score": "Score", "selected": "Selected"})
    display["Model"] = display["Model"].map(friendly)
    display["Selected"] = display["Selected"].eq(1)
    parameters = pd.json_normalize(parsed).add_prefix("Parameter: ").map(
        lambda value: json.dumps(value) if isinstance(value, (dict, list)) else str(value))
    display = pd.concat([display, parameters], axis=1).sort_values("Selected", ascending=False, kind="stable")
    st.dataframe(display.round(3), hide_index=True, width="stretch", alt="Evaluated tuning configurations, selected rows first")


predictions = {}
try:
    prediction_paths = sorted((ROOT / "results/predictions").glob("*.csv"))
except OSError:
    prediction_paths = []
if not prediction_paths:
    unavailable("results/predictions")
for path in prediction_paths:
    frame = table(str(path.relative_to(ROOT)), ["date", "split", "model", "y_true", "y_prob", "y_pred"],
                  ["y_true", "y_prob", "y_pred"])
    if frame is None:
        continue
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True).dt.tz_localize(None)
    valid = (frame["date"].notna() & frame["y_true"].isin([0, 1])
             & frame["y_pred"].isin([0, 1]) & frame["y_prob"].between(0, 1))
    test_rows = frame.loc[frame["split"].eq("test")].sort_values("date")
    if not valid.all() or test_rows.empty or test_rows["date"].duplicated().any():
        unavailable(path.name)
        continue
    for model, rows in test_rows.groupby("model"):
        predictions[str(model)] = rows

if predictions:
    reference = predictions.get("logreg", next(iter(predictions.values())))
    st.success("All numbers and charts on this page are the real Deliverable 2 results "
               f"(test set: {len(reference)} days, {reference['date'].min():%d %b %Y} to "
               f"{reference['date'].max():%d %b %Y}, {reference['y_true'].mean():.2%} up days).")
    if any(not frame[["date", "y_true"]].reset_index(drop=True).equals(
            reference[["date", "y_true"]].reset_index(drop=True)) for frame in predictions.values()):
        st.caption("Test coverage differs between prediction files; the banner uses the first available reference model.")
else:
    st.info("Deliverable 2 test-set dates and up-rate are not available.")

overview, training, errors, samples = st.tabs(["Overview", "Training", "Errors and Bias", "Sample predictions"])

with overview:
    st.subheader("Model comparison (test set)")
    metrics = ["accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]
    comparison = table("results/tables/model_comparison.csv", ["model", "split", *metrics], metrics)
    if comparison is not None:
        test = comparison.loc[comparison["split"].eq("test")].drop_duplicates("model")
        val = comparison.loc[comparison["split"].eq("val")].drop_duplicates("model").set_index("model")
        if test.empty:
            unavailable("model_comparison.csv (test rows)")
        else:
            display = test[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
            display["AUC (val)"] = display["model"].map(val["roc_auc"])
            display["model"] = display["model"].map(friendly)
            display = display.rename(columns={"model": "Model", "accuracy": "Accuracy", "precision": "Precision",
                                              "recall": "Recall", "f1": "F1", "roc_auc": "AUC (test)"})
            st.dataframe(display.round(3), hide_index=True, width="stretch", alt="Test metrics with validation AUC")
            up_models = [m for m, rows in predictions.items() if rows["y_pred"].eq(1).any()]
            st.caption(f"Best test AUC: {winners(test, 'roc_auc')}. Most balanced (macro F1): "
                       f"{winners(test, 'f1_macro')}. Most reliable up calls (precision): "
                       f"{winners(test.loc[test['model'].isin(up_models)], 'precision')}.")
            baseline = test.loc[test["model"].eq("majority")]
            learned = test.loc[~test["model"].eq("majority")]
            note = "AUC 0.5 is guessing."
            if not baseline.empty:
                accuracy = baseline.iloc[0]["accuracy"]
                always_up = "majority" in predictions and predictions["majority"]["y_pred"].eq(1).all()
                note += f" The {'always-up' if always_up else 'majority'} baseline scores {accuracy:.2%} accuracy."
                if not learned.empty:
                    better = learned.loc[learned["accuracy"] > accuracy, "model"].map(friendly)
                    note += (" No learned model beats it on accuracy." if better.empty else
                             f" Models exceeding its accuracy: {', '.join(better)}.")
            st.caption(note)
    for column, name, caption in zip(st.columns(3),
            ["roc_test.png", "confusion_lstm.png", "confusion_logreg.png"],
            ["ROC curves on the test set", "LSTM confusion matrix", "Logistic regression confusion matrix"]):
        with column:
            show_image("evaluation/" + name, caption)
    with st.expander("Metrics chart"):
        show_image("evaluation/metrics_test.png", "Test-set model metrics")

with training:
    left, right = st.columns(2)
    with left:
        show_image("deep/lstm_loss_curve.png", "LSTM training and validation loss")
    with right:
        show_image("deep/lstm_accuracy_curve.png", "LSTM training and validation accuracy")
    with st.expander("LSTM architecture"):
        show_image("deep/lstm_architecture.png", "Saved LSTM architecture")
    history_columns = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    history = table("results/history/lstm_history.csv", history_columns, history_columns)
    if history is not None:
        history = history.sort_values("epoch")
        best, last = history.loc[history["val_loss"].idxmin()], history.iloc[-1]
        st.write(f"Best epoch: {best['epoch']:g}; epochs run: {len(history)}; best validation loss: {best['val_loss']:.6f}.")
        st.write(f"At the best epoch, train accuracy was {best['train_acc']:.2%} and validation accuracy "
                 f"was {best['val_acc']:.2%}. At the last epoch ({last['epoch']:g}), they were "
                 f"{last['train_acc']:.2%} and {last['val_acc']:.2%}.")
        if last["val_loss"] > best["val_loss"]:
            st.caption(f"Validation loss rose from {best['val_loss']:.6f} to {last['val_loss']:.6f} "
                       "after the best epoch, suggesting overfitting.")
    st.subheader("Classical models (grid search, validation AUC)")
    tuning("results/tuning/classical_tuning.csv")
    st.subheader("LSTM tuning")
    tuning("results/tuning/lstm_tuning.csv")
    importance, _ = st.columns(2)
    with importance:
        show_image("classical/feature_importance.png", "Saved classical-model feature importance; importance is not causation.")

with errors:
    st.subheader("Test-set prediction bias")
    rates = ["true_up_rate", "pred_up_rate", "gap", "majority_baseline_acc", "model_acc"]
    bias = table("results/analysis/bias_check.csv", ["model", "split", *rates], rates)
    if bias is not None:
        bias = bias.loc[bias["split"].eq("test")]
        if bias.empty:
            unavailable("bias_check.csv (test rows)")
        else:
            display = bias[["model", *rates]].copy()
            display["model"] = display["model"].map(friendly)
            for rate in rates:
                display[rate] = display[rate].map(lambda x: f"{x * 100:+.2f}" if rate == "gap" else f"{x:.2%}")
            display.columns = ["Model", "True up-rate", "Predicted up-rate", "Gap (points)",
                               "Majority-baseline accuracy", "Model accuracy"]
            st.dataframe(display, hide_index=True, width="stretch", alt="Test-set actual and predicted up-rates")
            collapsed = bias.loc[bias["pred_up_rate"].isin([0, 1])]
            for row in collapsed.itertuples():
                st.write(f"{friendly(row.model)} predicts {'up' if row.pred_up_rate == 1 else 'down'} on every test day.")
            if collapsed["model"].isin(["random_forest", "hist_gb"]).any():
                st.caption("A likely cause of the tree-model collapse is that price-level features lie far outside "
                           "the training range in 2024 to 2026, and tree models cannot extrapolate.")
    for column, name, caption in zip(st.columns(3),
            ["accuracy_by_year.png", "accuracy_by_volatility.png", "recall_by_class.png"],
            ["Accuracy by year", "Accuracy by volatility (test split)", "Recall by class (test split)"]):
        with column:
            show_image("analysis/" + name, caption)
    with st.expander("Test accuracy by volatility and outlier segment"):
        segments = table("results/analysis/segment_metrics.csv",
                         ["model", "split", "segment_type", "segment", "n", "accuracy"], ["n", "accuracy"])
        if segments is not None:
            selected = segments.loc[segments["split"].eq("test") & segments["segment_type"].isin(["volatility", "outlier"]),
                                    ["model", "segment", "n", "accuracy"]].copy()
            selected["model"] = selected["model"].map(friendly)
            st.dataframe(selected.round(3), hide_index=True, width="stretch", alt="Test accuracy by selected segment")

with samples:
    if predictions:
        options = list(predictions)
        model = st.selectbox("Model", options, index=options.index("logreg") if "logreg" in options else 0,
                             format_func=friendly, key="insights_model")
        recent = predictions[model].tail(120).copy()
        recent["Actual"] = recent["y_true"].map({0: "Down", 1: "Up"})
        recent["Predicted"] = recent["y_pred"].map({0: "Down", 1: "Up"})
        base = alt.Chart(recent).encode(
            x=alt.X("date:T", title=None),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                     alt.Tooltip("y_prob:Q", title="Probability of up", format=".1%"), "Actual:N", "Predicted:N"])
        line = base.mark_line(color=AMBER).encode(
            y=alt.Y("y_prob:Q", title="Chance of up / actual outcome", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")))
        points = base.mark_point(filled=True, size=35).encode(
            y="y_true:Q", color=alt.Color("Actual:N", scale=alt.Scale(domain=["Up", "Down"], range=[GREEN, RED]),
                                         legend=alt.Legend(title=None, orient="bottom")))
        threshold = alt.Chart(pd.DataFrame({"threshold": [0.5]})).mark_rule(color=GREY, strokeDash=[5, 5]).encode(y="threshold:Q")
        st.altair_chart(styled(line + threshold + points).interactive(), width="stretch", theme=None,
                        alt=f"{friendly(model)} probabilities and actual outcomes on the last {len(recent)} test days")
        st.caption("Line: predicted chance of up. Dots: what really happened (green = up, red = down). "
                   "Above the dashed 50% line the model predicts up.")
        st.write(f"{friendly(model)} accuracy over these {len(recent)} test days: "
                 f"{recent['y_true'].eq(recent['y_pred']).mean():.2%}.")
    else:
        unavailable("Sample predictions")
