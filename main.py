"""Run the whole ProcurePulse data pipeline with one command.

    python main.py                   # all 7 steps, in order
    python main.py --skip-download   # reuse data/raw/*.csv, run steps 4-7 only
    python main.py --from 5          # start at step 5 (cleaning)
    python main.py --list            # show the steps without running them

Each step runs as its own Python process, exactly as if you had typed
"python data_scripts/<script>.py", so every script keeps its own log file.
The run stops at the first step that fails.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "data_scripts"

STEPS = [
    ("fetch_metals.py", "Download COMEX metal futures (needs internet)"),
    ("fetch_external.py", "Download USD index, oil, 10-year yield and CPI (needs internet)"),
    ("merge_sources.py", "Merge all series into data/raw/merged_raw.csv"),
    ("plot_raw.py", "Draw the raw-data charts"),
    ("clean_data.py", "Clean the data -> data/processed/cleaned.csv"),
    ("feature_engineering.py", "Build features, split and scale -> data/processed/"),
    ("eda.py", "Draw the EDA charts"),
]
FIRST_OFFLINE_STEP = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ProcurePulse data pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from", dest="start", type=int, choices=range(1, len(STEPS) + 1),
                       metavar="N", help="start at step N (1-7) instead of step 1")
    group.add_argument("--skip-download", action="store_true",
                       help="reuse the downloaded data and start at step 4")
    parser.add_argument("--list", action="store_true", help="list the steps and exit")
    return parser.parse_args()


def run_step(number, script, description):
    print(f"\n{'=' * 64}\nStep {number}/{len(STEPS)}: {script}\n{description}\n{'=' * 64}", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, str(SCRIPTS / script)], cwd=ROOT)
    seconds = time.time() - started
    status = "done" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"--> {script} {status} in {seconds:.0f}s", flush=True)
    return result.returncode == 0, seconds


def main():
    args = parse_args()
    start = FIRST_OFFLINE_STEP if args.skip_download else (args.start or 1)

    if args.list:
        for number, (script, description) in enumerate(STEPS, start=1):
            print(f"{number}. {script:<24} {description}")
        return 0

    timings = []
    for number, (script, description) in enumerate(STEPS, start=1):
        if number < start:
            continue
        ok, seconds = run_step(number, script, description)
        timings.append((script, seconds))
        if not ok:
            print(f"\nPipeline stopped at step {number} ({script}). Fix the error above, then "
                  f"continue with: python main.py --from {number}")
            return 1

    total = sum(seconds for _, seconds in timings)
    print(f"\nAll {len(timings)} step(s) finished in {total / 60:.1f} min:")
    for script, seconds in timings:
        print(f"  {script:<24} {seconds:6.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
