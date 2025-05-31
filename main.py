import os
import sys
import time
import shutil
import logging
import json
from datetime import datetime
from pathlib import Path
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import zipfile

# ───────────────────── Paths and Constants ─────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
WATCH_DIR     = Path(base_dir, "drop_zip_here")
OUTPUT_PARENT = Path(base_dir, "output")
DATA_DIR      = Path(base_dir, "data")
for p in (WATCH_DIR, OUTPUT_PARENT, DATA_DIR):
    p.mkdir(exist_ok=True)

STABILITY_SECONDS = 15
POLL_INTERVAL     = 5

# put this near the top of main.py
INBOX_DIR   = WATCH_DIR / "_inbox"
PROCESSED   = set()                       # {(name, size)}

INBOX_DIR.mkdir(exist_ok=True)            # ensure it exists

# ───────────────────── Logging Setup ─────────────────────
log_path = DATA_DIR / "logs.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
)
def log(msg, level=logging.INFO):
    logging.log(level, msg)

# ───────────────────── Watchdog Handler ─────────────────────
class ZipReadyHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self._found = None
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(".zip"):
            self._found = Path(event.src_path)
            log(f"🛈 Detected new zip: {self._found.name}")


def await_first_zip() -> Path:
    """
    Block until a .zip appears in WATCH_DIR, is stable for STABILITY_SECONDS,
    then MOVE it to WATCH_DIR/_inbox and return that new path.
    Duplicate-trigger safe.
    """
    handler  = ZipReadyHandler()
    observer = PollingObserver(timeout=POLL_INTERVAL)
    observer.schedule(handler, str(WATCH_DIR), recursive=False)
    observer.start()

    try:
        candidate: Path | None = None

        while True:
            # ─── 1. did watchdog report something new? ───
            if handler._found and candidate is None:
                candidate = handler._found
                log(f"🛈 Found {candidate.name}, waiting for it to stabilise…")

            # ─── 2. if we have a candidate, poll its size ───
            if candidate and candidate.exists():
                size_a = candidate.stat().st_size
                time.sleep(STABILITY_SECONDS)
                if candidate.exists() and candidate.stat().st_size == size_a:
                    sig = (candidate.name, size_a)
                    if sig in PROCESSED:                # already handled once
                        log(f"↪︎ Duplicate event for {candidate.name} – ignored")
                        candidate = None                 # reset and keep watching
                        handler._found = None
                        continue

                    # move out of watched dir BEFORE returning
                    target = INBOX_DIR / candidate.name
                    shutil.move(candidate, target)
                    PROCESSED.add(sig)

                    log(f"✅ {candidate.name} stable → moved to {target}; starting analysis.")
                    observer.stop()
                    return target

                # still growing – keep watching the same candidate
                continue

            time.sleep(POLL_INTERVAL)

    finally:
        observer.stop()
        observer.join()

# ───────────────────── Import pipeline modules ─────────────────────
sys.path.append("src")
from import_export_zip import prepare_export_and_load_conversations
from survey_schema       import survey_conversation_keys
from flatten_messages    import run_flatten_and_sample
from flatten_websearch   import extract_flattened_data
from flatten_images      import extract_image_records
from merge_flattened     import merge_all
from fill_model_names    import fill_model_names
from analyze_model_usage import analyze_model_usage
from token_counter       import count_tokens
from emulate_api_chat_costs          import main as emulate_api_chat_costs
from plot_monthly_summary            import plot_monthly_summary
from plot_token_costs_comparison     import main as plot_token_costs_comparison
from send_email_report               import send_email_report

# ───────────────────── Core Pipeline ─────────────────────
def run_pipeline():
    log("\n=== ChatGPT History Analysis Pipeline ===\n")

    zip_path = await_first_zip()

    conversations, folder = prepare_export_and_load_conversations(
        base_dir=base_dir,
        zip_path=str(zip_path)
    )

    # 1. Survey
    survey_conversation_keys(conversations)

    # 2. Flatten
    run_flatten_and_sample( conversations, DATA_DIR / "conversations_flat.csv", show_sample=False)
    extract_flattened_data(conversations).to_csv(DATA_DIR / "flattened_websearch_thoughts.csv", index=False, encoding="utf-8-sig")
    extract_image_records(conversations).to_csv(DATA_DIR / "image_generations.csv", index=False, encoding="utf-8-sig")

    # 3. Merge + fill
    merge_all(
        DATA_DIR / "conversations_flat.csv",
        DATA_DIR / "flattened_websearch_thoughts.csv",
        DATA_DIR / "image_generations.csv",
        DATA_DIR / "merged_conversations.csv",
        show_df=False
    )
    with open(DATA_DIR / folder / "conversations.json", encoding="utf-8") as jf:
        fill_model_names(
            DATA_DIR / "merged_conversations.csv",
            json.load(jf),
            DATA_DIR / "merged_conversations_filled.csv",
            debug=False
        )

    # 4. Stats, tokens, costs
    filled_csv = DATA_DIR / "merged_conversations_filled.csv"
    analyze_model_usage(filled_csv, show_table=True)
    count_tokens(filled_csv, DATA_DIR / "token_counts.csv")
    emulate_api_chat_costs(DATA_DIR / "token_counts.csv", DATA_DIR / "token_costs_true_api_emulated.csv")

    # 5. Plots
    plot_monthly_summary(merged_csv_path=filled_csv, output_dir=DATA_DIR)
    plot_token_costs_comparison()

    # ────────────── FINISHING TOUCHES ──────────────
    run_outdir = OUTPUT_PARENT / f"analysis-{datetime.now():%Y%m%d-%H%M%S}"
    run_outdir.mkdir(parents=True, exist_ok=True)

    # Move original zip file to output folder
    shutil.move(str(zip_path), run_outdir / zip_path.name)

    # Remove all decompressed folders like chatgpt-*
    for item in DATA_DIR.iterdir():
        if item.is_dir() and item.name.lower().startswith("chatgpt"):
            shutil.rmtree(item, ignore_errors=True)
            log(f"🗑️  Removed folder: {item.name}")

    # Build results.zip (excluding any .zip files)
    results_zip = run_outdir / "results.zip"
    import zipfile
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DATA_DIR):
            for f in files:
                if f.lower().endswith(".zip"):
                    continue
                p = Path(root) / f
                zf.write(p, arcname=p.relative_to(DATA_DIR))
    log(f"📦 Packed results → {results_zip.name}")

    # Flush logging and snapshot log
    for h in logging.getLogger().handlers:
        h.flush()
    shutil.copy2(log_path, run_outdir / "logs.txt")

    # Copy key deliverables into output folder
    for png in DATA_DIR.glob("*.png"):
        shutil.copy2(png, run_outdir / png.name)

    usage_csv = DATA_DIR / "model_usage_frequency.csv"
    if usage_csv.exists():
        shutil.copy2(usage_csv, run_outdir / usage_csv.name)

    # Send email with final results
    send_email_report(
        output_dir=str(run_outdir),
        log_filename="logs.txt",
        usage_csv="model_usage_frequency.csv"
    )

    # ─── Clean up any old residual CSV/PNG from output folder ───
    for ext in ("*.csv", "*.png"):
        for f in run_outdir.glob(ext):
            f.unlink(missing_ok=True)

    # ─── FINAL CLEANUP: wipe all of /data and recreate fresh logs.txt ───
    for item in DATA_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item, ignore_errors=True)
        except Exception as e:
            log(f"⚠️ Failed to remove {item}: {e}")
    open(log_path, "w").close()  # recreate empty log file
    log("🧽 Cleared data directory for next run")

    print(f"🎉 Pipeline complete! Output archived to: {run_outdir.resolve()}\n")

# ─── Remove any EMPTY analysis folders that might be left behind ───
for d in OUTPUT_PARENT.glob("analysis-*"):
    if not any(d.iterdir()):                      # directory is empty
        shutil.rmtree(d, ignore_errors=True)
       # log(f"🗑️  Removed empty output folder: {d.name}")

# ───────────────────── Entry Point with Loop ─────────────────────

if __name__ == "__main__":
    log("🚀 Watching for ChatGPT export ZIPs...\n")
    while True:
        try:
            run_pipeline()
            print("🕐 Waiting for next zip...\n")
        except Exception as e:
            log(f"❌ Error during pipeline: {e}", level=logging.ERROR)
        time.sleep(3)
