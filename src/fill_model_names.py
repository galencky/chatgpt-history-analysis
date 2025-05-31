import pandas as pd
import json
import re
from collections import defaultdict

def fill_model_names(
    merged_csv_path="data/merged_conversations.csv",
    conversations_json=None,
    output_csv_path="data/merged_conversations_filled.csv",
    usage_csv_path="data/model_usage_frequency.csv",
    debug=False
):
    """
    Fills missing or placeholder model names in merged conversations CSV by looking up
    model information in the original conversations JSON.
    Writes a new CSV with improved 'model' column and saves model usage frequency table as CSV.
    The frequency table is sorted chronologically by first use (oldest to most recent),
    and includes the first use timestamp per model.
    """
    # --- 1) Load merged CSV ---
    df = pd.read_csv(merged_csv_path, dtype=str)

    # --- 2) Prepare regex for known model names ---
    model_patterns = [
        "text-davinci-002-render-sha",
        "gpt-4", "gpt-4-mobile", "gpt-4-browsing", "gpt-4-plugins",
        "text-davinci-002-render-sha-mobile", "gpt-4-gizmo",
        "gpt-3.5-turbo", "gpt-4o", "o1-preview", "gpt-4o-canmore",
        "o1", "o1-mini", "o3-mini", "o3-mini-high", "gpt-4-5",
        "gpt-4o-mini", "o3", "o4-mini-high", "gpt-4-1", "o4-mini"
    ]
    pattern_re = re.compile("|".join(model_patterns), re.IGNORECASE)

    # --- 3) Build lookup: message_id → model from JSON metadata ---
    metadata_lookup = {}
    for conv in conversations_json:
        for msg_id, entry in (conv.get("mapping") or {}).items():
            if not isinstance(entry, dict):
                continue
            msg = entry.get("message")
            if not isinstance(msg, dict):
                continue
            md = msg.get("metadata") or {}
            for key in ("model_slug", "default_model_slug", "model"):
                val = md.get(key)
                if val and pattern_re.search(val):
                    metadata_lookup[msg_id] = val
                    if debug:
                        print(f"[Metadata lookup] msg_id={msg_id} -> model='{val}'")
                    break

    # --- 4) First pass fill from JSON ---
    df["model_filled"] = df["model"].fillna("unknown")
    first_pass_counts = defaultdict(int)

    for idx, row in df.iterrows():
        original = row["model_filled"]
        if original in ("auto", "research", "unknown"):
            mid = row["message_id"]
            if mid in metadata_lookup:
                new_model = metadata_lookup[mid]
                df.at[idx, "model_filled"] = new_model
                first_pass_counts[new_model] += 1
                if debug:
                    print(f"[First pass] idx={idx}, msg_id={mid}, '{original}' -> '{new_model}'")

    # --- 5) Second pass fallback: fill within conversations by mode ---
    fallback_counts = defaultdict(int)
    for conv_id, group in df.groupby("conversation_id"):
        # identify known models in this conversation
        known = group.loc[
            ~group["model_filled"].isin(("auto", "research", "unknown")),
            "model_filled"
        ]
        if not known.empty:
            mode_model = known.mode()[0]
            for idx in group.index:
                if df.at[idx, "model_filled"] in ("auto", "research", "unknown"):
                    df.at[idx, "model_filled"] = mode_model
                    fallback_counts[mode_model] += 1
                    if debug:
                        print(f"[Fallback] idx={idx}, conv_id={conv_id}, filled -> '{mode_model}'")

    # --- 6) Replace original model column ---
    df["model"] = df["model_filled"]
    df.drop(columns="model_filled", inplace=True)

    # --- 7) Report filling statistics ---
    print("\nFirst-pass fills from JSON metadata:")
    for model, cnt in first_pass_counts.items():
        print(f"  {model}: {cnt}")

    print("\nSecond-pass fallback fills:")
    for model, cnt in fallback_counts.items():
        print(f"  {model}: {cnt}")

    remaining = df["model"].isin(["auto", "research", "unknown"]).sum()
    print(f"\nRemaining unfilled rows: {remaining}")

    # --- 8) Save updated CSV ---
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved updated CSV to {output_csv_path}")

    # --- 9) Calculate and save model frequency table ---
    # Parse 'create_time' to datetime for sorting and first-use calculation
    df['create_time_parsed'] = pd.to_datetime(
        df['create_time'], format='%Y%m%d_%H%M%S.%f', errors='coerce'
    )

    # Group by model
    usage_stats = (
        df.groupby('model', as_index=False)
        .agg(
            frequency=('model', 'size'),
            first_use=('create_time_parsed', 'min')
        )
        .sort_values('first_use', ascending=True)  # Chronological, oldest first
        .reset_index(drop=True)
    )

    # Add total frequency as the last row (optional)
    total_row = pd.DataFrame({
        'model': ['TOTAL'],
        'frequency': [usage_stats['frequency'].sum()],
        'first_use': [pd.NaT]
    })
    usage_stats = pd.concat([usage_stats, total_row], ignore_index=True)

    # Save the frequency/chronology table
    usage_stats.to_csv(usage_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved model usage frequency CSV to {usage_csv_path}")

    return df, usage_stats

# === CLI/Script usage ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fill model names in merged conversations CSV.")
    parser.add_argument("--merged_csv", type=str, default="data/merged_conversations.csv")
    parser.add_argument("--conversations_json", type=str, default=None, required=True)
    parser.add_argument("--output_csv", type=str, default="data/merged_conversations_filled.csv")
    parser.add_argument("--usage_csv", type=str, default="data/model_usage_frequency.csv")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Load conversations.json
    with open(args.conversations_json, "r", encoding="utf-8") as f:
        conversations_json = json.load(f)

    fill_model_names(
        merged_csv_path=args.merged_csv,
        conversations_json=conversations_json,
        output_csv_path=args.output_csv,
        usage_csv_path=args.usage_csv,
        debug=args.debug
    )
