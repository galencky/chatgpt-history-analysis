import pandas as pd
from datetime import datetime
import os

# === Normalize helper: "" or "nan" → <NA> ===
def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Convert empty string and 'nan' to <NA> for all object columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                  .astype(str)
                  .str.strip()
                  .replace({"": pd.NA, "nan": pd.NA})
            )
    return df

# === Dedupe each DF on message_id by the longest non-null string per column ===
def choose_longest(series: pd.Series):
    """Return longest non-null string from a Series, or <NA> if all NA."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return pd.NA
    return max(non_null, key=len)

def dedupe_on_message_id(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate on message_id by taking longest string per column."""
    return df.groupby("message_id", as_index=False).agg(choose_longest)

# === Choose longer/non-null value for merging ===
def choose_better(v1, v2):
    """Choose the longer string or the non-null value for merging."""
    if pd.isna(v1): return v2
    if pd.isna(v2): return v1
    return v1 if len(str(v1)) >= len(str(v2)) else v2

# === Robust merge_two: outer-merge and resolve overlaps ===
def merge_two(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-merge two DataFrames and resolve column conflicts using choose_better.
    Handles missing columns gracefully.
    """
    merged = pd.merge(left, right, on="message_id", how="outer",
                      suffixes=("_1", "_2"))
    cols_left  = set(left.columns)  - {"message_id"}
    cols_right = set(right.columns) - {"message_id"}
    all_cols   = sorted(cols_left | cols_right)

    out = pd.DataFrame({"message_id": merged["message_id"]})
    for col in all_cols:
        c1, c2 = f"{col}_1", f"{col}_2"
        if col in left.columns and col in right.columns:
            # If both have this col, pick the longer value
            out[col] = merged.apply(
                lambda row: choose_better(row.get(c1), row.get(c2)),
                axis=1
            )
        elif col in left.columns:
            out[col] = merged[c1] if c1 in merged.columns else pd.NA
        elif col in right.columns:
            out[col] = merged[c2] if c2 in merged.columns else pd.NA
        else:
            out[col] = pd.NA  # fallback, should not occur

    return normalize_missing(out)

# === Timestamp formatting helper (epoch → YYYYMMDD_HHMMSS.cc) ===
def safe_format_ts(v):
    """Convert float timestamp to 'YYYYMMDD_HHMMSS.cc' or <NA> if not valid."""
    if pd.isna(v):
        return pd.NA
    try:
        ts = float(v)
    except:
        return pd.NA
    dt = datetime.fromtimestamp(ts)
    centis = dt.microsecond // 10000
    return dt.strftime("%Y%m%d_%H%M%S") + f".{centis:02d}"

# === Merge all three sources: main, websearch, images ===
def merge_all(csv1, csv2, csv3, output_csv, show_df=True):
    """
    Merge three flattened CSVs (main, websearch, images) into a single DataFrame and write to disk.
    """
    print("🔄 Loading CSVs...")
    df1 = pd.read_csv(csv1, dtype=str)
    df2 = pd.read_csv(csv2, dtype=str)
    df3 = pd.read_csv(csv3, dtype=str)
    print("  - Loaded all sources.")

    df1 = dedupe_on_message_id(normalize_missing(df1))
    df2 = dedupe_on_message_id(normalize_missing(df2))
    df3 = dedupe_on_message_id(normalize_missing(df3))
    print("  - Deduplication complete.")

    # Merge in two stages: (main + websearch) → then + images
    df12   = merge_two(df1, df2)
    result = merge_two(df12, df3)
    print("  - Merging complete.")

    # Reorder columns and fill missing
    final_order = [
        "conversation_id", "message_id", "parent_id", "role", "type",
        "conversation_create_time", "create_time", "update_time", "model",
        "conversation_title", "content", "summary", "end_turn", "recipient",
        "status", "weight"
    ]
    for col in final_order:
        if col not in result.columns:
            result[col] = pd.NA
    result = result[final_order]

    # Timestamp formatting
    for tc in ["conversation_create_time", "create_time", "update_time"]:
        result[tc] = result[tc].map(safe_format_ts)

    # Sort, clean, and save
    result = result.sort_values("conversation_create_time", ascending=False, ignore_index=True)
    result.replace("nan", pd.NA, inplace=True)
    result.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ Merged {len(result)} rows into {output_csv}")

    if show_df:
        try:
            from IPython.display import display
            display(result)
        except ImportError:
            print("(display skipped; not in Jupyter)")
    return result

# === CLI usage ===
if __name__ == "__main__":
    # Script usage: will prompt for any missing CSV paths
    default_dir = "data"
    csv1 = os.path.join(default_dir, "conversations_flat.csv")
    csv2 = os.path.join(default_dir, "flattened_websearch_thoughts.csv")
    csv3 = os.path.join(default_dir, "image_generations.csv")
    output_csv = os.path.join(default_dir, "merged_conversations.csv")

    for path in [csv1, csv2, csv3]:
        if not os.path.isfile(path):
            path = input(f"CSV not found: {path}\nPlease enter the path: ").strip()
    merge_all(csv1, csv2, csv3, output_csv)
