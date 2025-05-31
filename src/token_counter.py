import pandas as pd
import tiktoken
import os

def count_tokens(
    input_csv,
    output_csv,
    debug=False,
    encoding_name="cl100k_base",
    preview_rows=5
):
    """
    Counts input and output tokens for each message in a filled conversations CSV
    and writes the result as a new CSV.

    Args:
        input_csv (str): Path to the filled CSV.
        output_csv (str): Path for the output CSV.
        debug (bool): If True, print debug info.
        encoding_name (str): tiktoken encoding name (default: cl100k_base).
        preview_rows (int): Number of rows to print debug info for.

    Returns:
        pd.DataFrame: The resulting token counts table.
    """

    # --- Ensure output directory exists ---
    out_dir = os.path.dirname(output_csv)
    if not os.path.isdir(out_dir):
        if debug:
            print(f"[Debug] Creating output directory: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

    # --- Load the filled conversations table ---
    df = pd.read_csv(input_csv, dtype=str)

    # --- Prepare the tokenizer ---
    enc = tiktoken.get_encoding(encoding_name)

    # --- Iterate and count tokens ---
    records = []
    for idx, row in df.iterrows():
        conv_id = row.get("conversation_id", "")
        msg_id  = row.get("message_id", "")
        conv_ct = row.get("conversation_create_time", "")
        model   = (row.get("model") or "").strip().lower()
        role    = row.get("role", "")
        raw     = row.get("content", None)
        content = "" if pd.isna(raw) else str(raw)

        inp_toks, out_toks = 0, 0
        if model and model not in ("auto", "research", "unknown"):
            toks = enc.encode(content)
            n = len(toks)
            if role == "user":
                inp_toks = n
            else:
                out_toks = n
            if debug and idx < preview_rows:
                print(f"[Debug] idx={idx} role={role!r} model={model!r} tokens={n}")
        else:
            if debug and idx < preview_rows:
                print(f"[Debug] idx={idx} role={role!r} model={model!r} → tokens=0")

        records.append({
            "conversation_id": conv_id,
            "message_id": msg_id,
            "conversation_create_time": conv_ct,
            "input_tokens": inp_toks,
            "output_tokens": out_toks,
            "model": model
        })

    # --- Create DataFrame and attempt to save ---
    out_df = pd.DataFrame(records)
    try:
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"✅ Wrote {len(out_df)} token records to {output_csv}")
    except Exception as e:
        print(f"❌ Failed to write CSV to {output_csv}: {e}")

    return out_df

# === CLI/Script usage ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Count tokens per message in filled conversations CSV.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to filled conversations CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Output path for token counts CSV")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--encoding", type=str, default="cl100k_base")
    parser.add_argument("--preview_rows", type=int, default=5)
    args = parser.parse_args()

    count_tokens(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        debug=args.debug,
        encoding_name=args.encoding,
        preview_rows=args.preview_rows
    )
