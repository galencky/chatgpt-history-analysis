import json
import pandas as pd
from pathlib import Path
import os

def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize empty strings and 'nan' to pandas NA for all object columns."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .replace({"": pd.NA, "nan": pd.NA})
            )
    return df

FINAL_ORDER = [
    "conversation_id", "message_id", "parent_id", "role", "type",
    "conversation_create_time", "create_time", "update_time", "model",
    "conversation_title", "content", "summary", "end_turn", "recipient",
    "status", "weight"
]

def extract_image_records(conversations):
    """
    Extracts image generation/upload records from conversations.
    Returns a DataFrame in FINAL_ORDER column order.
    """
    records = []
    for conv_idx, conv in enumerate(conversations):
        cid   = conv.get("id", "")
        ctime = conv.get("create_time", pd.NA)
        title = conv.get("title", "")
        mapping = conv.get("mapping", {}) or {}
        if not isinstance(mapping, dict):
            continue

        for mid, node in mapping.items():
            msg = node.get("message") or {}
            if not isinstance(msg, dict):
                continue

            # common fields
            role   = msg.get("author", {}).get("role", pd.NA)
            model  = msg.get("metadata", {}).get("model_slug", pd.NA)
            c_t    = msg.get("create_time", pd.NA)
            u_t    = msg.get("update_time", pd.NA)
            parent = node.get("parent", pd.NA)

            content = msg.get("content") or {}
            if content.get("content_type") != "multimodal_text":
                continue

            for part in content.get("parts", []):
                if not (isinstance(part, dict) and part.get("content_type") == "image_asset_pointer"):
                    continue

                pm    = part.get("metadata", {}) or {}
                gen   = pm.get("generation", {}) or {}
                dalle = pm.get("dalle", {}) or {}

                img_type = "image_generation" if gen.get("gen_id") else "image_upload"

                # Flatten all relevant image metadata into one dict
                img_meta = {
                    "image_gen_title":     msg.get("metadata", {}).get("image_gen_title", pd.NA),
                    "asset_pointer":       part.get("asset_pointer", pd.NA),
                    "width":               part.get("width", gen.get("width", pd.NA)),
                    "height":              part.get("height", gen.get("height", pd.NA)),
                    "gen_id":              gen.get("gen_id", pd.NA),
                    "serialization_title": dalle.get("serialization_title", pd.NA),
                }

                records.append({
                    "conversation_id":          cid,
                    "message_id":               mid,
                    "parent_id":                parent,
                    "role":                     role,
                    "type":                     img_type,
                    "conversation_create_time": ctime,
                    "create_time":              c_t,
                    "update_time":              u_t,
                    "model":                    model,
                    "conversation_title":       title,
                    "content":                  img_meta,
                    "summary":                  pd.NA,
                    "end_turn":                 msg.get("end_turn", pd.NA),
                    "recipient":                msg.get("recipient", pd.NA),
                    "status":                   msg.get("status", pd.NA),
                    "weight":                   msg.get("weight", pd.NA),
                })

        # Print progress for large exports
        if (conv_idx + 1) % 100 == 0 or conv_idx == len(conversations) - 1:
            print(f"  Processed {conv_idx + 1} / {len(conversations)} conversations for image extraction")

    df = pd.DataFrame(records)
    df = normalize_missing(df)
    for col in FINAL_ORDER:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[FINAL_ORDER]
    print(f"✅ Extracted {len(df)} image rows")
    return df

if __name__ == "__main__":
    # Script usage: ask for path to conversations.json and output CSV
    in_path = input("Enter path to conversations.json: ").strip()
    out_csv = "data/image_generations.csv"
    if not os.path.isfile(in_path):
        print(f"File not found: {in_path}")
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        df = extract_image_records(conversations)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} rows to {out_csv}")

# Jupyter notebook usage:
# df = extract_image_records(conversations)
# df.to_csv("data/image_generations.csv", index=False, encoding="utf-8-sig")
# from IPython.display import display
# display(df)
