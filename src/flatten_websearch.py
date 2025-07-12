import json
import pandas as pd
import os

# ===[ Consistent Output Columns ]===
COLUMN_ORDER = [
    "conversation_id", "message_id", "parent_id", "role", "type",
    "conversation_create_time", "create_time", "update_time", "model",
    "conversation_title", "content", "summary",
    "end_turn", "recipient", "status", "weight"
]

def extract_flattened_data(conversations):
    """
    Extracts 'thought', 'search_query', code-based queries, quotes,
    and web references from each conversation message into a DataFrame.
    """
    rows = []
    for conv_idx, conv in enumerate(conversations):
        conv_id = conv.get("id", "")
        title = conv.get("title", "")
        conv_time = conv.get("create_time", "")
        mapping = conv.get("mapping", {})
        if not isinstance(mapping, dict):
            continue

        for msg_id, node in mapping.items():
            msg = node.get("message", {})
            if not isinstance(msg, dict):
                continue

            # --- Common fields for every row ---
            role = msg.get("author", {}).get("role", "")
            model = msg.get("metadata", {}).get("model_slug", "")
            ctime = msg.get("create_time")
            utime = msg.get("update_time")
            parent = node.get("parent")
            end_turn = msg.get("end_turn")
            recipient = msg.get("recipient")
            status = msg.get("status")
            weight = msg.get("weight")
            metadata = msg.get("metadata", {})
            content_obj = msg.get("content", {})
            if not isinstance(content_obj, dict):
                continue

            # Build the base dict once per message
            base = {
                "conversation_id": conv_id,
                "message_id": msg_id,
                "parent_id": parent,
                "role": role,
                "type": None,
                "conversation_create_time": conv_time,
                "create_time": ctime,
                "update_time": utime,
                "model": model,
                "conversation_title": title,
                "content": None,
                "summary": None,
                "end_turn": end_turn,
                "recipient": recipient,
                "status": status,
                "weight": weight
            }

            # 1) Assistant Thought Blocks
            if content_obj.get("content_type") == "thoughts":
                thoughts = content_obj.get("thoughts", [])
                if thoughts is not None:
                    for t in thoughts:
                        row = base.copy()
                        row.update({
                            "content": t.get("content"),
                            "type":    "thought",
                            "summary": t.get("summary")
                        })
                        rows.append(row)

            # 2) Search Queries from metadata
            search_queries = metadata.get("search_queries", [])
            if search_queries is not None:
                for q in search_queries:
                    row = base.copy()
                    row.update({
                        "content": q.get("q", ""),
                        "type":    "search_query"
                    })
                    rows.append(row)

            # 3) Code-embedded operations (search_query & open_url)
            if content_obj.get("content_type") == "code":
                try:
                    parsed = json.loads(content_obj.get("text", "{}"))
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                if isinstance(parsed, dict):
                    search_query = parsed.get("search_query", [])
                    if search_query is not None:
                        for q in search_query:
                            row = base.copy()
                            row.update({
                                "content": q.get("q", ""),
                                "type":    "search_query"
                            })
                            rows.append(row)
                    open_urls = parsed.get("open", [])
                    if open_urls is not None:
                        for o in open_urls:
                            row = base.copy()
                            row.update({
                                "content": o.get("ref_id", ""),
                                "type":    "open_url"
                            })
                            rows.append(row)

            # 4) Tether Quotes
            if content_obj.get("content_type") == "tether_quote":
                row = base.copy()
                row.update({
                    "content": content_obj.get("text", ""),
                    "type":    "tether_quote",
                    "summary": content_obj.get("title", content_obj.get("domain", ""))
                })
                rows.append(row)

            # 5) Extended webpage references
            if content_obj.get("content_type") == "webpage_extended":
                content_refs = content_obj.get("content_references", [])
                if content_refs is not None:
                    for ref in content_refs:
                        row = base.copy()
                        row.update({
                            "content": ref.get("snippet", ""),
                            "type":    "webpage_extended",
                            "summary": ref.get("attribution", "")
                        })
                        rows.append(row)

        if (conv_idx + 1) % 100 == 0 or conv_idx == len(conversations) - 1:
            print(f"  Processed {conv_idx + 1} / {len(conversations)} conversations for web/thought/code extraction")

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]
    print(f"✅ Extracted {len(df)} web/thought/code rows")
    return df

if __name__ == "__main__":
    # For direct script usage:
    in_path = input("Enter path to conversations.json: ").strip()
    out_csv = "data/flattened_websearch_thoughts.csv"
    if not os.path.isfile(in_path):
        print(f"File not found: {in_path}")
    else:
        with open(in_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        df = extract_flattened_data(conversations)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Saved to {out_csv}")

# Jupyter usage example:
# df = extract_flattened_data(conversations)
# df.to_csv("data/flattened_websearch_thoughts.csv", index=False, encoding="utf-8-sig")
# display(df)
