import os
import pandas as pd
import traceback

# ===[ Unified Column Order for Output ]===
COLUMN_ORDER = [
    "conversation_id", "message_id", "parent_id", "role", "type",
    "conversation_create_time", "create_time", "update_time",
    "model", "conversation_title", "content", "summary",
    "role_type", "end_turn", "recipient", "status", "weight"
]

def extract_text_from_parts(parts):
    """
    Extracts visible text from a list of parts.
    Handles both strings and dicts with common text keys.
    """
    def extract(p):
        if isinstance(p, str):
            return p
        if isinstance(p, dict):
            for key in ("text", "value", "caption", "content"):
                if key in p and isinstance(p[key], str):
                    return p[key]
        return ""
    return "\n".join(extract(p) for p in parts).strip()

def flatten_all_messages_to_df(conversations, error_log_path="flat_error.txt"):
    """
    Flattens a list of conversation dicts into a pandas DataFrame.
    Returns (DataFrame, error_count, error_conversation_count).
    Writes errors to error_log_path.
    """
    rows = []
    error_logs = []
    errored_conversations = set()

    for conv_index, conv in enumerate(conversations):
        conv_id = conv.get("id", f"unknown_{conv_index}")
        try:
            title = conv.get("title", "")
            default_model = conv.get("default_model_slug", "")
            conv_time = conv.get("create_time", "")
            mapping = conv.get("mapping", {})

            if not isinstance(mapping, dict):
                error_logs.append(f"[Conversation {conv_id}] mapping is not a dict\n")
                errored_conversations.add(conv_id)
                continue

            for node_id, node in mapping.items():
                try:
                    message = node.get("message")
                    if not isinstance(message, dict):
                        continue

                    # Flatten fields
                    msg_id = message.get("id", "")
                    parent_id = node.get("parent", "")
                    role = message.get("author", {}).get("role", "")
                    create_time = message.get("create_time", "")
                    update_time = message.get("update_time", "")
                    model = message.get("metadata", {}).get("model_slug", default_model)
                    parts = message.get("content", {}).get("parts", [])
                    content = extract_text_from_parts(parts)
                    end_turn = message.get("end_turn", "")
                    recipient = message.get("recipient", "")
                    status = message.get("status", "")
                    weight = message.get("weight", "")

                    rows.append({
                        "conversation_id": conv_id,
                        "message_id": msg_id,
                        "parent_id": parent_id,
                        "role": role,
                        "conversation_create_time": conv_time,
                        "create_time": create_time,
                        "update_time": update_time,
                        "model": model,
                        "conversation_title": title,
                        "content": content,
                        "type": "message",
                        "summary": None,
                        "end_turn": end_turn,
                        "recipient": recipient,
                        "status": status,
                        "weight": weight,
                        "role_type": f"{role}_message"
                    })

                except Exception as e:
                    errored_conversations.add(conv_id)
                    error_logs.append(f"[Conv {conv_id} - Node {node_id}] {e}\n" + traceback.format_exc())
        except Exception as e:
            errored_conversations.add(conv_id)
            error_logs.append(f"[Conv {conv_id}] {e}\n" + traceback.format_exc())

    if error_logs:
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Total errors: {len(error_logs)}\nErrored IDs:\n")
            f.writelines(f" - {cid}\n" for cid in sorted(errored_conversations))
            f.write("Tracebacks:\n")
            f.writelines(error_logs)
        print(f"⚠️  {len(error_logs)} errors written to {error_log_path}")

    df = pd.DataFrame(rows)
    # Only keep columns that exist in df (to avoid KeyErrors if not all columns present)
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]
    print(f"✅ Flattened {len(df)} messages from {len(conversations)} conversations")
    return df, len(error_logs), len(errored_conversations)

def run_flatten_and_sample(conversations, output_csv_path="data/conversations_flat.csv", error_log_path="data/flat_error.txt", show_sample=True):
    """
    Flatten all conversations and write to CSV. Optionally display a sample (in Jupyter).
    """
    df, errors, conv_errors = flatten_all_messages_to_df(conversations, error_log_path)
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} rows to {output_csv_path}")
    print(f"⚠️ {errors} errors, {conv_errors} conversations with errors")
    if errors > 0: print(f"  (See {error_log_path} for details)")

    # Display sample only if in Jupyter and requested
    if show_sample:
        try:
            from IPython.display import display
            if df.empty:
                sample = pd.DataFrame(columns=["conversation_id", "role", "content"])
            else:
                total = len(df)
                idxs = [0, total - 1] if total > 1 else [0]
                if total > 10:
                    step = total // 9
                    idxs += [i * step for i in range(1, 9)]
                idxs = sorted(set([i for i in idxs if 0 <= i < total]))
                sample = df.iloc[idxs]
            display(sample)
        except ImportError:
            print("(Sample display skipped: not in Jupyter)")

    return df

# ===[ Direct script usage ]===
if __name__ == "__main__":
    # This block will only run if you execute the script directly
    # You can modify this to load your conversations file as needed
    conversations_path = input("Enter the path to conversations.json: ").strip()
    if not os.path.isfile(conversations_path):
        print(f"File not found: {conversations_path}")
    else:
        with open(conversations_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        run_flatten_and_sample(conversations)
