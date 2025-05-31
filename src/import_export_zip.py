import os
import json
import zipfile
import shutil
from datetime import datetime

def prepare_export_and_load_conversations(base_dir, zip_path):
    """
    1. Prompt for a complete directory to the ChatGPT export zip file.
    2. Build the 'data' directory (if missing) in the same location as main.py.
    3. Decompress the zip, rename the decompressed folder as 'chatgpt-YYYYMMDD-HHMM'.
    4. Copy/move the decompressed contents into 'data'.
    5. Load conversations.json.
    Returns: conversations (list), folder (str)
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file does not exist: {zip_path}")
    if not zip_path.lower().endswith(".zip"):
        raise ValueError("zip_path must point to a .zip")

    # 2. Ensure 'data' directory exists (same as main.py)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 3. Decompress zip to a temp location
    temp_dir = os.path.join(base_dir, "_decompress_temp")
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    print(f"📦 Decompressing zip file...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    print(f"✅ Decompressed to {temp_dir}")

    # 4. Determine what we extracted
    # Find conversations.json (recursively)
    json_path = None
    for root, dirs, files in os.walk(temp_dir):
        if "conversations.json" in files:
            json_path = os.path.join(root, "conversations.json")
            extracted_root = root
            break
    if json_path is None:
        raise RuntimeError("❌ conversations.json not found in extracted zip!")

    # 5. Build the rename target as 'chatgpt-YYYYMMDD-HHMM'
    # Try to get date from file/folder name, else fallback to now
    try:
        import re
        zip_base = os.path.basename(zip_path)
        match = re.search(r'(\d{8,})(?:[_-]?(\d{4,}))?', zip_base)
        if match:
            date_str = match.group(1)
            time_str = match.group(2) if match.group(2) else "0000"
            dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        else:
            dt = datetime.now()
    except Exception:
        dt = datetime.now()
    folder_name = dt.strftime("chatgpt-%Y%m%d-%H%M")
    target_folder = os.path.join(data_dir, folder_name)

    # 6. Move/copy the *parent folder of conversations.json* to data/
    # If extracted_root == temp_dir (no subfolder), just copy files
    if os.path.exists(target_folder):
        print(f"⚠️ Folder {target_folder} already exists. Removing old copy.")
        shutil.rmtree(target_folder)
    if extracted_root == temp_dir:
        # Flat structure: create folder, copy files
        os.makedirs(target_folder, exist_ok=True)
        for fname in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, fname)
            if os.path.isfile(fpath):
                shutil.copy2(fpath, target_folder)
            elif os.path.isdir(fpath):
                shutil.copytree(fpath, os.path.join(target_folder, fname))
    else:
        # The json was in a subfolder: move/copy that whole folder
        shutil.copytree(extracted_root, target_folder)

    print(f"✅ Copied export to {target_folder}")

    # 7. Clean up temp dir
    shutil.rmtree(temp_dir)

    # 8. Load conversations.json from the new location
    json_path = os.path.join(target_folder, "conversations.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"❌ conversations.json not found in {target_folder}")
    with open(json_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    print(f"✅ Loaded {len(conversations)} conversations from {json_path}")

    # Optional: show first conversation title
    if conversations:
        print(f"🔎 First conversation title: {conversations[0].get('title', '<no title>')}")
    else:
        print("⚠️ No conversations loaded.")

    # Return conversations and folder name for further steps
    return conversations, folder_name
