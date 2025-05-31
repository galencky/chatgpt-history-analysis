#!/bin/sh

# ─────────────────────────────────────────────────────
# 🛠️  Fix permissions if Synology NAS reset them
#     - 775: allows user and group to read/write/execute
#     - `|| true`: prevent exit if chmod fails (e.g., folder missing)
#     - `2>/dev/null`: suppress permission denied errors
# ─────────────────────────────────────────────────────
chmod -R 775 /app/data /app/output /app/drop_zip_here 2>/dev/null || true

# ─────────────────────────────────────────────────────
# 🚀 Start the main Python application
# ─────────────────────────────────────────────────────
exec python main.py
