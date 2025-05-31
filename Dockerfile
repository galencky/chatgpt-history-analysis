FROM python:3.11-slim

# ---------- system packages ----------
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc tini && \
    rm -rf /var/lib/apt/lists/*

# ---------- working dir ----------
WORKDIR /app

# ---------- python deps ----------
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ---------- project files ----------
COPY . .

# ---------- create non-root user ----------
RUN useradd -ms /bin/bash syno && \
    chown -R syno:syno /app

# ---------- Optional: ensure dirs exist and are writable ----------
RUN mkdir -p /app/data /app/output /app/drop_zip_here && \
    chmod -R 775 /app/data /app/output /app/drop_zip_here

# ---------- switch to non-root ----------
USER syno

# ---------- ensure startup wrapper is executable ----------
RUN chmod +x /app/start.sh

# ---------- container entry ----------
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/app/start.sh"]
