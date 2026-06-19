FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps (cached layer unless requirements.txt changes) ─────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ─────────────────────────────────────────────
RUN mkdir -p generated_reports models

# ── Expose API port ────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health-check (Docker will mark container unhealthy if API is down) ─────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# ── Start server ───────────────────────────────────────────────────────────
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
