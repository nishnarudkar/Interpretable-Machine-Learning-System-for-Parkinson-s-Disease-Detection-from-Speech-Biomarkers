# ── Stage 1: install dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir --user -r requirements-api.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy only what the API needs at runtime
COPY api/        api/
COPY src/config.py              src/config.py
COPY src/__init__.py            src/__init__.py
COPY src/mlflow_comparison.py   src/mlflow_comparison.py
COPY src/model_selection.py     src/model_selection.py
COPY models/     models/
COPY static/     static/
COPY templates/  templates/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
