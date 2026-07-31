FROM python:3.12-slim

LABEL org.opencontainers.image.title="VeriWeave-VITA-PRO" \
      org.opencontainers.image.description="Provenance-robust verification for auditable LLM policy decisions" \
      org.opencontainers.image.source="https://github.com/vtavakkoli/VeriWeave-VITA" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install .

COPY data ./data
RUN mkdir -p /app/result

CMD ["python", "-m", "veriweave.main"]
