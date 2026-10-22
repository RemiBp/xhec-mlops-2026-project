FROM python:3.11.15-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PREFECT_HOME=/app/.prefect \
    PREFECT_API_URL=http://127.0.0.1:4201/api \
    PREFECT_SERVER_ANALYTICS_ENABLED=false

WORKDIR /app
RUN pip install --no-cache-dir uv==0.9.4
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --python /usr/local/bin/python
COPY src ./src
COPY data/abalone.csv ./data/abalone.csv
COPY bin/run_services.sh ./bin/run_services.sh
RUN useradd --create-home student \
    && mkdir -p .prefect mlruns reports src/web_service/local_objects \
    && chmod +x bin/run_services.sh \
    && chown -R student:student /app
USER student
EXPOSE 8001 4201
HEALTHCHECK --start-period=60s --interval=10s --timeout=5s --retries=6 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8001/health',timeout=2); urlopen('http://127.0.0.1:4201/api/health',timeout=2)"
CMD ["bash", "bin/run_services.sh"]
