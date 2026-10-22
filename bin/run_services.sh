#!/bin/bash
set -euo pipefail

cleanup() {
  for pid in $(jobs -pr); do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 0' TERM INT

# Train before readiness; a changed input CSV cannot silently retain an old model.
python -m src.modelling.main > reports/startup-training.json
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001 &
api_pid=$!
prefect server start --host 0.0.0.0 --port 4201 --analytics-off &
prefect_pid=$!

python - <<'PY'
import time
from urllib.request import urlopen
for attempt in range(90):
    try:
        urlopen('http://127.0.0.1:4201/api/health', timeout=2)
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError('Prefect did not become ready within 90 seconds')
PY

python -m src.modelling.workflows serve &
retraining_pid=$!
wait -n "$api_pid" "$prefect_pid" "$retraining_pid"
