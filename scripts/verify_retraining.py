"""Trigger the registered Prefect deployment and verify that the API reloads it."""

import asyncio
import json
from pathlib import Path

from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound

from scripts.smoke_test import call
from src.web_service.lib.models import Abalone


async def verify():
    base = "http://127.0.0.1:8526"
    before = call(base, "/health")
    for _ in range(90):
        try:
            call("http://127.0.0.1:4251", "/api/health")
            break
        except OSError:
            await asyncio.sleep(1)
    else:
        raise RuntimeError("Prefect API did not become ready")
    async with get_client() as client:
        for _ in range(90):
            try:
                deployment = await client.read_deployment_by_name("abalone-retraining/weekly")
                break
            except (ObjectNotFound, OSError):
                await asyncio.sleep(1)
        else:
            raise RuntimeError("Weekly deployment was not registered")
        run = await client.create_flow_run_from_deployment(deployment.id, name="verification")
        for _ in range(120):
            observed = await client.read_flow_run(run.id)
            if observed.state.is_completed():
                break
            if observed.state.is_final():
                raise RuntimeError(f"Retraining ended in {observed.state.name}")
            await asyncio.sleep(1)
        else:
            raise RuntimeError("The serving runner did not complete the verification run")
    prediction = call(base, "/predict", Abalone.Config.schema_extra["example"])
    after = call(base, "/health")
    assert before["trained_at"] != after["trained_at"]
    assert prediction["model_version"] == after["model_version"]
    result = {
        "deployment": "abalone-retraining/weekly",
        "flow_run_id": str(run.id),
        "state": observed.state.name,
        "api_reloaded_after_retraining": True,
        "previous_trained_at": before["trained_at"],
        "current_trained_at": after["trained_at"],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/retraining_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(verify())
