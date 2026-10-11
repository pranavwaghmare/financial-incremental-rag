import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])


@router.get("/latest")
def latest_benchmark():
    reports = sorted(
        Path("benchmark_runs").glob("incremental_eval_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not reports:
        raise HTTPException(
            status_code=404,
            detail="No incremental benchmark report found. Run incremental_eval.py first.",
        )

    try:
        with reports[0].open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="Could not read benchmark report.") from exc

    return {
        "report_name": reports[0].name,
        "summary": report.get("summary", {}),
    }
