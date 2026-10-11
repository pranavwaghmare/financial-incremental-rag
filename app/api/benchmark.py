import json
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])
benchmark_process = None


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


@router.post("/run")
def run_benchmark():
    global benchmark_process

    if benchmark_process is not None and benchmark_process.poll() is None:
        return {"status": "running", "message": "A benchmark is already running."}

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "benchmark_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = (output_dir / "latest_benchmark.log").open("w", encoding="utf-8")

    benchmark_process = subprocess.Popen(
        [
            sys.executable,
            "incremental_eval.py",
            "--n",
            "10",
            "--base-fraction",
            "0.8",
            "--rounds",
            "2",
            "--judge",
            "semantic",
        ],
        cwd=project_root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )

    return {
        "status": "started",
        "message": "Benchmark started with 10 questions, 2 incremental rounds, and the semantic judge.",
    }


@router.get("/status")
def benchmark_status():
    if benchmark_process is None:
        return {"status": "idle"}

    return {
        "status": "running" if benchmark_process.poll() is None else "complete",
        "return_code": benchmark_process.poll(),
    }
