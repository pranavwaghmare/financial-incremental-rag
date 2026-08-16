from pathlib import Path

from fastapi import APIRouter

from app.services.analytics_service import AnalyticsService
from app.services.document_service import DocumentProcessingService

router = APIRouter()

analytics = AnalyticsService()


@router.get("/analytics")
def analytics_dashboard():

    return {

        "documents": analytics.document_statistics(),

        "incremental": analytics.incremental_statistics(),

        "queries": analytics.query_statistics()

    }


@router.get("/benchmark")
def benchmark_dashboard():
    v1_path = Path("demo_docs/v1/Acme_Report.pdf")
    v2_path = Path("demo_docs/v2/Acme_Report.pdf")

    if not v1_path.exists() or not v2_path.exists():
        return {
            "status": "missing_demo_files",
            "message": "Benchmark demo documents were not found. Generate them first with the demo PDF setup or add the sample reports into demo_docs/v1 and demo_docs/v2.",
        }

    benchmark_root = Path("benchmark_runs")
    benchmark_root.mkdir(exist_ok=True)

    result = DocumentProcessingService().benchmark_against_full_rebuild(
        str(v1_path),
        str(v2_path),
        temp_root=str(benchmark_root / "latest")
    )

    return {
        "status": "ok",
        "message": "Incremental vs full rebuild benchmark completed using isolated temp storage.",
        "result": result,
    }