from fastapi import APIRouter

from app.services.analytics_service import AnalyticsService

router = APIRouter()

analytics = AnalyticsService()


@router.get("/analytics")
def analytics_dashboard():

    return {

        "documents": analytics.document_statistics(),

        "incremental": analytics.incremental_statistics(),

        "queries": analytics.query_statistics()

    }