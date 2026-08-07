from app.database.analytics_db import AnalyticsDB


class AnalyticsService:

    def __init__(self):

        self.db = AnalyticsDB()

    # -----------------------------------------
    # Upload Metrics
    # -----------------------------------------

    def save_upload_metrics(

        self,

        filename,

        company,

        sector,

        pages,

        chunks,

        new_chunks,

        modified_chunks,

        deleted_chunks,

        unchanged_chunks,

        embedding_calls,

        embedding_calls_saved,

        reused_percentage,

        processing_time

    ):

        self.db.insert_upload_metrics(

            filename,

            company,

            sector,

            pages,

            chunks,

            new_chunks,

            modified_chunks,

            deleted_chunks,

            unchanged_chunks,

            embedding_calls,

            embedding_calls_saved,

            reused_percentage,

            processing_time

        )

    # -----------------------------------------
    # Query Metrics
    # -----------------------------------------

    def save_query_metrics(

        self,

        question,

        response_time,

        document,

        page

    ):

        self.db.insert_query_metrics(

            question,

            response_time,

            document,

            page

        )

    # -----------------------------------------
    # Dashboard
    # -----------------------------------------

    def document_statistics(self):

        return self.db.get_document_statistics()


    def incremental_statistics(self):

        return self.db.get_incremental_statistics()


    def query_statistics(self):

        return self.db.get_query_statistics()