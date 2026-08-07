import sqlite3
from pathlib import Path


class AnalyticsDB:

    def __init__(self):

        Path("analytics_db").mkdir(exist_ok=True)

        self.conn = sqlite3.connect(
            "analytics_db/analytics.db",
            check_same_thread=False
        )

        self.cursor = self.conn.cursor()

        self.create_tables()

    # ---------------------------------------------------
    # Create Tables
    # ---------------------------------------------------

    def create_tables(self):

        self.cursor.execute("""

        CREATE TABLE IF NOT EXISTS upload_metrics(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            filename TEXT,

            company TEXT,

            sector TEXT,

            pages INTEGER,

            chunks INTEGER,

            new_chunks INTEGER,

            modified_chunks INTEGER,

            deleted_chunks INTEGER,

            unchanged_chunks INTEGER,

            embedding_calls INTEGER,

            embedding_calls_saved INTEGER,

            reused_percentage REAL,

            processing_time REAL,

            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

        """)

        self.cursor.execute("""

        CREATE TABLE IF NOT EXISTS query_metrics(

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            question TEXT,

            response_time REAL,

            retrieved_document TEXT,

            retrieved_page INTEGER,

            query_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )

        """)

        self.conn.commit()

    # ---------------------------------------------------
    # Upload Analytics
    # ---------------------------------------------------

    def insert_upload_metrics(

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

        self.cursor.execute("""

        INSERT INTO upload_metrics(

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

        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)

        """,

        (

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

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Query Analytics
    # ---------------------------------------------------

    def insert_query_metrics(

        self,

        question,

        response_time,

        document,

        page

    ):

        self.cursor.execute("""

        INSERT INTO query_metrics(

            question,

            response_time,

            retrieved_document,

            retrieved_page

        )

        VALUES(?,?,?,?)

        """,

        (

            question,

            response_time,

            document,

            page

        ))

        self.conn.commit()

    # ---------------------------------------------------
    # Dashboard Statistics
    # ---------------------------------------------------

    def get_document_statistics(self):

        self.cursor.execute("""

        SELECT

            COUNT(*) AS reports,

            COUNT(DISTINCT company) AS companies,

            COALESCE(SUM(chunks),0),

            COUNT(DISTINCT sector)

        FROM upload_metrics

        """)

        row = self.cursor.fetchone()

        return {

            "reports": row[0],

            "companies": row[1],

            "chunks": row[2],

            "sectors": row[3]

        }

    # ---------------------------------------------------

    def get_incremental_statistics(self):

        self.cursor.execute("""

        SELECT

            COALESCE(SUM(new_chunks),0),

            COALESCE(SUM(modified_chunks),0),

            COALESCE(SUM(deleted_chunks),0),

            COALESCE(SUM(unchanged_chunks),0),

            COALESCE(SUM(embedding_calls),0),

            COALESCE(SUM(embedding_calls_saved),0),

            COALESCE(AVG(processing_time),0)

        FROM upload_metrics

        """)

        row = self.cursor.fetchone()

        return {

            "new_chunks": row[0],

            "modified_chunks": row[1],

            "deleted_chunks": row[2],

            "unchanged_chunks": row[3],

            "embedding_calls": row[4],

            "embedding_calls_saved": row[5],

            "average_processing_time": round(row[6], 3)

        }

    # ---------------------------------------------------

    def get_query_statistics(self):

        self.cursor.execute("""

        SELECT

            COUNT(*),

            COALESCE(AVG(response_time),0)

        FROM query_metrics

        """)

        row = self.cursor.fetchone()

        return {

            "total_queries": row[0],

            "average_response_time": round(row[1], 3)

        }

    # ---------------------------------------------------

    def close(self):

        self.conn.close()