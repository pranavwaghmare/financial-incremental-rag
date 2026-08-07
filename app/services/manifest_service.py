from app.database.sqlite_db import ManifestDB


class ManifestService:

    def __init__(self):

        self.db = ManifestDB()

        self.connection = self.db.conn

        self.cursor = self.db.get_cursor()

    def add_chunk(

        self,

        document_name,

        document_id,

        chunk_id,

        chunk_index,

        page,

        chunk_hash,

        embedding_id

    ):

        self.cursor.execute(

            """

            INSERT INTO manifest(

                document_name,

                document_id,

                chunk_id,

                chunk_index,

                page,

                chunk_hash,

                embedding_id

            )

            VALUES(?,?,?,?,?,?,?)

            """,

            (

                document_name,

                document_id,

                chunk_id,

                chunk_index,

                page,

                chunk_hash,

                embedding_id

            )

        )

        self.connection.commit()

    def get_document_chunks(self, document_name):

        self.cursor.execute(

            """

            SELECT *

            FROM manifest

            WHERE document_name=?

            ORDER BY chunk_index

            """,

            (document_name,)

        )

        return self.cursor.fetchall()

    def get_hash_map(self, document_name):

        self.cursor.execute(

            """

            SELECT chunk_index, chunk_hash

            FROM manifest

            WHERE document_name=?

            """,

            (document_name,)

        )

        rows = self.cursor.fetchall()

        return {

            row[0]: row[1]

            for row in rows

        }

    def delete_chunk(self, chunk_id):

        self.cursor.execute(

            """

            DELETE

            FROM manifest

            WHERE chunk_id=?

            """,

            (chunk_id,)

        )

        self.connection.commit()

    def update_chunk(

        self,

        chunk_id,

        new_hash,

        embedding_id

    ):

        self.cursor.execute(

            """

            UPDATE manifest

            SET

            chunk_hash=?,

            embedding_id=?,

            version=version+1,

            updated_at=CURRENT_TIMESTAMP

            WHERE chunk_id=?

            """,

            (

                new_hash,

                embedding_id,

                chunk_id

            )

        )

        self.connection.commit()

    def chunk_exists(self, chunk_id):

        self.cursor.execute(

            """

            SELECT COUNT(*)

            FROM manifest

            WHERE chunk_id=?

            """,

            (chunk_id,)

        )

        return self.cursor.fetchone()[0] > 0