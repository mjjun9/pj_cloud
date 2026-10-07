import pymysql
from langchain_openai import OpenAIEmbeddings
from config import settings
import math
import json

class MariaDBVectorStore:
    def __init__(self, table_name=None):
        self.table = table_name or settings.DB_TABLE
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.OPENAI_API_KEY
        )

    #1. DB연결
    def _connect(self):
        return pymysql.connect(
            host=settings.DB_HOST,
            user=settings.DB_USER,
            password=settings.DB_PASS,
            database=settings.DB_NAME,
            port=settings.DB_PORT,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )

    #2. 테이블 존재 여부 확인
    def table_exists(self):
        conn = self._connect()
        cur = conn.cursor()
        sql = """
        SELECT COUNT(*) AS cnt
        FROM information_schema.tables
        WHERE table_schema = %s
        AND table_name = %s
        """
        cur.execute(sql, (settings.DB_NAME, self.table))
        result = cur.fetchone()
        conn.close()
        return result["cnt"] > 0

    #3. 테이블 생성
    def create_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            text TEXT NOT NULL,
            embedding LONGTEXT NOT NULL,
            metadata LONGTEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        conn = self._connect()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()

    #4. 문서 저장
    def add_documents(self, docs):
        if not self.table_exists():
            raise Exception("테이블이 존재하지 않습니다. 먼저 테이블을 생성하세요.")

        conn = self._connect()
        cur = conn.cursor()

        for doc in docs:
            embedding = self.embeddings.embed_query(doc.page_content)

            sql = f"""
            INSERT INTO {self.table} (text, embedding, metadata)
            VALUES (%s, %s, %s)
            """

            cur.execute(sql, (
                doc.page_content,
                json.dumps(embedding),
                json.dumps(doc.metadata)
            ))

        conn.commit()
        conn.close()

    #5. 코사인 유사도 계산
    def _cosine_similarity(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    #6. similarity_search
    def similarity_search(self, query, k=5):
        if not self.table_exists():
            return None  # 테이블 없음

        query_vec = self.embeddings.embed_query(query) #질문 임베딩

        conn = self._connect()
        cur = conn.cursor()
        cur.execute(f"SELECT id, text, embedding, metadata FROM {self.table}")
        rows = cur.fetchall()
        conn.close()

        scored = []
        for row in rows:
            emb = json.loads(row["embedding"])
            score = self._cosine_similarity(query_vec, emb)

            from langchain_core.documents import Document
            doc = Document(
                page_content=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            doc.metadata["score"] = score

            scored.append(doc)

        # 정렬 후 상위 k개만
        scored.sort(key=lambda x: x.metadata["score"], reverse=True)
        return scored[:k]