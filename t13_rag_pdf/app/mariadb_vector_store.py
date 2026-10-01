import json
import pymysql
from langchain_openai import OpenAIEmbeddings
from app.config import settings
import math

class MariaDBVectorStore:
    #(0) 초기화
    def __init__(self, table_name="ai_vector_store", embedding_model="text-embedding-3-small"): #문자열 -> 숫자배열
        self.table = table_name
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=settings.OPENAI_API_KEY
        )
        self._create_table()

    def _create_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            text TEXT NOT NULL,
            embedding JSON NOT NULL,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
        conn.close()

    def _connect(self):
        return pymysql.connect(
            host=settings.DB_HOST,
            user=settings.DB_USER,
            password=settings.DB_PASS,
            port=settings.DB_PORT,
            database=settings.DB_NAME,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor
        )

    #(1) 문서 저장
    def add_documents(self, documents):
        """
        documents: List[Document]
        """
        conn = self._connect()
        cur = conn.cursor()

        for doc in documents:
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

    #(2) 유사도 계산
    def _cosine_similarity(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    #(3) 유사도 검색
    def similarity_search(self, query, k=5, filter=None):

        # 1) 쿼리 embedding 생성
        query_vec = self.embeddings.embed_query(query)

        # 2) 모든 문서 가져오기 (교육용: 간단히 처리)
        conn = self._connect()
        cur = conn.cursor()

        sql = f"SELECT id, text, embedding, metadata FROM {self.table}"
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()

        scored_docs = []
        for row in rows:
            vec = json.loads(row["embedding"])
            score = self._cosine_similarity(query_vec, vec)

            # metadata 필터 적용
            if filter:
                match = True
                for key, val in filter.items():
                    if not row["metadata"]:
                        match = False
                        break
                    meta = json.loads(row["metadata"])
                    if key not in meta or meta[key] != val:
                        match = False
                        break
                if not match:
                    continue

            scored_docs.append((score, row))


        # Score 내림차순 정렬
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        scored_docs = scored_docs[:k]

        # LangChain Document 로 변환
        results = []
        from langchain_core.documents import Document

        for score, row in scored_docs:
            doc = Document(
                page_content=row["text"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {}
            )
            doc.metadata["score"] = score
            results.append(doc)

        return results

    #(4) 스코어를 이용한 유사도 검색
    def similarity_search_with_score(self, query, k=5):
        docs = self.similarity_search(query, k=k)
        return [(d, d.metadata["score"]) for d in docs]