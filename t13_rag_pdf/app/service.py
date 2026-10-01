from langchain_core.documents import Document
from app.mariadb_vector_store import MariaDBVectorStore
from app.sample_docs import sample_documents
from app.config import settings
import pymysql

TOP_K = 5 #상위 몇 개 가져올지
SIMILARITY_THRESHOLD = 0.7

class VectorService:
    def __init__(self):
        self.store = MariaDBVectorStore()

    #(1) 샘플 문서 저장
    def load_sample_docs(self):
        docs = [
            Document(page_content=d["text"], metadata=d["metadata"])
            for d in sample_documents
        ]

        self.store.add_documents(docs)
        return len(docs)

    #(2) search1 — 기본 similaritySearch
    def search1(self, query: str):
        return self.store.similarity_search(query, k=5)

    #(3) search2 — threshold + filterExpression
    def search2(self, query: str):
        filter_ = {"author": "john", "article_type": "blog"}
        return self.store.similarity_search(query, k=TOP_K, filter=filter_)

    #(4) Score 출력
    def search_with_scores(self, query: str):
        return self.store.similarity_search_with_score(query, k=2)

    #(5) Native SQL 조회
    def native_query(self):
        conn = pymysql.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASS,
            database=settings.DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {settings.DB_TABLE} LIMIT 20")
        rows = cur.fetchall()
        conn.close()
        return rows