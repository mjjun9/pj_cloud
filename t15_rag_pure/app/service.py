from mariadb_vector_store import MariaDBVectorStore
from openai import OpenAI
from config import settings
from langchain_core.documents import Document

class VectorService:
    def __init__(self):
        self.store = MariaDBVectorStore()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    #1. 테이블 생성
    def create_table(self):
        self.store.create_table()
        return "기존 테이블 존재 or 테이블 생성 완료!"

    #2. 샘플 문서 저장
    def load_sample_docs(self):
        samples = [
            Document(
                page_content="2347년 양자 엔진이 완성되어 인류는 성간 항해에 성공했다.",
                metadata={"meta1": "info"}
            ),
            Document(
                page_content="K-17 달 식민지에서 발견된 크리스털 코어는 멸종한 문명의 기억을 담고 있었다.",
                metadata={}
            ),
            Document(
                page_content="시간은 나선이며 크로노 워커들은 다른 시간선에서 경고를 받았다.",
                metadata={"author": "john", "type": "blog"}
            )
        ]

        self.store.add_documents(samples)
        return len(samples)

    #3. RAG 질문/답변
    def rag_query(self, query, k=3):
        results = self.store.similarity_search(query, k=k)

        if results is None:
            return "테이블이 존재하지 않습니다.\n먼저 '1. DB 테이블 생성'을 눌러주세요."

        if len(results) == 0:
            return "관련 문서를 찾을 수 없습니다."

        # 문맥 구성
        context = "\n\n".join(d.page_content for d in results)

        prompt = f"""
        아래 문서를 참고하여 질문에 답하세요.

        [문서 내용]
        {context}

        [질문]
        {query}
        """

        # 최신 OpenAI SDK 방식 — responses.create 사용
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        return response.output_text