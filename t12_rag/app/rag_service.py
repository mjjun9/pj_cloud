from openai import OpenAI
from app.config import settings
from app.mariadb_vector_store import MariaDBVectorStore

class RagService:
    def __init__(self):
        self.vector_store = MariaDBVectorStore()
        self.llm = OpenAI(api_key=settings.OPENAI_API_KEY)

    #RAG 답변 생성 메소드
    def generate_answer(self, question: str):
        #(1) 벡터검색
        docs = self.vector_store.similarity_search(question, k=5)

        if not docs:
            return "검색된 문서가 없어 RAG 응답을 만들 수 없습니다."

        #(2) 검색된 문서를 하나의 Context 텍스트로 합침
        context_text = '\n\n'.join([ f'- {d.page_content}' for d in docs ])

        #(3) LLM에게 전달할 프롬프트
        prompt = f'''
당신은 검색 문서 기반으로 답변하는 RAG 전문가입니다.

[관련 문서들]
{context_text}

[사용자 질문]
{question}

위 문서를 근거로 정확하고 친절한 한국어 답변을 생성하세요.
'''
        #(4) LLM 호출
        response = self.llm.chat.completions.create(
            model = 'gpt-5',
            #temperature=0.1, #0.0 ~ 2.0 ( 기본값: 1.0 )
            messages=[
                {'role':'system', 'content':'당신은 문서 기반 RAG QA 전문가입니다.'},
                {'role':'user', 'content':prompt},
            ]
        )
        return response.choices[0].message.content