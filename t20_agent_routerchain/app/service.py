from mariadb_vector_store import MariaDBVectorStore
from config import settings
from langchain_core.documents import Document

# from openai import OpenAI
from langchain_openai import ChatOpenAI #추가
from langchain_core.prompts import PromptTemplate #추가
from langchain_core.output_parsers import StrOutputParser #추가

class VectorService:
    def __init__(self):
        self.store = MariaDBVectorStore()
        # self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY
        )

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

        # prompt = f"""
        # 아래 문서를 참고하여 질문에 답하세요.
        # [문서 내용]
        # {context}
        # [질문]
        # {query}
        # """
        prompt = PromptTemplate(
			input_variables=["context", "question"],
			template="""
			아래 문서를 참고하여 질문에 답하세요.

			문서:
			{context}

			질문:
			{question}
			"""
		)

        # response = self.client.responses.create(
        #     model="gpt-4o-mini",
        #     input=prompt
        # )
        #return response.output_text

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})

    #4. Agent 로직 추가
    def agent_answer(self, query: str):
        """
        Agent 역할:
        1. 질문을 분석해서
        2. RAG가 필요한지 판단
        3. 행동 선택
        """

        decision = self.decide_action(query)

        if decision == "RAG":
            return f"[Agent 판단] RAG검색 필요\n\n" + self.rag_query(query) #상단의 기존정의
        else:
            return f"[Agent 판단] 일반 응답\n\n" + self.general_answer(query) #하단의 새정의
        
    ROUTER_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
    너는 AI 에이전트의 라우터이다.
    아래 질문을 보고, 어떤 행동이 더 적절한지 판단하라.

    [행동 선택지]
    - RAG : 내부 문서(벡터 DB)에 근거해 답해야 하는 경우
    - GENERAL : 일반 지식으로 바로 답할 수 있는 경우

    규칙:
    - 내부 문서에 있을 법한 내용이면 RAG
    - 일반적인 상식, 개념 설명이면 GENERAL
    - 반드시 아래 중 하나만 출력하라
    - 다른 설명은 절대 출력하지 마라

    출력 형식:
    RAG 또는 GENERAL

    질문:
    {question}
    """
    )

    def decide_action(self, query: str) -> str:
        router_chain = (self.ROUTER_PROMPT | self.llm | StrOutputParser())
        action = router_chain.invoke({"question": query}).strip()

        if action not in ['RAG', 'GENERAL']:
            return 'GENERAL'
        
        return action

    def general_answer(self, query: str) -> str:
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            다음 질문에 대해 간단하고 명확하게 답변하세요.

            질문: {question}
            """
        )

        # response = self.client.responses.create(
        #     model="gpt-4o-mini",
        #     input=query
        # )
        # return response.output_text

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": query})