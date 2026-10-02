from mariadb_vector_store import MariaDBVectorStore
from openai import OpenAI
from config import settings
from langchain_core.documents import Document
import requests

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

    #3. RAG+FC 질문/답변
    def rag_fc_query(self, query, k=3): #rag_query() -> rag_fc_query()
        #(1) 사용자 질문을 영어로 번역
        translated_query = self.translate_query_to_english(query)

        #(2) 번역된 영어 문장에서 도시명 추출
        location = self.extract_location(translated_query)

        #(3) RAG 검색 실행 (translated_query로 변경)
        results = self.store.similarity_search(translated_query, k=k)

        if results is None:
            return "테이블이 존재하지 않습니다.\n먼저 '1. DB 테이블 생성'을 눌러주세요."

        #(4) 날씨 키워드 리스트 준비
        weather_keywords = ["weather", "temperature", "temp", "climate", "rain", "snow"]
        translated_query_lower = translated_query.lower()

        #(5) RAG 검색결과 없을 때 > FC 실행
        if len(results) == 0:
            if any(key in translated_query_lower for key in weather_keywords):
                return self.get_weather(location)
            return "관련 문서를 찾을 수 없습니다."

        #(6) 문맥 구성
        context = "\n\n".join(d.page_content for d in results)

        #(7) 유사도 필터링 > FC 실행 (유사도가 너무 낮을 때)
        top_score = results[0].metadata.get("score", 0.0)
        print("#top_score: ", top_score)
        if top_score < 0.3:
            if any(key in translated_query_lower for key in weather_keywords):
                return self.get_weather(location)

        #(8) 프롬프트 작성
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

    # FC 관련 함수들 추가
    def translate_query_to_english(self, query):
        prompt = f"""
        다음 문장을 영어로 번역하세요. 다른 설명 없이 번역된 문장만 출력하세요.
        문장: "{query}"
        """
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        return response.output_text.strip()
    
    def extract_location(self, query):
        prompt = f"""
        아래 문장에서 '날씨를 알고 싶은 도시명'만 한 단어로 출력하세요.
        문장: "{query}"
        출력 형식: 도시명 단독 출력 (예: Seoul)
        """
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )
        city = response.output_text.strip()

        import re #St. John's -> St Johns
        city = re.sub(r"[^a-zA-Z\s]", "", city).strip()

        if len(city)==0:
            return 'Seoul'

        return city

    def get_weather(self, location='Seoul'):
        url = (
            "http://api.openweathermap.org/data/2.5/weather"
            f"?q={location}&units=metric&APPID={settings.OPENWEATHER_API_KEY}"
        )

        res = requests.get(url)
        if res.status_code != 200:
            return f"날씨 정보를 가져올 수 없습니다. (status: {res.status_code})"

        data = res.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        city = data["name"]

        return f"{city}의 현재 기온은 {temp}°C, 날씨는 {desc} 입니다."