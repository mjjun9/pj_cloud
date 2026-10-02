from mariadb_vector_store import MariaDBVectorStore
from openai import OpenAI
from config import settings
from langchain_core.documents import Document

# import requests
from tools import tool_weather, tool_weather_spec
import json

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
    def rag_tool_query(self, query, k=3): #rag_fc_query() -> rag_tool_query()
        #(1) 사용자 질문을 영어로 번역
        translated_query = self.translate_query_to_english(query)

        #(2) 번역된 영어 문장에서 도시명 추출
        location = self.extract_location(translated_query)

        #(3) RAG 검색 실행 (translated_query로 변경)
        results = self.store.similarity_search(translated_query, k=k)

        if results is None:
            return "테이블이 존재하지 않습니다.\n먼저 '1. DB 테이블 생성'을 눌러주세요."

        #(4) 날씨 관련 질문 여부 판단
        weather_detect_prompt = """
        Determine whether the user's question is about weather.
        If true, return YES. If not, return NO.
        Only output YES or NO.
        """
        detect_resp = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": weather_detect_prompt},
                {"role": "user", "content": query}
            ]
        )
        is_weather_question = detect_resp.output_text.strip().upper() == "YES"

        #(5) RAG검색결과 없을 때 > FC실행
        if is_weather_question:
            system_prompt = """
            You MUST use weather_tool to get weather information.
            Do NOT answer directly. Always return a tool_call.
            """

            user_prompt = f"""
            User question: {query}
            Extracted city: {location}
            """

            call_resp = self.client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=[tool_weather_spec],
                tool_choice="auto"
            )

            msg = call_resp.output[0]

            # tool_call 발생
            if msg.type == "tool_call":
                args = json.loads(msg.arguments)
                tool_result = tool_weather(**args)
                print("@tool_result: " + str(tool_result))

                return self.summarize_weather(tool_result) #최종자연어 답변생성

            # tool_call 실패 > 직접 tool_weather 실행
            direct = tool_weather(location)
            return self.summarize_weather(direct)

        #(6) (날씨 질문이 아니면) RAG를 위한 문맥 구성과 프롬프트 작성
        if len(results) > 0:
            context = "\n\n".join(d.page_content for d in results)

            prompt = f"""
            Answer the user's question using the following documents.

            [Documents]
            {context}

            [Question]
            {query}
            """

            response = self.client.responses.create(
                model="gpt-4o-mini",
                input=prompt
            )
            return response.output_text

        return "관련 문서를 찾을 수 없습니다."

    # tool호출 결과(dict)를 자연스러운 한국어 문장으로 변환하는 함수
    def summarize_weather(self, tool_result:dict):
        city = tool_result.get("city", "Unknown")
        temp = tool_result.get("temp", "?")
        desc = tool_result.get("description", "?")

        final_prompt = f"""
        아래 날씨 정보를 한국어로 자연스럽게 요약해 주세요.

        도시: {city}
        기온: {temp}°C
        날씨: {desc}
        """

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=final_prompt
        )

        return response.output_text.strip()

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

'''
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
'''