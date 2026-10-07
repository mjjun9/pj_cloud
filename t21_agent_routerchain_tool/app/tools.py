import requests
from config import settings

#OpenAI Tools 형식으로 호출되는 함수
def tool_weather(city:str):
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city}&units=metric&APPID={settings.OPENWEATHER_API_KEY}"
    )

    res = requests.get(url)
    if res.status_code != 200:
        return {"error": f"tool_weather() error: {res.status_code})"}

    data = res.json()

    return {
        "city": data["name"],
        "temp": data["main"]["temp"],
        "description": data["weather"][0]["description"]
    }

# OpenAI Tools JSON Schema //함수 호출 인터페이스 정의서
tool_weather_spec = {
    "type": "function",
    "name": "tool_weather",
    "function": {
        "name": "tool_weather",
        "description": "city 이름을 받아 해당 도시의 현재 날씨 정보를 반환합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "영문 도시명 (예: Seoul, Tokyo, New York)"
                }
            },
            "required": ["city"]
        }
    }
}