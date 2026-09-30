from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv('OPENAI_API_KEY')
# key = ''
client = OpenAI(api_key=key)

response = client.chat.completions.create(
    model = 'gpt-5',
    temperature=0.1, # 0.0(이성적) ~ 2.0(감성적)
    messages=[
        {'role':'system', 'content':'You are a helpful assistant.'},
        {'role':'user', 'content':'2002년 월드컵 4위 팀은 어디야?'}, 
    ]
)

# print(response)
print(response.choices[0].message.content)