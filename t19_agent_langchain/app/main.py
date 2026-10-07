import streamlit as st
from service import VectorService

service = VectorService()
st.title('t19_agent_langchain')

#1. DB테이블 생성
st.header('1. DB테이블 생성')
if st.button('테이블 생성하기'):
    msg = service.create_table()
    st.success(msg)

#2. 샘플문서 저장
st.header('2. 샘플문서 저장')
if st.button('샘플문서 저장'):
    count = service.load_sample_docs()
    st.success(f'{count}개의 문서가 저장되었습니다')

#3. RAG 기반 질문/답변
st.header('3. LangChain Agent에게 질문/답변')
query = st.text_input('질문을 입력하세요', placeholder='K-17 달 식민지에서 발견된 것은 무엇이야? / 당신의 오늘 기분은?')

# if st.button('질문하기'):
#     answer = service.rag_query(query)
#     st.write(answer)
if st.button('Agent 실행'):
    if query.strip() == '':
        st.warning('질문을 입력해주세요')
    else:
        answer = service.agent_answer(query)
        st.write(answer)