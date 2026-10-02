import streamlit as st
from service import VectorService

service = VectorService()
st.title('t16_rag_fc')

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

#3. RAG + FC(Tools) 기반 질문/답변
st.header('3. RAG + FC(Tools) 기반 질문/답변')
query = st.text_input('질문을 입력하세요')

if st.button('질문하기'):
    answer = service.rag_tool_query(query)
    st.write(answer)