import streamlit as st
from service import VectorService

service = VectorService()

st.title("t31_rag_csv")

# 1. DB 테이블 생성
st.header("1. DB 테이블 생성")
if st.button("테이블 생성하기"):
    msg = service.create_table()
    st.success(msg)

# 2-1. 샘플 문서 저장 (기존 유지)
st.header("2-1. 샘플 문서 저장 (옵션)")
if st.button("샘플 문서 저장"):
    count = service.load_sample_docs()
    st.success(f"{count}개의 문서가 저장되었습니다.")

# 2-2. CSV > Chunk > Embedding > VectorDB 적재 (추가)
st.header("2-2. CSV 데이터 적재 (edu_flow_data 5종)")
rows_per_chunk = st.number_input(
    "chunk 당 rows 수 (추천 10~50)",
    min_value=5,
    max_value=200,
    value=25,
    step=5
)
st.caption("data/*.csv 가 있어야 합니다.")
if st.button("CSV 데이터 적재하기"):
    result = service.load_csv_docs(rows_per_chunk=int(rows_per_chunk))
    if result["ok"]:
        st.success(result["message"])
    else:
        st.error(result["message"])

# 3. RAG 기반 질문/답변
st.header("3. RAG 기반 질문/답변")
query = st.text_input("질문을 입력하세요:")

if st.button("질문하기"):
    answer = service.rag_query(query)
    st.write(answer)