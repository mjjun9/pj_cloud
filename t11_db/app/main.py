import sys, os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from app.service import VectorService

service = VectorService()

st.title("t11_db")

# (1) 샘플 문서 저장
if st.button("샘플 문서 저장"):
    count = service.load_sample_docs()
    st.success(f"{count}개의 문서를 벡터스토어에 저장했습니다.")

st.divider()

# (2) 검색
st.subheader("검색 기능")
method = st.selectbox("검색 메소드", ["search1", "search2"])
query = st.text_input("검색어 입력")

if st.button("검색 실행"):
    if method == "search1":
        results = service.search1(query)
    else: #"search2"
        results = service.search2(query)

    for r in results:
        st.write("---")
        st.write("**Text:**", r.page_content)
        st.write("**Metadata:**", r.metadata)

st.divider()

# (3) Score 출력
st.subheader("문서 Score 출력")
score_query = st.text_input("Score 검색어 입력")

if st.button("Score 확인"):
    results = service.search_with_scores(score_query)
    for doc, score in results:
        st.write("---")
        st.write("**Text:**", doc.page_content)
        st.write("**Metadata:**", doc.metadata)
        st.write(f"**Score:** {score}")

st.divider()

# (4) Native SQL 접근
if st.button("Native SQL 조회"):
    rows = service.native_query()
    st.write(rows)