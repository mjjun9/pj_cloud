import os
import math
import pandas as pd

from mariadb_vector_store import MariaDBVectorStore
from openai import OpenAI
from config import settings


class VectorService:
    def __init__(self):
        self.store = MariaDBVectorStore()
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    # 1. 테이블 생성
    def create_table(self):
        self.store.create_table()
        return "테이블 생성 완료!"

    # 2-1. 샘플 문서 저장 (기존 유지)
    def load_sample_docs(self):
        from langchain_core.documents import Document

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

    # 2-2. CSV > 자연어 chunk 생성 > 임베딩 > MariaDB 적재 (추가)
    def load_csv_docs(self, rows_per_chunk: int = 25):
        #data하위의 *.csv를 자연어 chunk(Document)로 변환 후 VectorDB에 적재.

        from langchain_core.documents import Document

        # t31_rag_csv/data 폴더 찾기
        app_dir = os.path.dirname(__file__)
        project_root = os.path.abspath(os.path.join(app_dir, ".."))
        data_dir = os.path.join(project_root, "data")

        targets = [
            ("users", os.path.join(data_dir, "users.csv")),
            ("courses", os.path.join(data_dir, "courses.csv")),
            ("enrollments", os.path.join(data_dir, "enrollments.csv")),
            ("payments", os.path.join(data_dir, "payments.csv")),
            ("reviews", os.path.join(data_dir, "reviews.csv")),
        ]

        # 테이블 존재 체크(기존 add_documents가 예외를 던지므로, 여기서도 친절히 처리)
        if not self.store.table_exists():
            return {
                "ok": False,
                "message": "테이블이 존재하지 않습니다.\n먼저 '1. DB 테이블 생성'을 눌러주세요.",
                "inserted_docs": 0,
            }

        total_docs = 0

        for name, path in targets:
            if not os.path.exists(path):
                return {
                    "ok": False,
                    "message": f"CSV 파일을 찾을 수 없습니다: {path}",
                    "inserted_docs": total_docs,
                }

            df = pd.read_csv(path)

            # NaN 정리
            df = df.where(pd.notnull(df), None)

            # 각 row를 "자연어 문장"으로 만들고, 여러 줄을 합쳐 chunk로 저장
            lines = []
            for idx, row in df.iterrows():
                line = self._row_to_nl(name, row, idx)
                if line:
                    lines.append(line)

            # chunking: rows_per_chunk 줄 단위로 합쳐 Document 생성
            docs = []
            for chunk_idx in range(0, len(lines), rows_per_chunk):
                chunk_lines = lines[chunk_idx : chunk_idx + rows_per_chunk]
                chunk_text = "\n".join(chunk_lines).strip()
                if not chunk_text:
                    continue

                docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "source": f"csv:{name}",
                            "file": os.path.basename(path),
                            "chunk_rows": len(chunk_lines),
                            "chunk_index": chunk_idx // rows_per_chunk,
                        },
                    )
                )

            if docs:
                self.store.add_documents(docs)
                total_docs += len(docs)

        return {
            "ok": True,
            "message": f"CSV 데이터 적재 완료! (저장된 chunk 문서 수: {total_docs})",
            "inserted_docs": total_docs,
        }

    def _row_to_nl(self, dataset_name: str, row, idx: int) -> str:
        """
        데이터셋별 row → 자연어 1줄로 변환
        """
        try:
            if dataset_name == "users":
                # user_id, age, gender, job, experience_level, signup_date, marketing_channel
                return (
                    f"[USERS#{idx}] user_id={row.get('user_id')} | "
                    f"age={row.get('age')} | gender={row.get('gender')} | job={row.get('job')} | "
                    f"experience_level={row.get('experience_level')} | signup_date={row.get('signup_date')} | "
                    f"marketing_channel={row.get('marketing_channel')}"
                )

            if dataset_name == "courses":
                # course_id, category, level, price, total_videos, total_duration
                return (
                    f"[COURSES#{idx}] course_id={row.get('course_id')} | "
                    f"category={row.get('category')} | level={row.get('level')} | "
                    f"price={row.get('price')} | total_videos={row.get('total_videos')} | "
                    f"total_duration={row.get('total_duration')}"
                )

            if dataset_name == "enrollments":
                # enrollment_id, user_id, course_id, enroll_date, completion_rate, completed, dropout_day
                return (
                    f"[ENROLLMENTS#{idx}] enrollment_id={row.get('enrollment_id')} | "
                    f"user_id={row.get('user_id')} | course_id={row.get('course_id')} | "
                    f"enroll_date={row.get('enroll_date')} | completion_rate={row.get('completion_rate')} | "
                    f"completed={row.get('completed')} | dropout_day={row.get('dropout_day')}"
                )

            if dataset_name == "payments":
                # user_id, payment_date, amount, payment_type, refunded
                return (
                    f"[PAYMENTS#{idx}] user_id={row.get('user_id')} | "
                    f"payment_date={row.get('payment_date')} | amount={row.get('amount')} | "
                    f"payment_type={row.get('payment_type')} | refunded={row.get('refunded')}"
                )

            if dataset_name == "reviews":
                # user_id, course_id, rating, review_text, review_date
                # review_text는 길 수 있으니 앞부분이 아니라 전체를 담되, 줄바꿈/공백은 정리
                review_text = row.get("review_text")
                if isinstance(review_text, str):
                    review_text = " ".join(review_text.split())
                return (
                    f"[REVIEWS#{idx}] user_id={row.get('user_id')} | course_id={row.get('course_id')} | "
                    f"rating={row.get('rating')} | review_date={row.get('review_date')} | "
                    f"review_text={review_text}"
                )

            return ""

        except Exception:
            return ""

    # 3. RAG 기반 질문/답변
    def rag_query(self, query, k=3):
        results = self.store.similarity_search(query, k=k)

        if results is None:
            return "테이블이 존재하지 않습니다.\n먼저 '1. DB 테이블 생성'을 눌러주세요."

        if len(results) == 0:
            return "관련 문서를 찾을 수 없습니다."

        # 문맥 구성
        context = "\n\n".join(d.page_content for d in results)

        prompt = f"""
        아래 문서를 참고하여 질문에 답하세요.

        [문서 내용]
        {context}

        [질문]
        {query}
        """

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        return response.output_text