import os
import io
import time
import base64
from pdf2image import convert_from_path
from openai import OpenAI, RateLimitError
from langchain_core.documents import Document
from app.mariadb_vector_store import MariaDBVectorStore
from app.config import settings

# Poppler Path -> pdf2image 작동
POPPLER_PATH = r"C:\MJ\AiCloud\pj_cloud\poppler-26.09.0\Library\bin"

class PdfOcrService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.vector_store = MariaDBVectorStore()

    #(1) 이미지 > Base64 문자열 변환
    def image_to_base64(self, image):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    #(2) (단일 PDF페이지 > OCR) 모듈
    def ocr_page(self, image):
        img_b64 = self.image_to_base64(image)
        model_name = "gpt-4o" # Vision 모델: gpt-4o

        for attempt in range(5):   # 최대 5번 재시도
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "이 이미지에서 보이는 텍스트를 OCR 방식으로 정확하게 추출해줘. "
                                            "설명 없이 텍스트만 반환해줘."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    }
                                }
                            ]
                        }
                    ]
                )
                return response.choices[0].message.content.strip()
            except RateLimitError:
                wait = 1 + attempt * 1.5  # 1초 → 2.5초 → 4초 → …
                print(f"[RateLimit] Wait {wait} seconds...")
                time.sleep(wait)
            except Exception as e:
                print(f"[OCR Error] {e}")
                raise e

        raise Exception("OCR 실패: 여러 번 재시도했으나 응답을 얻지 못했습니다.")

    #(3) 전체 PDF페이지 > OCR
    def extract_text(self, file_path: str):
        pages = convert_from_path(file_path, poppler_path=POPPLER_PATH)
        full_text = ""
        for idx, page_img in enumerate(pages):
            print(f"OCR Processing Page {idx+1}/{len(pages)}...")
            page_text = self.ocr_page(page_img)
            full_text += f"\n\n# Page {idx+1}\n{page_text}"
            time.sleep(0.5) # 딜레이

        return full_text.strip()

    #(4) 줄바꿈 단위로 분할
    def split_text(self, text: str):
        return [p.strip() for p in text.split("\n") if p.strip()]

    #(5) DB 저장
    def save_pdf_to_vectorstore(self, file_path: str):
        raw_text = self.extract_text(file_path)
        paragraphs = self.split_text(raw_text)
        docs = [
            Document(
                page_content=p,
                metadata={"source": "pdf_ocr"}
            )
            for p in paragraphs
        ]
        self.vector_store.add_documents(docs)

        return len(docs)