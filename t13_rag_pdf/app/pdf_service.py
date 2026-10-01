from pypdf import PdfReader
from langchain_core.documents import Document
from app.config import settings
from app.mariadb_vector_store import MariaDBVectorStore

class PdfService:
    def __init__(self):
        self.vector_store = MariaDBVectorStore()

    # (1) PDF 텍스트 추출
    def extract_text(self, file_path: str):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        return text.strip()

    # (2) 줄바꿈 단위로 텍스트 분할
    def split_text(self, text: str):
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        return paragraphs

    # (3) 벡터를 저장
    def save_pdf_to_vectorstore(self, file_path: str):
        raw_text = self.extract_text(file_path)
        paragraphs = self.split_text(raw_text)

        docs = [
            Document(page_content=p, metadata={"source": "pdf"})
            for p in paragraphs
        ]

        self.vector_store.add_documents(docs)

        return len(docs)