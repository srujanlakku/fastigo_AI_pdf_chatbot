import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", os.path.join(BASE_DIR, "chromadb"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_PROJECT = os.getenv("GOOGLE_API_PROJECT", "")
GOOGLE_API_LOCATION = os.getenv("GOOGLE_API_LOCATION", "us-central1")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
PDF_OCR_ENABLED = os.getenv("PDF_OCR_ENABLED", "True").lower() in ("true", "1", "yes")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


def validate_api_settings() -> None:
    if not GOOGLE_API_KEY:
        raise EnvironmentError(
            "Missing required environment variable: GOOGLE_API_KEY. "
            "Please add it to your .env file or environment."
        )
