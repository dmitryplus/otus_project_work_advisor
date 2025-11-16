from yandex_cloud_ml_sdk import YCloudML
from typing import List
import os
from dotenv import load_dotenv
from pathlib import Path


# Загружаем переменные из .env файла
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class YandexEmbeddingService:
    """
    Сервис для генерации эмбеддингов с помощью Yandex Cloud ML.
    Переменные окружения берутся из .env файла в корне проекта:
    - FOLDER_ID
    - IAM_TOKEN
    """

    def __init__(self, folder_id: str = None, iam_token: str = None):
        # Позволяет передать значения вручную (для тестов), иначе берёт из .env
        self.folder_id = folder_id or os.getenv("FOLDER_ID")
        self.iam_token = iam_token or os.getenv("IAM_TOKEN")

        if not self.folder_id:
            raise ValueError("FOLDER_ID не указан ни в .env, ни в аргументах.")
        if not self.iam_token:
            raise ValueError("IAM_TOKEN не указан ни в .env, ни в аргументах.")

        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.iam_token)
        self.doc_model = self.sdk.models.text_embeddings("doc")
        self.query_model = self.sdk.models.text_embeddings("query")

    def embed_text(self, text: str) -> List[float]:
        """Генерирует эмбеддинг для документа."""
        return self.doc_model.run(text).embedding

    def embed_query(self, query: str) -> List[float]:
        """Генерирует эмбеддинг для поискового запроса."""
        return self.query_model.run(query).embedding