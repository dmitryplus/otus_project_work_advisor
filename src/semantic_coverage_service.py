import numpy as np
from typing import List

from dotenv import load_dotenv
from pathlib import Path

from src.embedding_service import YandexEmbeddingService

# Загружаем переменные из .env файла
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class SemanticCoverageService:
    def __init__(self):

        self.embedding_service = YandexEmbeddingService()


    def _get_embedding(self, text: str) -> List[float]:
        """Генерирует эмбеддинг для текста через YandexGPT (query-модель)."""
        return self.embedding_service.embed_query(text)


    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Вычисляет косинусное сходство между двумя векторами."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


    def calculate(self, context: str, response: str, relevant_docs: List[dict]) -> float:
        """
        Рассчитывает семантическое покрытие ответа:
        - Усредняет эмбеддинги релевантных документов → context_embedding
        - Получает эмбеддинг ответа → response_embedding
        - Возвращает косинусное сходство.
        """
        if not relevant_docs or not response.strip():
            return 0.0


        # Собираем эмбеддинги всех релевантных чанков
        doc_embeddings = []
        for doc in relevant_docs:
            try:
                emb = self.embedding_service.embed_text(doc["text"])
                doc_embeddings.append(emb)
            except:
                continue

        if not doc_embeddings:
            return 0.0

        # Усреднённый вектор контекста
        context_embedding = np.mean(np.array(doc_embeddings), axis=0)

        # Эмбеддинг ответа
        try:
            response_embedding = np.array(self.embedding_service.embed_query(response))
        except:
            return 0.0

        # Косинусное сходство
        similarity = self._cosine_similarity(context_embedding, response_embedding)
        return float(similarity)  # от -1 до 1, но обычно ~ от 0 до 1