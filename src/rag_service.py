# src/rag_service.py
from typing import List, Dict, Any
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embedding_service import YandexEmbeddingService
from src.clickhouse_service import ClickHouseVectorStore


class RAGService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        self.embedding_service = YandexEmbeddingService()
        self.vector_store = ClickHouseVectorStore()

    def prepare_documents(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Загружает документы из DataFrame, разбивает на чанки и генерирует эмбеддинги.
        Возвращает список словарей с метаданными и эмбеддингами.
        """
        # Загружаем документы
        loader = DataFrameLoader(df, page_content_column='search')
        docs = loader.load()

        # Разбиваем на чанки
        texts = self.text_splitter.split_documents(docs)

        embeddings_data = []
        for doc in texts:
            text = doc.page_content
            metadata = doc.metadata
            embedding = self.embedding_service.embed_text(text)

            embeddings_data.append({
                "id": metadata.get("id"),
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "text": text,
                "embedding": embedding
            })

            print(f"✅ Эмбеддинг для документа {metadata.get('id')} сгенерирован. Длина: {len(embedding)}")

        return embeddings_data

    def ingest(self, df: pd.DataFrame):
        """
        Полный процесс: подготовка документов и сохранение в ClickHouse.
        """
        documents_with_embeddings = self.prepare_documents(df)
        self.vector_store.add_documents(documents_with_embeddings)
        print(f"✅ Векторизация и сохранение завершены. Загружено {len(documents_with_embeddings)} чанков.")