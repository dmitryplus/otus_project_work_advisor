# src/clickhouse_service.py

import clickhouse_connect
from typing import List, Dict


class ClickHouseVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        username: str = "default",
        password: str = "",
        table_name: str = "tasks_clickhouse"
    ):
        """
        Инициализация клиента ClickHouse и настройка имени таблицы.

        Parameters:
            host (str): Адрес ClickHouse сервера.
            username (str): Имя пользователя.
            password (str): Пароль.
            table_name (str): Имя таблицы для хранения документов.
        """
        self.client = clickhouse_connect.get_client(host=host, username=username, password=password)
        self.table_name = table_name

    def create_table(self):
        """Создаёт таблицу для хранения документов и эмбеддингов."""
        self.client.command(f"DROP TABLE IF EXISTS {self.table_name}")
        self.client.command(f"""
        CREATE TABLE {self.table_name} (
            id UInt64,
            text String,
            title String,
            url String,
            embedding Array(Float32)
        ) ENGINE = MergeTree() ORDER BY id
        """)

    def add_documents(self, documents: List[Dict]):
        """Добавляет документы (уже с готовыми эмбеддингами)."""
        rows = []
        for doc in documents:
            rows.append((
                doc["id"],
                doc["text"],
                doc["title"],
                doc["url"],
                doc["embedding"]
            ))
        self.client.insert(self.table_name, rows)

    def search_similar(self, query_embedding: List[float], limit: int = 2) -> List[Dict]:
        """
        Ищет ближайшие по косинусному расстоянию документы.

        Parameters:
            query_embedding (List[float]): Вектор-эмбеддинг поискового запроса.
            limit (int): Количество возвращаемых документов.

        Returns:
            List[Dict]: Список найденных документов с полями id, title, url, text.
        """
        query_str = ",".join(map(str, query_embedding))
        result = self.client.query(f"""
            SELECT id, title, url, text, cosineDistance(embedding, [{query_str}]) AS dist
            FROM {self.table_name}
            ORDER BY dist ASC
            LIMIT {limit}
        """)

        return [
            {
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "text": row[3]
            }
            for row in result.result_rows
        ]