# rag_create.py
import os
import pandas as pd


# Очищаем хранилище, создаем таблицу
from src.clickhouse_service import ClickHouseVectorStore

vector_store = ClickHouseVectorStore()
vector_store.create_table()


# Исходные документы
documents = [
    {
        "id": 166213,
        "title": "Изменить формат общего шаблона excel при скачивании файлов",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/166213/"
    },
    {
        "id": 151543,
        "title": "Реализовать страницу со списком акселераторов",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/151543/"
    },
    {
        "id": 196677,
        "title": "[Bug][Major] Доработать статусы заявок в Акселераторах",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/196677/"
    },
    {
        "id": 181744,
        "title": "Реализация группы прав \"Менеджер Акселератора партнер\"",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/181744/"
    },
    {
        "id": 190897,
        "title": "Доработка кабинета Акселератора, добавление внешних модераторов",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/190897/"
    },
    {
        "id": 184464,
        "title": "Доработка ролевой модели в Акселерации (Демо)",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/184464/"
    },
    {
        "id": 187834,
        "title": "Добавление рассылки видеозаписи прошедшего мероприятия",
        "url": "https://inside.notamedia.ru/company/personal/user/4658/tasks/task/view/187834/"
    },
]

# Добавляем context из .md файлов
for doc in documents:
    file_name = f"{doc['id']}.md"
    file_path = os.path.join("tasks", file_name)

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            doc["context"] = f.read()
    else:
        doc["context"] = ""

    doc["search"] = f"{doc['title']} {doc['context']}"

# Преобразуем в DataFrame
df = pd.DataFrame(documents)

# Используем RAGService
from src.rag_service import RAGService

rag_service = RAGService(chunk_size=1000, chunk_overlap=0)
rag_service.ingest(df)