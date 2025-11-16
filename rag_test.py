# rag_test.py
# Проверка работы RAG (эмбединги и clickhouse)

# Импортируем сервисы
from src.embedding_service import YandexEmbeddingService
from src.clickhouse_service import ClickHouseVectorStore

# Инициализация сервисов
embedding_service = YandexEmbeddingService()
vector_store = ClickHouseVectorStore()

# Тестовый запрос
query = "Какие поля есть в карточке мероприятия?"
print(f"🔍 Запрос: {query}\n")

# Генерация эмбеддинга запроса
query_embedding = embedding_service.embed_query(query)
print(f"✅ Эмбеддинг запроса сгенерирован. Длина вектора: {len(query_embedding)}\n")

# Поиск похожих документов
results = vector_store.search_similar(query_embedding, limit=3)
print(f"🎯 Найдено {len(results)} результата(ов):\n")

# Форматированный вывод
for i, doc in enumerate(results, start=1):
    print(f"--- Результат {i} ---")
    print(f"📌 ID:        {doc['id']}")
    print(f"📄 Заголовок: {doc['title']}")
    print(f"🔗 Ссылка:    {doc['url']}")
    print(f"📝 Текст:     {doc['text'][:500]}...")  # Обрезаем длинный текст
    print("\n")