from src.rag_service import RAGService
from src.llm_service import LLMService

# --- Инициализация сервисов ---
rag_service = RAGService()
llm_service = LLMService(model="yandexgpt-lite")

# --- Основной запрос ---
query = "Что нужно сделать с акселераторами?"

print("🔍 Поиск релевантных документов...")
relevants = rag_service.search_relevant_documents(query, top_k=3)

if not relevants:
    print("❌ Релевантные документы не найдены.")
else:
    # Формируем контекст через RAGService
    context = rag_service.format_context(relevants)

    # Генерируем ответ
    print("🧠 Генерация ответа...")
    response = llm_service.generate_response(question=query, context=context)

    # Выводим результат
    print("\n" + "="*50)
    print("Вопрос:")
    print(query)

    print("\nОтвет:")
    print(response.strip())

    print(f"\n\n📚 Подробнее в задачах:")
    for doc in relevants:
        print(f"• {doc['title']}")
        print(f"  {doc['url']}")
    print("="*50)