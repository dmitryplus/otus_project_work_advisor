import base64
import os

from src.ocr_service import OCRService
from src.rag_service import RAGService
from src.llm_service import LLMService
from src.prompt_service import PromptService


# --- Инициализация сервисов ---
rag_service = RAGService()
llm_service = LLMService(model="yandexgpt-lite")

def image_analyze(image_data: str) -> str:
    ocr_service = OCRService()
    return ocr_service.analyze_image(image_data)


image_path = "img/users_count.png"

# Проверяем, существует ли файл
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Файл не найден: {image_path}. Убедитесь, что путь корректен.")

# Чтение изображения и конвертация в base64
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Вызов функции распознавания
recognized_text = image_analyze(encoded_image)

print("Распознанный текст:")
print(recognized_text)


# --- Инициализация сервисов ---
rag_service = RAGService()
prompt_service = PromptService(template_path='prompts/text_from_image_to_query.txt')
prompt_template = prompt_service.get_prompt_template()
llm_service = LLMService(model="yandexgpt-lite", prompt_template=prompt_template)

# --- Основной запрос ---
query = recognized_text

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