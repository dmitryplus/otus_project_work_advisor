# text_query_test.py
# Проверка работы тестовых запроса с использованием langgraph и early stopping
import base64
import os

from src.graph_service import GraphService, GraphState


if __name__ == "__main__":
    # Пример текстового запроса
    query = "Что нужно сделать с акселераторами?"

    # Раскомментируйте, если нужно обработать изображение
    image_path = "img/users_count.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    inputs: GraphState = {
        "query": query,
        "relevants": [],
        "context": "",
        "response": "",
        #"image_data": "",
         "image_data": f"data:image/png;base64,{encoded_image}",
        "prompt_template": None,
    }

    print("🚀 Запуск обработки запроса с использованием langgraph...")
    graph_service = GraphService()

    # print("\n📋 Mermaid-код для визуализации (скопируйте в Mermaid Live Editor):")
    # print(graph_service.get_mermaid_code())

    result = graph_service.invoke(inputs)

    print("\n" + "=" * 50)
    print("Вопрос:")
    print(inputs["query"] or "Распознанный текст (из изображения):")
    print("\nОтвет:")
    print(result["response"])
    print("=" * 50)