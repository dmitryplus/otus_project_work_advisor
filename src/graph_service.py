from typing import TypedDict, Annotated
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

from src.rag_service import RAGService
from src.llm_service import LLMService
from src.ocr_service import OCRService
from src.prompt_service import PromptService


# --- Состояние графа ---
class GraphState(TypedDict):
    query: str
    relevants: list
    context: str
    response: str
    image_data: str
    prompt_template: PromptTemplate


# --- Узлы графа ---
def retrieve_rag_node(state: GraphState) -> dict:
    """Поиск релевантных документов с помощью RAG."""
    print("🔍 Поиск релевантных документов...")

    if not state["query"]:
        return {"response": "❌ Запрос пуст. Невозможно выполнить поиск."}

    rag_service = RAGService()
    relevants = rag_service.search_relevant_documents(state["query"], top_k=3)

    if not relevants:
        return {"response": "❌ Релевантные документы не найдены."}

    context = rag_service.format_context(relevants)

    return {
        "relevants": relevants,
        "context": context,
    }


def decide_to_generate(state: GraphState) -> str:
    """Решает, переходить ли к генерации или завершить граф."""
    if state["response"]:
        return "end"
    return "generate"


def generate_node(state: GraphState) -> dict:
    """Генерация ответа с помощью LLM и добавление информации 'Подробнее в задачах'."""

    if "prompt_template" not in state or state["prompt_template"] is None:
        return {"response": "❌ Не удалось сгенерировать ответ: отсутствует шаблон запроса."}

    print("🧠 Генерация ответа...")

    llm_service = LLMService(model="yandexgpt-lite", prompt_template=state["prompt_template"])
    response = llm_service.generate_response(question=state["query"], context=state["context"], state=state).strip()

    full_response = response
    if state["relevants"]:
        full_response += "\n\n📚 Подробнее в задачах:\n"
        for doc in state["relevants"]:
            full_response += f"• {doc['title']}\n"
            full_response += f"  {doc['url']}\n"

    print("✅ Ответ получен")

    return {"response": full_response}


def ocr_image_node(state: GraphState) -> dict:
    """Распознавание текста на изображении и сохранение в query."""
    if not state["image_data"]:
        return {}

    print("🖼️ Распознавание текста на изображении...")
    ocr_service = OCRService()

    try:
        base64_data = state["image_data"].split(",")[1] if state["image_data"].startswith("data:image") else state["image_data"]
        recognized_text = ocr_service.analyze_image(base64_data)

        if not recognized_text or not recognized_text.strip():
            return {"response": "⚠️ Не удалось распознать текст на изображении."}

        return {"query": recognized_text}

    except Exception as e:
        return {"response": f"⚠️ Ошибка при распознавании изображения: {e}"}


def route_image_or_query(state: GraphState) -> str:
    """Решает, нужно ли обрабатывать изображение или сразу переходить к поиску."""
    if state["image_data"]:
        return "ocr"
    return "retrieve"


def init_prompt_template_node(state: GraphState) -> dict:
    """Инициализация prompt_template в зависимости от наличия изображения."""
    template_path = 'prompts/answer_from_documents.txt'
    if state["image_data"]:
        template_path = 'prompts/text_from_image_to_query.txt'

    prompt_service = PromptService(template_path=template_path)
    prompt_template = prompt_service.get_prompt_template()

    print("📝 Инициализация шаблона подстановки...")
    return {"prompt_template": prompt_template}


# --- Построение графа ---
class GraphService:
    def __init__(self):
        self.app = self._compile_graph()

    def _compile_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("ocr", ocr_image_node)
        workflow.add_node("retrieve", retrieve_rag_node)
        workflow.add_node("init_prompt", init_prompt_template_node)
        workflow.add_node("generate", generate_node)

        workflow.set_conditional_entry_point(
            route_image_or_query,
            {
                "ocr": "ocr",
                "retrieve": "retrieve"
            }
        )

        workflow.add_edge("ocr", "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            decide_to_generate,
            {
                "generate": "init_prompt",
                "end": END
            }
        )

        workflow.add_edge("init_prompt", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def invoke(self, inputs: dict) -> dict:
        """Запуск графа с переданными входными данными."""
        return self.app.invoke(inputs)

    def get_mermaid_code(self) -> str:
        """Возвращает Mermaid-код для визуализации графа (можно вставить в VS Code или Mermaid Live Editor)."""
        try:
            return self.app.get_graph().draw_mermaid()
        except Exception as e:
            return f"Ошибка генерации Mermaid-диаграммы: {e}"