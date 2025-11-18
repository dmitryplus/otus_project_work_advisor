import json
import os
import time

from dotenv import load_dotenv
from pathlib import Path

from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

from langfuse import get_client, Langfuse
from langfuse.langchain import CallbackHandler

from src.prompt_service import PromptService
from src.semantic_coverage_service import SemanticCoverageService

# Загружаем переменные из .env файла
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class LLMService:
    def __init__(self,
                 folder_id: str = None,
                 iam_token: str = None,
                 model: str = "yandexgpt",
                 temperature: float = 0.3,
                 max_tokens: int = 500,
                 prompt_template: PromptTemplate = None,
                 langfuse_secret_key: str = None,
                 langfuse_public_key: str = None,
                 langfuse_host: str = None
                 ):

        self.folder_id = folder_id or os.getenv("FOLDER_ID")
        self.iam_token = iam_token or os.getenv("IAM_TOKEN")

        if not self.folder_id:
            raise ValueError("FOLDER_ID не указан ни в .env, ни в аргументах.")
        if not self.iam_token:
            raise ValueError("IAM_TOKEN не указан ни в .env, ни в аргументах.")

        self.llm = YandexGPT(
            folder_id=self.folder_id,
            iam_token=self.iam_token,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if not prompt_template:
            self.prompt_service = PromptService()
            self.prompt_template = self.prompt_service.get_prompt_template()
        else:
            self.prompt_template = prompt_template

        # Цепочка: промпт → LLM
        self.chain: RunnableSequence = self.prompt_template | self.llm

        # Настройка Langfuse
        self.langfuse = Langfuse(
            secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
            public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
            host=langfuse_host or os.getenv("LANGFUSE_HOST"),
        )

        self.semantic_coverage_service = SemanticCoverageService()

    def generate_response(self, question: str, context: str, state: dict) -> str:
        """
        Генерирует ответ на вопрос с учётом контекста.
        Добавляет мониторинг через Langfuse.
        """

        langfuse = get_client()

        # Generate deterministic trace ID from external system
        external_request_id = "home_request_15"
        predefined_trace_id = Langfuse.create_trace_id(seed=external_request_id)

        langfuse_handler = CallbackHandler()

        # Отслеживание времени выполнения
        start_time = time.time()  # Начало всей операции

        # Разделяем документы и оценки
        documents = state["relevants"]
        scores = [doc.get("score", 0.0) for doc in documents]

        # Отслеживание количества и релевантности
        document_count = len(documents)
        average_relevance = sum(scores) / len(scores) if document_count > 0 else 0

        with langfuse.start_as_current_span(
                name="langchain-request-1",
                trace_context={"trace_id": predefined_trace_id}
        ) as span:
            # Формируем input для логирования
            input_data = {
                "question": question,
                "context": context,
                "document_count": document_count,
                "average_relevance": average_relevance,
                "relevance_scores": scores
            }

            print("\n" + "="*60)
            print("🟢 INPUT:")
            print("="*60)
            print(json.dumps(input_data, ensure_ascii=False, indent=4))
            print("="*60)

            span.update_trace(user_id="user_123", input=input_data)

            # Замер времени начала выполнения LLM-цепочки
            llm_start_time = time.time()

            response = self.chain.invoke(
                {
                    "question": question,
                    "context": context
                },
                config={
                    "callbacks": [langfuse_handler],
                    "metadata": {
                        "langfuse_user_id": "random-user",
                        "langfuse_session_id": "random-session",
                        "langfuse_tags": ["random-tag-1", "random-tag-2"],
                        "model_name": "yandexgpt"
                    }
                }
            )

            # Замер времени окончания выполнения LLM-цепочки
            llm_end_time = time.time()
            llm_duration = llm_end_time - llm_start_time  # Время выполнения LLM

            # Общее время выполнения
            total_end_time = time.time()
            total_duration = total_end_time - start_time  # Общее время операции

            semantic_coverage = self.semantic_coverage_service.calculate(context, response, documents)

            output = {
                "response": response,
                "llm_duration_seconds": llm_duration,  # Время работы LLM
                "total_duration_seconds": total_duration,
                "document_count": document_count,
                "relevance_scores": scores,
                "average_relevance": average_relevance,
                "semantic_coverage": semantic_coverage
            }

            print("\n" + "🟩 OUTPUT:")
            print("="*60)
            print(json.dumps(output, ensure_ascii=False, indent=4))
            print("="*60 + "\n")

        span.update_trace(output=output, )

        return response
