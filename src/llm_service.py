import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.llms import YandexGPT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# Загружаем переменные из .env файла
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class LLMService:
    def __init__(self,
                 folder_id: str = None,
                 iam_token: str = None,
                 model: str = "yandexgpt",
                 temperature: float = 0.3,
                 max_tokens: int = 500
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
        self.prompt_template = PromptTemplate.from_template("""
Ответь на вопрос, используя следующие документы:

Контекст:
{context}

Вопрос:
{question}

Ответ:
        """.strip())

        # Цепочка: промпт → LLM
        self.chain: RunnableSequence = self.prompt_template | self.llm

    def generate_response(self, question: str, context: str) -> str:
        """
        Генерирует ответ на вопрос с учётом контекста.
        """
        return self.chain.invoke({
            "question": question,
            "context": context
        })