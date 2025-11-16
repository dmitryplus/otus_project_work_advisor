from langchain_core.prompts import PromptTemplate
from pathlib import Path


class PromptService:
    """
    Сервис для управления промптами. Позволяет загружать шаблоны из файла или из переданной строки.
    """

    def __init__(self,
                 prompt_template: str = None,
                 template_path: str = 'prompts/answer_from_documents.txt'
                 ):
        """
        Инициализация сервиса промптов.

        :param prompt_template: Готовая строка-шаблон. Если указано, используется вместо файла.
        :param template_path: Путь к файлу с шаблоном. По умолчанию — 'prompts/answer_from_documents.txt'.
        """
        self.template = None

        if prompt_template is not None:
            self.template = prompt_template.strip()
        else:
            # Загружаем из файла, если строка не передана
            template_file = Path(template_path)
            if not template_file.exists():
                raise FileNotFoundError(f"Файл шаблона не найден: {template_file.resolve()}")
            self.template = template_file.read_text(encoding="utf-8").strip()

        self.prompt_template = PromptTemplate.from_template(self.template)

    def get_prompt_template(self):
        """
        Возвращает объект PromptTemplate — можно использовать в цепочках LangChain.
        """
        return self.prompt_template