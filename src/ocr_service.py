import os
from dotenv import load_dotenv
from pathlib import Path
import requests

# Загружаем переменные из .env файла
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class OCRService:
    def __init__(self, folder_id: str = None, iam_token: str = None,):
        """
        Инициализация OCR-сервиса для распознавания текста с изображений через Yandex Cloud OCR.

        :param folder_id: Идентификатор каталога в Yandex Cloud (берётся из переменной окружения, если не передан).
        :param iam_token: API-ключ для Yandex Cloud (берётся из переменной окружения, если не передан).
        """
        self.folder_id = folder_id or os.getenv("FOLDER_ID")
        self.iam_token = iam_token or os.getenv("IAM_TOKEN")

        if not self.folder_id:
            raise ValueError("FOLDER_ID не указан ни в .env, ни в аргументах.")
        if not self.iam_token:
            raise ValueError("IAM_TOKEN не указан ни в .env, ни в аргументах.")

        self.vision_url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    def analyze_image(self, image_data: str) -> str:
        """
        Отправляет изображение в Yandex Cloud OCR и возвращает распознанный текст.

        :param image_data: Изображение в виде base64-строки.
        :return: Распознанный текст.
        """
        payload = {
            "mimeType": "image",
            "languageCodes": ["ru", "en"],
            "model": "handwritten",
            "content": image_data
        }

        headers = {
            "Authorization": f"Api-Key {self.iam_token}",
            "x-folder-id": self.folder_id
        }

        response = requests.post(self.vision_url, headers=headers, json=payload)
        response.raise_for_status()  # Проверка на HTTP-ошибки

        blocks = response.json().get("result", {}).get("textAnnotation", {}).get("blocks", [])
        text = ""
        for block in blocks:
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    text += word.get("text", "") + " "
                text += "\n"

        return text.strip()