# main.py
# Точка входа: запуск бота, и передача данных в граф
import os
import base64
import time
import requests
import telebot
from dotenv import load_dotenv

from src.bot import keyboards
from src.bot.structure import create_bot
from src.graph_service import GraphService, GraphState

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")


def handler(event, _):
    try:
        message = telebot.types.Update.de_json(event["body"])

        if message.message.from_user.username == 'dmitry_plus':
            bot = create_bot(BOT_TOKEN)

            inputs: GraphState = {
                "query": "",
                "relevants": [],
                "context": "",
                "response": "",
                "image_data": "",
                "prompt_template": None,
            }

            if message.message.content_type == 'photo':
                file_id = message.message.photo[-1].file_id
                file_info = bot.get_file(file_id)
                downloaded_file = bot.download_file(file_info.file_path)
                image_data = base64.b64encode(downloaded_file).decode('utf-8')

                inputs["image_data"] = f"data:image/png;base64,{image_data}"

                graph_service = GraphService()

                answer = graph_service.invoke(inputs)

                result = answer["response"]

            if message.message.content_type == 'text':
                inputs["query"] = message.message.text

                graph_service = GraphService()

                answer = graph_service.invoke(inputs)

                result = answer["response"]

            bot.send_message(
                message.message.chat.id,
                result,
                reply_markup=keyboards.EMPTY,
            )

    finally:
        return {
            "statusCode": 200,
            "body": "!",
        }

def get_updates(offset=0):

    result = requests.get(f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates?offset={offset}').json()

    return result['result']

def run():

    updates = get_updates()
    update_id = updates[-1]['update_id'] # Присваиваем ID последнего отправленного сообщения боту

    while True:
        time.sleep(2)
        messages = get_updates(update_id) # Получаем обновления
        for message in messages:

            if update_id < message['update_id']:
                update_id = message['update_id']

                event = {'body': message}

                handler(event, '')

if __name__ == '__main__':
    run()

