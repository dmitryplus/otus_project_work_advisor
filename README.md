# WorkAdvisor: интеллектуальный ассистент по выполненным работам

# Процесс запуска проекта

### Создаем .env файл

Копируем файл `.env.template` в `.env` и заполняем переменные окружения.

### Запускаем clickhouse сервер
```shell
docker run -d --rm --network=host --name otus-project-clickhouse --ulimit nofile=262144:262144 clickhouse
```

### Запускаем сборку RAG
```shell
.venv/bin/python rag_create.py 
```

### Запускаем телеграм-бот
```shell
.venv/bin/python main.py 
```

# Описание проекта

Структура графа
![graph_mermaid.png](graph_mermaid.png)

# Скрины с демонстрацией работы проекта

## Текстовый запрос в телеграм-бот
![telegram_text.png](img/telegram_text.png)

## Запрос в виде картинки в телеграм-бот
![telegram_image.png](img/telegram_image.png)

## Результат выполнения кода в консоли при текстовом запросе
![console_text.png](img/console_text.png)

## Результат выполнения кода в консоли при запросе картинкой
![console_image.png](img/console_image.png)

## Экспорт данных мониторинга в langfuse
![langfuse.png](img/langfuse.png)