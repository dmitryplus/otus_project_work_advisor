# WorkAdvisor: интеллектуальный ассистент по выполненным работам


Запускаем clickhouse сервер
```shell
docker run -d --rm --network=host --name otus-project-clickhouse --ulimit nofile=262144:262144 clickhouse
```

Запускаем сборку RAG
```shell
.venv/bin/python rag_create.py 
```

Структура графа
![graph_mermaid.png](graph_mermaid.png)