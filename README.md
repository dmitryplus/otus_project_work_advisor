# WorkAdvisor: интеллектуальный ассистент по выполненным работам


Запускаем clickhouse сервер
```shell
docker run -d --rm --network=host --name otus-project-clickhouse --ulimit nofile=262144:262144 clickhouse
```