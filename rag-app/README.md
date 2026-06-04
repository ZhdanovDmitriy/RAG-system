# RAG System

## Обзор

Это проект RAG-помощника на основе локального векторного поиска и FastAPI. Он включает в себя:
- PostgreSQL с расширением `pgvector` для хранения векторных эмбеддингов
- Python FastAPI backend для обработки запроса пользователя и тестов
- bootstrapper для генерации эмбеддингов из текстовых источников
- простой frontend на HTML/JS
- nginx для проксирования запросов и создания самоподписанного HTTPS

## Структура проекта

```
rag-app/
  docker-compose.yml
  backend/
    Dockerfile
    requirements.txt
    app/
      __init__.py
      bootstrapper.py
      main.py
      schemas/
        users.py
  frontend/
    Dockerfile
    index.html
    nginx.conf
  nginx/
    Dockerfile
    entrypoint.sh
    nginx.conf
  postgresql/
    init.sql
  sources/
    DM2024_module4.txt
    test_questions.json
```

## Компоненты

### `docker-compose.yml`

Определяет 5 сервисов:
- `postgres` — PostgreSQL с образом `pgvector/pgvector:pg15`
- `backend` — Python-приложение FastAPI
- `bootstrapper` — скрипт для создания эмбеддингов и загрузки данных
- `frontend` — статический фронтенд, обслуживаемый nginx
- `nginx` — обратный прокси и HTTPS-терминатор

### Backend

Файлы:
- `backend/Dockerfile`
- `backend/requirements.txt`
- `backend/app/main.py`
- `backend/app/bootstrapper.py`
- `backend/app/schemas/users.py`

Функции:
- `main.py` запускает FastAPI-приложение и подключается к базе по `DATABASE_URL`
- использует модель `SentenceTransformer("intfloat/multilingual-e5-base")`
- хранит кэш трансформеров в `/models/cache`
- реализует API:
  - `POST /api/question` — поиск похожих фрагментов в таблицах `documents_short` и `documents_long`
  - `GET /api/test/generate` — генерация теста из 5 вопросов
  - `POST /api/test/submit` — проверка ответов на тестовые вопросы

### Bootstrapper

`backend/app/bootstrapper.py`:
- читает `.txt` файлы из `sources/`
- разделяет текст на «короткие» и «длинные» чанки
- генерирует эмбеддинги и сохраняет их в PostgreSQL
- также загружает `test_questions.json` и сохраняет эталонные ответы
- управляется переменной `REBUILD_EMBEDDINGS`

### Frontend

`frontend/index.html` предоставляет интерфейс:
- вкладка «Вопрос-ответ» для отправки запроса на `/api/question`
- вкладка «Тест» для генерации и проверки теста
- JavaScript работает через `fetch('/api/...')`

### nginx

`nginx/entrypoint.sh` генерирует самоподписанный SSL-сертификат при старте.
`nginx/nginx.conf` проксирует:
- `/api/` → `backend:8000`
- `/` → `frontend:80`

### PostgreSQL

`postgresql/init.sql` создаёт:
- таблицы `documents_short`, `documents_long` и `test_question_embeddings`
- индексы для `pgvector`
- таблицу `test_questions`

## Зависимости

В `backend/requirements.txt`:
- fastapi
- uvicorn
- psycopg2-binary
- pgvector
- sentence-transformers
- transformers
- torch
- numpy
- tqdm

## Запуск на Windows 11

### Требуется

1. Docker Desktop
   - Linux-контейнеры
   - WSL 2
2. Опционально Git, если вы клонируете репозиторий

### Команды

1. Откройте терминал в `rag-app`:
   ```powershell
   cd "..\RAG-system\rag-app"
   ```

2. Создайте файл `.env` рядом с `docker-compose.yml`:
   ```env
   POSTGRES_DB=ragdb
   POSTGRES_USER=raguser
   POSTGRES_PASSWORD=secret
   DATABASE_URL=postgresql://raguser:secret@postgres:5432/ragdb
   REBUILD_EMBEDDINGS=true
   ```

3. Запустите сервисы:
   ```powershell
   docker compose up --build
   ```

4. После запуска откройте в браузере:
   - `https://localhost:8443`

> Важно: nginx слушает HTTPS на хост-порту `8443`, а HTTP на `8080` только редиректит на HTTPS.

### Остановить

```powershell
docker compose down
```

## Как работает запрос

### Вопрос-ответ

1. Фронтенд отправляет `POST /api/question` с телом:
   ```json
   { "question": "Ваш вопрос" }
   ```
2. Бэкенд создаёт эмбеддинг вопроса и ищет ближайшие фрагменты по `pgvector`.
3. Возвращает два списка:
   - `short` — короткие фрагменты
   - `long` — длинные фрагменты

### Тест

1. Фронтенд запрашивает `GET /api/test/generate`
2. Бэкенд выбирает 5 случайных вопросов из таблицы `test_questions`
3. После отправки ответов фронтенд посылает `POST /api/test/submit`
4. Бэкенд сравнивает эмбеддинги ответа с эталонами и возвращает оценку

## Источники данных

- `sources/DM2024_module4.txt` — основной текст
- `sources/test_questions.json` — структура тестовых вопросов и ответов

## Что можно улучшить

- добавить `README` в корень репозитория
- вынести frontend-скрипт в отдельный JS-файл
- добавить healthcheck для backend
- настроить переменные окружения через `.env.example`
- добавить документацию API и примеры запросов

---

## Быстрый чек

- `docker compose ps` — проверка запущенных контейнеров
- `docker compose logs -f nginx` — логи nginx
- `docker compose logs -f backend` — логи backend
- `docker compose logs -f bootstrapper` — генерация эмбеддингов
