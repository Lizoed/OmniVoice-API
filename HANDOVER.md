# OmniVoice API - Быстрый старт


## 1. Клонирование репозитория модели
Для полноценной работы сервера вам понадобится исходный код базовой модели OmniVoice. 
1. Скачайте репозиторий: [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice/tree/master).
2. Распакуйте или склонируйте его в папку `omnivoice-master` так, чтобы она находилась в корне вашего проекта (рядом с папкой `api/`).

## 2. Установка зависимостей
Создайте виртуальное окружение и установите все пакеты из файла `requirements.txt`. В нём уже прописан локальный путь к исходному коду модели (`-e ./omnivoice-master`) и необходимые библиотеки для STT и работы API.

**Для Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Для Linux / MacOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Запуск сервера
Сервер написан на FastAPI. Для запуска приложения (по умолчанию на порту 8000), выполните команду:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
После успешного старта вы сможете открыть интерактивную Swagger-документацию по адресу: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
