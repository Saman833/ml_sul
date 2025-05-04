FROM python:3.12-slim

# Установка необходимых зависимостей для дальнейшей установки psycopg2
RUN apt-get update && apt-get install -y \
    postgresql-client \
    gcc \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Скопируем и установим зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем папку для логов
RUN mkdir -p /app/logs
VOLUME ["/app/logs"]

# Скопируем сам код
COPY . .

# Указываем порт, на котором работает приложение
EXPOSE 8672

# Запуск приложения
CMD ["python3", "main.py"]
