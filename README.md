# 1. Создание виртуального окружения
python3 -m venv venv

# 2. Активация окружения
source venv/bin/activate  # Для Linux/macOS
# .\venv\Scripts\activate # Для Windows CMD

# 3. Установка зависимостей (включая PyTorch и transformers)
pip install -r requirements.txt

# 4. Установка ключа (в файле .env)
# OPENAI_API_KEY="sk-..."

# 5. Запуск FastAPI-сервиса
uvicorn app.main:app --reload