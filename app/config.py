import os
from dotenv import load_dotenv

# Загружаем переменные из .env файла
load_dotenv()

class Settings:
    """Настройки приложения."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    if not OPENAI_API_KEY:
        print("🚨 ВНИМАНИЕ: OPENAI_API_KEY не установлен. API-запросы не будут работать.")

settings = Settings()