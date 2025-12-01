from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from dotenv import load_dotenv
load_dotenv()

# Импорт наших компонентов
from app.schemas import AnalyzeRequest, AnalyzeResponse, Message, DiagnosisResult
from app.services.vision_local import local_vision_service, LocalVisionService
from app.services.llm_service import LLMService

# Инициализация LLM Service
# Мы передаем ему LocalVisionService, чтобы он знал, откуда брать features в 'strict' режиме
llm_service = LLMService(local_vision_service=local_vision_service)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Контекстный менеджер запуска/остановки приложения."""
    # При старте: можно добавить проверку загрузки моделей или подключение к DB
    # LocalVisionService инициализируется при импорте, что соответствует нашему подходу.
    print("🚀 Сервис готов к работе.")
    yield
    # При остановке: можно освободить ресурсы
    print("🛑 Сервис завершает работу.")

app = FastAPI(
    title="Skin Rash LLM Analyzer Prototype",
    description="Единый endpoint для мультимодального анализа высыпаний.",
    version="1.0.0",
    lifespan=lifespan
)

# Добавляем CORS для удобства тестирования с фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse, summary="Единый endpoint для анализа и диалога")
async def analyze_rash(request: AnalyzeRequest):
    """
    Основная логика сервиса. Обрабатывает запрос, определяет режим и этап диалога.
    """
    
    local_features = None
    
    # Шаг 1: Vision Encoder (только если выбран 'strict_local' режим)
    if request.pipeline_mode == "strict_local":
        try:
            # Получаем детальный текстовый отчет о признаках от локальной BLIP-модели
            local_features = local_vision_service.analyze_image(request.image_base64)
            print(f"🔬 Локальные фичи извлечены: {local_features[:80]}...")
        except Exception as e:
            # Если BLIP не смог обработать изображение (например, не хватает памяти/GPU)
            error_msg = f"Ошибка Vision Encoder (BLIP): {str(e)}"
            raise HTTPException(status_code=500, detail=error_msg)
            
    # Шаг 2: LLM Reasoning (обработка логики диалога)
    response = llm_service.process_conversation(request, local_features)
    
    return response

# Для запуска локально (необязательно, можно использовать uvicorn прямо в терминале)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)