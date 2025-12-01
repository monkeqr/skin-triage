from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# Тип сообщения в диалоге
class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

# Входящий запрос от клиента
class AnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Изображение в формате Base64")
    pipeline_mode: Literal["strict_local", "native_gpt4o"] = Field(
        "native_gpt4o", 
        description="Выбор режима: локальный энкодер или нативный GPT-4o"
    )
    # История диалога. Если пустая — значит это первое обращение.
    conversation_history: List[Message] = Field(default_factory=list)

# Структура ответа LLM (структурированный вывод)
# Мы заставим LLM отдавать JSON строго этой структуры
class DiagnosisResult(BaseModel):
    diagnosis_options: List[str] = Field(..., description="2-3 возможных диагноза")
    key_differences: str = Field(..., description="Ключевые отличия между ними")
    symptoms_to_check: List[str] = Field(..., description="Какие симптомы уточнить")
    diagnostic_methods: List[str] = Field(..., description="Рекомендованные методы диагностики")
    disclaimer: str = Field("Это не медицинский диагноз. Обратитесь к врачу.", description="Обязательный дисклеймер")

# Ответ нашего API клиенту
class AnalyzeResponse(BaseModel):
    stage: Literal["questioning", "diagnosis"] # Текущий этап: вопросы или вердикт
    content: str # Текст вопроса или пусто, если вердикт готов
    final_diagnosis: Optional[DiagnosisResult] = None # Заполняется только в конце
    updated_history: List[Message] # Обновленная история для возврата клиенту