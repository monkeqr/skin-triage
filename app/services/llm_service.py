import os
import json
from typing import List, Optional, Union
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import Dict, Any
# Импортируем наши схемы из app/schemas.py
from app.schemas import AnalyzeRequest, AnalyzeResponse, Message, DiagnosisResult

# Получаем API ключ из переменных окружения
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL_NAME = "gpt-4o-mini" # Используем мини-версию для скорости и экономии, она отлично справляется с JSON.

class LLMService:
    """
    Класс для управления логикой диалога с GPT-4o.
    Отвечает за промпты, выбор режима (Native/Strict) и парсинг ответов.
    """
    def __init__(self, local_vision_service: Optional['LocalVisionService'] = None):
        # LocalVisionService нужен, только если мы работаем в 'strict_local' режиме
        self.local_vision = local_vision_service
        
    def _get_system_prompt(self, is_final_stage: bool) -> str:
        """Определяет роль LLM и правила игры."""
        stage = "для запроса 2-3 уточняющих вопросов"
        if is_final_stage:
            stage = "для генерации финального дифференциального диагноза в формате JSON"
        
        return (
            "You are a professional AI Dermatology Differential Diagnosis Assistant. "
            "Your task is to analyze the user's skin condition based on provided visual features and dialogue history. "
            "You MUST adhere strictly to the following process:\n"
            "You are a dermatology differential-diagnosis assistant."

            "RULES:"
            "1. If user provided an image (or visual features) AND no answers yet → ask 2-3 clarifying questions."
            "2. If user sends [ANSWERS], you MUST call the function submit_diagnosis."
            "3. Never output diagnosis without calling the tool."
            "4. Tool output must strictly follow the JSON schema."

            "5. **CRITICAL:** Do NOT provide medical advice or final diagnoses. Always include a strong disclaimer.\n"
            f"Current goal: {stage}"
        )

    def _get_diagnosis_tool(self):
        """Возвращает Pydantic схему, конвертированную в формат Tool для OpenAI."""
        # Используем современный метод Pydantic V2 для получения схемы
        schema = DiagnosisResult.model_json_schema()
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_diagnosis",
                    "description": "Используй эту функцию, когда ты готов выдать финальный дифференциальный диагноз и рекомендации.",
                    "parameters": schema,
                },
            }
        ]

    def _prepare_messages(self, request: AnalyzeRequest, features_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Собирает полный список сообщений для отправки в API, включая системный промпт
        и правильное включение изображения (для Native) или текста фич (для Strict).
        """
        
        # Определяем, это первый запрос или продолжение диалога
        is_initial_request = len(request.conversation_history) == 0
        
        # Создаем системный промпт
        is_final_stage = len(request.conversation_history) >= 2
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._get_system_prompt(is_final_stage)}
        ]

        # Добавляем стартовый запрос/контекст
        if is_initial_request:
            user_content_parts = []

            if request.pipeline_mode == "native_gpt4o":
                # Вариант B (Native): Передаем изображение в Base64
                user_content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}
                })
                user_content_parts.append({
                    "type": "text",
                    "text": "Please analyze this image and ask 2-3 necessary questions."
                })

            elif request.pipeline_mode == "strict_local":
                # Вариант A (Strict): Передаем только текст фич
                user_content_parts.append({
                    "type": "text",
                    "text": f"Please analyze the following visual features and ask 2-3 necessary questions:\n\n---\n{features_text}"
                })

            messages.append({"role": "user", "content": user_content_parts})

        # Добавляем историю диалога, если это не первый запрос
        if not is_initial_request:
            messages.extend([m.model_dump() for m in request.conversation_history])

        return messages


    def process_conversation(self, request: AnalyzeRequest, local_features: Optional[str] = None) -> AnalyzeResponse:
        """Основной метод обработки запроса."""
        
        # 1. Сборка контекста и сообщений
        messages = self._prepare_messages(request, local_features)
        
        # 2. Определение инструментов (Tools) для GPT
        tools = self._get_diagnosis_tool()
        
        # 3. Вызов API
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto", # Пусть модель сама решит, когда использовать функцию submit_diagnosis
            )
        except Exception as e:
            return AnalyzeResponse(
                stage="diagnosis", 
                content=f"Error: OpenAI API call failed: {str(e)}", 
                updated_history=request.conversation_history
            )

        assistant_message = response.choices[0].message
        
        # 4. Обработка ответа
        # Если модель решила вызвать функцию (значит, пора ставить диагноз)
        if assistant_message.tool_calls:
            
            # Получаем аргументы из первого (и единственного) вызова функции
            tool_call = assistant_message.tool_calls[0]
            
            if tool_call.function.name == "submit_diagnosis":
                try:
                    # Парсим JSON-строку в нашу Pydantic модель
                    diagnosis_data = json.loads(tool_call.function.arguments)
                    final_diagnosis = DiagnosisResult(**diagnosis_data)
                    
                    # Добавляем "ответ" ассистента в историю
                    updated_history = [
                        *request.conversation_history, 
                        Message(role="assistant", content="Diagnosis submitted via tool call.")
                    ]
                    
                    return AnalyzeResponse(
                        stage="diagnosis",
                        content="Финальный дифференциальный диагноз предоставлен.",
                        final_diagnosis=final_diagnosis,
                        updated_history=updated_history
                    )
                except (json.JSONDecodeError, ValidationError) as e:
                    # Если парсинг не удался, просим модель исправить JSON (в реальном приложении)
                    # Но в прототипе просто сообщим об ошибке
                    return AnalyzeResponse(
                        stage="diagnosis", 
                        content=f"Error: Failed to parse final diagnosis JSON from LLM: {str(e)}",
                        updated_history=request.conversation_history
                    )
        
        # Если модель не вызвала функцию (значит, она задает вопросы)
        else:
            assistant_text = assistant_message.content
            
            # Обновляем историю
            updated_history = [
                *request.conversation_history,
                Message(role="assistant", content=assistant_text)
            ]
            
            return AnalyzeResponse(
                stage="questioning",
                content=assistant_text,
                updated_history=updated_history
            )

# Примечание: В main.py мы будем создавать экземпляр LLMService