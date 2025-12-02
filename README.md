# Skin Rash LLM Analyzer Prototype

Прототип сервиса для мультимодального анализа кожных высыпаний. Позволяет загружать изображение, автоматически извлекать признаки через локальную BLIP‑модель или напрямую использовать GPT‑4o, вести диалог с LLM и формировать дифференциальный диагноз.

---

## 1. Используемые модели

### BLIP (Bootstrapped Language-Image Pretraining)
Локальная модель для извлечения текстовых фич из изображения.

### LLM (GPT‑4o / GPT‑4o‑mini)
Используется для ведения диалога и формирования дифференциального диагноза.

---

## 2. Pipeline сервиса

Сервис поддерживает два режима работы.

---

## 2.1. strict_local (BLIP + LLM)

### Диаграмма процесса

```
   ┌─────────────┐      ┌─────────────┐       ┌─────────────────┐     ┌──────────────┐
   │   Клиент    │ ---> │  Изображение│ --->  │ BLIP (локально) │ --->│Текстовые фичи│
   └─────────────┘      └─────────────┘       └─────────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌─────────────────┐
                                                              │     LLM (GPT)   │
                                                              └─────────────────┘
                                                                      │
                                                                      ▼
                                                        ┌─────────────────────────┐
                                                        │    Вопросы или диагноз  │
                                                        └─────────────────────────┘
```

---

## 2.2. native_gpt4o (без BLIP)

### Диаграмма процесса

```
   ┌─────────────┐      ┌─────────────┐       ┌──────────────────────────┐
   │   Клиент    │ ---> │ Изображение │ --->  │LLM (GPT‑4o, vision input)│
   └─────────────┘      └─────────────┘       └──────────────────────────┘
                                                                  │
                                                                  ▼
                                                    ┌─────────────────────────┐
                                                    │   Вопросы или диагноз   │
                                                    └─────────────────────────┘
```

---

## 3. Локальный запуск

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Создайте файл `.env`:

```
OPENAI_API_KEY="sk-..."
```

Запуск:

```bash
uvicorn app.main:app --reload
```

---

## 4. Примеры запросов/ответов

### 4.1 strict_local — первый запрос

```json
{
  "image_base64": "<base64>", # Для перевода фотографии в base64 используйте translate.py
  "pipeline_mode": "strict_local",
  "conversation_history": []
}
```

Ответ:

```json
{
  "stage": "questioning",
  "content": "Thank you for sharing the information about your skin condition. To help narrow down potential diagnoses, I have a few clarifying questions:\n\n1. How long have you been experiencing this skin condition, and has there been any change in its appearance?\n2. Are there any accompanying symptoms, such as itching, pain, or swelling in the affected area?\n3. Have you been exposed to any new products, materials, or allergens recently that could be related to this condition?\n\nYour responses will help guide the analysis further.",
  "final_diagnosis": null,
  "updated_history": [
     {
      "role": "assistant",
      "content": "Extracted features from image:\nVisual Analysis Report:\n1. General View: a person holding their finger in the air with their thumb\n2. Texture Details: a close up photo of skin texture which is a fingernailing\n3. Coloration: the color of the skin rash is visible on the finger"
    },
    {
      "role": "assistant",
      "content": "Thank you for sharing the information about your skin condition. To help narrow down potential diagnoses, I have a few clarifying questions:\n\n1. How long have you been experiencing this skin condition, and has there been any change in its appearance?\n2. Are there any accompanying symptoms, such as itching, pain, or swelling in the affected area?\n3. Have you been exposed to any new products, materials, or allergens recently that could be related to this condition?\n\nYour responses will help guide the analysis further."
    }
  ]
}
```

Ответ на вопросы LLM-ки:
```json
{
  "image_base64": "",
  "pipeline_mode": "strict_local",
  "conversation_history": [
     {
      "role": "assistant",
      "content": "Extracted features from image:\nVisual Analysis Report:\n1. General View: a person holding their finger in the air with their thumb\n2. Texture Details: a close up photo of skin texture which is a fingernailing\n3. Coloration: the color of the skin rash is visible on the finger"
    },
    {
      "role": "assistant",
      "content": "Thank you for sharing the information about your skin condition. To help narrow down potential diagnoses, I have a few clarifying questions:\n\n1. How long have you been experiencing this skin condition, and has there been any change in its appearance?\n2. Are there any accompanying symptoms, such as itching, pain, or swelling in the affected area?\n3. Have you been exposed to any new products, materials, or allergens recently that could be related to this condition?\n\nYour responses will help guide the analysis further."
    },
    {
      "role": "user",
      "content": "Ответы на вопросы"
    }
  ]
}
```

и так продолжаете пока не будет выведен предположительный диагноз чатомгпт
---

## 5. Видео‑демо

Видео демонстрирует:
1. Загрузка изображения.
2. Извлечение признаков BLIP.
3. Диалог с LLM.
4. Финальный диагноз.

Файл: `https://disk.yandex.ru/i/UVIfFFRdk-ds-Q`

---

## 6. Примечания

- В режиме strict_local всегда используется BLIP.
- В режиме native_gpt4o изображение анализируется непосредственно моделью GPT‑4o.
- Вся история диалога отправляется на каждый шаг в `conversation_history`.
