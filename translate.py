import base64

# Открываем изображение в бинарном режиме
with open("image.png", "rb") as f:
    image_bytes = f.read()

# Конвертируем в Base64
image_base64 = base64.b64encode(image_bytes).decode("utf-8")
with open('photo_base64.txt', "w") as f:
    f.writelines(image_base64)
# # Формируем JSON
# payload = {
#     "image_base64": image_base64,
#     "pipeline_mode": "native_gpt4o",
#     "conversation_history": [
#         {
#             "role": "user",
#             "content": "Привет! Проанализируй это изображение."
#         }
#     ]
# }
