import base64
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class LocalVisionService:
    def __init__(self):
        print("🔄 Загрузка локальной Vision-модели (BLIP)...")
        # BLIP Base отлично подходит для captioning
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"✅ Vision-модель готова на {self.device}")

    def _image_from_base64(self, b64_str: str) -> Image.Image:
        # Убираем заголовок data:image/jpeg;base64, если он есть
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data)).convert('RGB')

    def _generate_description(self, raw_image, text_prompt: str = None) -> str:
        """
        Внутренний метод: генерирует описание. 
        Если есть text_prompt, модель продолжает эту фразу, глядя на фото.
        """
        if text_prompt:
            # Conditional generation: "The skin texture is..." -> model completes it
            inputs = self.processor(raw_image, text_prompt, return_tensors="pt").to(self.device)
        else:
            # Unconditional generation: просто опиши что видишь
            inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        out = self.model.generate(**inputs, max_new_tokens=50, min_length=10)
        return self.processor.decode(out[0], skip_special_tokens=True)

    def analyze_image(self, b64_image: str) -> str:
        """
        Извлекает визуальные признаки, задавая несколько вопросов модели.
        """
        raw_image = self._image_from_base64(b64_image)
        
        # 1. Общее описание (Base caption)
        general_desc = self._generate_description(raw_image)
        
        # 2. Уточнение деталей через Conditional Prompting
        # Мы заставляем модель обратить внимание на конкретные аспекты
        texture_desc = self._generate_description(raw_image, "a close up photo of skin texture which is")
        color_desc = self._generate_description(raw_image, "the color of the skin rash is")
        
        # Собираем итоговый "промпт" для LLM
        features_report = (
            f"Visual Analysis Report:\n"
            f"1. General View: {general_desc}\n"
            f"2. Texture Details: {texture_desc}\n"
            f"3. Coloration: {color_desc}"
        )
        
        return features_report

local_vision_service = LocalVisionService()