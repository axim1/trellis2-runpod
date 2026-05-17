from typing import *
import os
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


PUBLIC_BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
GATED_BIREFNET_MODELS = {"briaai/RMBG-2.0"}


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        model_name = self._resolve_model_name(model_name)
        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                model_name, trust_remote_code=True
            )
        except OSError as exc:
            if self._should_fallback(model_name, exc):
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    PUBLIC_BIREFNET_MODEL, trust_remote_code=True
                )
            else:
                raise
        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        override = os.environ.get("TRELLIS_REMBG_MODEL")
        if override:
            return override
        return model_name

    @staticmethod
    def _should_fallback(model_name: str, exc: OSError) -> bool:
        if model_name not in GATED_BIREFNET_MODELS:
            return False
        if isinstance(exc, (GatedRepoError, RepositoryNotFoundError)):
            return True
        message = str(exc)
        return "gated repo" in message.lower() or "403 client error" in message.lower()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    
