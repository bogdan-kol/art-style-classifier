import base64
import io
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

app = FastAPI(title="Art Style Classifier API")

# Lazy-loaded model (populated on startup)
MODEL = None

# Simple preprocess pipeline (matches training image size)
IMAGE_SIZE = (224, 224)
preprocess = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def read_imagefile(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


@app.on_event("startup")  # type: ignore[misc]
def load_model() -> None:
    """Load model from checkpoint if provided via `MODEL_CKPT` env var.

    If no checkpoint is provided the endpoint will still run but return
    predictions from a freshly initialized model.
    """
    global MODEL
    ckpt = os.environ.get("MODEL_CKPT", "")
    arch = os.environ.get("MODEL_ARCH", "efficientnet_b0")

    try:
        if ckpt and os.path.exists(ckpt):
            # Import here to avoid pulling heavy deps on module import

            if arch.startswith("efficient"):
                from art_style_classifier.models.advanced import AdvancedModel

                ModelClass: Any = AdvancedModel
            else:
                from art_style_classifier.models.baseline import BaselineModel

                ModelClass = BaselineModel

            MODEL = ModelClass.load_from_checkpoint(ckpt, map_location="cpu")
        else:
            # Fallback: create an untrained model instance
            if arch.startswith("efficient"):
                from art_style_classifier.models.advanced import AdvancedModel

                ModelClass = AdvancedModel
            else:
                from art_style_classifier.models.baseline import BaselineModel

                ModelClass = BaselineModel

            MODEL = ModelClass()
        MODEL.eval()
        MODEL.to("cpu")
    except Exception as err:
        raise HTTPException(status_code=500, detail="Failed to load model") from err


@app.get("/health")  # type: ignore[misc]
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info")  # type: ignore[misc]
def info() -> dict[str, Any]:
    return {
        "loaded": MODEL is not None,
        "architecture": getattr(MODEL, "hparams", {}).get("architecture", None)
        if MODEL
        else None,
        "num_classes": getattr(MODEL, "num_classes", None),
    }


class Prediction(BaseModel):
    class_index: int
    probability: float


@app.post("/predict")  # type: ignore[misc]
async def predict(
    file: UploadFile = File(None),
    image_base64: Optional[str] = None,
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    """Predict top-k classes for an uploaded image.

    Accepts either a multipart file upload (`file`) or a base64 string
    in `image_base64`. Returns top-k class indices and probabilities.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file is None and not image_base64:
        raise HTTPException(status_code=400, detail="No image provided")

    try:
        if file:
            contents = await file.read()
            img = read_imagefile(contents)
        else:
            if image_base64 is None:
                raise ValueError("No image data provided")
            decoded = base64.b64decode(image_base64)
            img = read_imagefile(decoded)
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid image data") from err

    tensor = preprocess(img).unsqueeze(0).to("cpu")
    with torch.no_grad():
        logits = MODEL(tensor)
        probs = F.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k=min(top_k, probs.numel()))
        values, indices = topk.values, topk.indices
        preds = [
            Prediction(class_index=int(idx.item()), probability=float(val.item()))
            for val, idx in zip(values, indices)
        ]

    return {"predictions": [p.dict() for p in preds]}


__all__ = ["app"]
