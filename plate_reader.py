import cv2
import numpy as np
import torch
import easyocr

from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from config import (
    PLATE_MODEL_REPO,
    PLATE_MODEL_FILE
)


def load_plate_model(device):
    """Hazır plaka tespit modelini indirir ve yükler."""
    weights = hf_hub_download(
        repo_id=PLATE_MODEL_REPO,
        filename=PLATE_MODEL_FILE
    )
    return YOLO(weights)


def load_ocr_reader():
    """Yabancı Latin harfli plakalar için OCR hazırlar."""
    return easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )


def clean_plate(text):
    """OCR metnindeki boşlukları ve özel karakterleri temizler."""
    return "".join(
        char for char in text.upper()
        if char.isascii() and char.isalnum()
    )


def enhance_plate_crop(crop):
    """
    Plaka görüntüsünü büyütür, yerel kontrastı artırır,
    hafif gürültü azaltma ve keskinleştirme uygular.

    Not: Bu işlem var olmayan ayrıntıları geri getirmez.
    """
    if crop is None or crop.size == 0:
        return None, []

    h, w = crop.shape[:2]

    # Küçük plakaları OCR'ın daha rahat okuyacağı boyuta getir.
    scale = max(2, min(5, int(np.ceil(400 / max(1, w)))))

    enlarged = cv2.resize(
        crop,
        (w * scale, h * scale),
        interpolation=cv2.INTER_CUBIC
    )

    # LAB renk uzayında parlaklık kontrastını iyileştir.
    lab = cv2.cvtColor(enlarged, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.cvtColor(
        cv2.merge((l_channel, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR
    )

    # Hafif keskinleştirme.
    blurred_color = cv2.GaussianBlur(
        enhanced, (0, 0), 1.0
    )
    enhanced = cv2.addWeighted(
        enhanced, 1.5, blurred_color, -0.5, 0
    )

    # OCR için alternatif gri görüntüler.
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(
        gray, d=5, sigmaColor=35, sigmaSpace=35
    )

    blurred_gray = cv2.GaussianBlur(
        gray, (0, 0), 1.0
    )
    sharpened = cv2.addWeighted(
        gray, 1.5, blurred_gray, -0.5, 0
    )

    # Alternatif: Otsu eşikleme.
    _, otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return enhanced, [sharpened, otsu]


def read_plate(crop, reader):
    """
    Orijinal plaka kırpımından iyileştirilmiş görüntü üretir.
    Birden fazla OCR ön işlemesini dener ve en iyi skorlu
    metni döndürür.
    """
    enhanced, variants = enhance_plate_crop(crop)

    if enhanced is None:
        return "", 0.0, None

    best_text = ""
    best_confidence = 0.0
    best_quality = -1.0

    allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    for variant in variants:
        results = reader.readtext(
            variant,
            detail=1,
            paragraph=False,
            allowlist=allowlist
        )

        if not results:
            continue

        # Metin kutularını soldan sağa sırala.
        results.sort(
            key=lambda item: min(point[0] for point in item[0])
        )

        text = "".join(
            clean_plate(item[1])
            for item in results
        )

        if not text:
            continue

        confidence = sum(
            float(item[2]) for item in results
        ) / len(results)

        # Uzunluğu yalnızca eşitliğe yakın sonuçlarda
        # küçük bir bağlayıcı ölçüt olarak kullan.
        quality = confidence + min(len(text), 8) * 0.003

        if quality > best_quality:
            best_quality = quality
            best_text = text
            best_confidence = confidence

    return best_text, best_confidence, enhanced
