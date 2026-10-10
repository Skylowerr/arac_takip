from pathlib import Path

# Video yolları
INPUT_VIDEO = Path("input_videos/arac_giris.mp4")
OUTPUT_DIR = Path("outputs")
OUTPUT_VIDEO = OUTPUT_DIR / "output_results_full.mp4"
# Sonuçları ayrı klasörlerde tut
VEHICLE_CROP_DIR = OUTPUT_DIR / "vehicle_crops"
RAW_PLATE_DIR = OUTPUT_DIR / "plate_crops" / "original"
ENHANCED_PLATE_DIR = OUTPUT_DIR / "plate_crops" / "enhanced"

# Video ayarları
MAX_SECONDS = None
LINE_RATIO = 0.70  # Görüntünün üstünden %70 aşağıda

# Araç modeli
VEHICLE_MODEL = "yolov8x.pt"
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_CONF = 0.20
VEHICLE_IMGSZ = 960

# Plaka modeli
PLATE_MODEL_REPO = "Koushim/yolov8-license-plate-detection"
PLATE_MODEL_FILE = "best.pt"
PLATE_CONF = 0.15
PLATE_IMGSZ = 640

# Plaka arama sıklığı
PLATE_CHECK_INTERVAL = 3
PLATE_SEARCH_BAND = 0.35

# Araç sınıfları (COCO)
VEHICLE_NAMES = {
    2: "Otomobil",
    3: "Motosiklet",
    5: "Otobüs",
    7: "Kamyon"
}



# OpenCV renkleri: BGR
VEHICLE_COLORS = {
    2: (255, 0, 0),       # Otomobil: mavi
    3: (0, 255, 255),     # Motosiklet: sarı
    5: (255, 0, 255),     # Otobüs: mor
    7: (0, 165, 255)      # Kamyon: turuncu
}

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
