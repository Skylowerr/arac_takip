import csv
import json

from config import OUTPUT_DIR


FIELDS = [
    "time",
    "elapsed_seconds",
    "frame",
    "vehicle_id",
    "vehicle_type",
    "direction",
    "vehicle_confidence",
    "plate",
    "plate_origin",
    "plate_detection_confidence",
    "ocr_confidence",
    "vehicle_image",
    "raw_plate_image",
    "enhanced_plate_image"
]


def save_results(events):
    """Bütün geçiş kayıtlarını CSV, JSON ve TXT olarak dışa aktarır."""

    csv_path = OUTPUT_DIR / "vehicle_events.csv"
    json_path = OUTPUT_DIR / "vehicle_events.json"
    txt_path = OUTPUT_DIR / "vehicle_events.txt"

    # CSV: Excel veya pandas ile açılabilir.
    with open(
        csv_path, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(events)

    # JSON: Yapısal kayıt.
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(events, file, ensure_ascii=False, indent=2)

    # TXT: İnsan tarafından kolay okunabilir rapor.
    with open(txt_path, "w", encoding="utf-8") as file:
        for event in events:
            file.write(
                f"Zaman: {event['time']} | "
                f"Yön: {event['direction']} | "
                f"Araç ID: {event['vehicle_id']} | "
                f"Tür: {event['vehicle_type']} | "
                f"Plaka: {event['plate']} | "
                f"Araç güveni: {event['vehicle_confidence']} | "
                f"Plaka tespit güveni: "
                f"{event['plate_detection_confidence']} | "
                f"OCR güveni: {event['ocr_confidence']}\n"
            )

    print("\nSonuç dosyaları oluşturuldu:")
    print(csv_path)
    print(json_path)
    print(txt_path)
