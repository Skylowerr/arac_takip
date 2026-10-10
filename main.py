import time
import cv2
import torch

from ultralytics import YOLO

from config import (
    INPUT_VIDEO,
    OUTPUT_DIR,
    OUTPUT_VIDEO,
    VEHICLE_CROP_DIR,
    RAW_PLATE_DIR,
    ENHANCED_PLATE_DIR,
    LINE_RATIO,
    VEHICLE_MODEL,
    VEHICLE_CLASSES,
    VEHICLE_CONF,
    VEHICLE_IMGSZ,
    VEHICLE_NAMES,
    VEHICLE_COLORS,
    PLATE_CONF,
    PLATE_IMGSZ,
    PLATE_CHECK_INTERVAL,
    PLATE_SEARCH_BAND,
)

from plate_reader import (
    load_plate_model,
    load_ocr_reader,
    read_plate,
)

from utils import (
    format_video_time,
    draw_dashboard,
)

from results import save_results


# ==================================================
# 1. MODELLERİ VE KLASÖRLERİ HAZIRLA
# ==================================================

def main():

    for folder in (
        OUTPUT_DIR,
        VEHICLE_CROP_DIR,
        RAW_PLATE_DIR,
        ENHANCED_PLATE_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(
            f"Video bulunamadı: {INPUT_VIDEO}"
        )

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("GPU bulunamadı. CPU kullanılacak.")

    print("YOLOv8x araç modeli yükleniyor...")
    vehicle_model = YOLO(VEHICLE_MODEL)

    print("Plaka tespit modeli yükleniyor...")
    plate_model = load_plate_model(device)

    print("EasyOCR yükleniyor...")
    reader = load_ocr_reader()

    # ==================================================
    # 2. VİDEOYU AÇ
    # ==================================================

    cap = cv2.VideoCapture(str(INPUT_VIDEO))

    if not cap.isOpened():
        raise RuntimeError(
            f"Video açılamadı: {INPUT_VIDEO}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        video_fps = 25.0

    # Tam video için herhangi bir süre sınırı uygulanmıyor.
    line_y = int(height * LINE_RATIO)

    out = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (width, height),
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError(
            f"Çıktı videosu oluşturulamadı: {OUTPUT_VIDEO}"
        )

    # ==================================================
    # 3. TAKİP VE SAYIM DEĞİŞKENLERİ
    # ==================================================

    # Önceki karelerdeki araç merkezleri
    previous_positions = {}

    # Aynı ID'nin aynı yönde tekrar sayılmasını önle
    counted_keys = set()

    # Araç başına en kaliteli plaka sonucu
    best_plates = {}
    last_plate_check = {}

    # CSV / JSON / TXT için geçiş kayıtları
    events = []

    # Bir araçla ilişkili olay kayıtlarının indeksleri
    event_indices_by_vehicle = {}

    in_count = 0
    out_count = 0
    frame_no = 0

    start_time = time.time()
    last_log_frame = 0

    print("\nVideo işleniyor: TAM VİDEO")
    print(f"Çözünürlük: {width}x{height}")
    print(f"Kaynak FPS: {video_fps:.2f}")
    print(f"Sayım çizgisi: y={line_y}")

    try:

        # ==================================================
        # 4. VİDEOYU SONUNA KADAR İŞLE
        # ==================================================

        while True:

            ok, frame = cap.read()

            # Kaynak videonun sonuna gelince döngüyü bitir
            if not ok:
                break

            frame_no += 1

            # Orijinal videoya göre zaman
            elapsed_video = (frame_no - 1) / video_fps

            # YOLOv8x + ByteTrack araç tespiti ve takibi
            result = vehicle_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES,
                conf=VEHICLE_CONF,
                imgsz=VEHICLE_IMGSZ,
                device=device,
                verbose=False,
            )[0]

            current_count = 0

            # O karede yeni bulunan plaka kutularını göster
            current_plate_boxes = []

            # ==================================================
            # 5. ALGILANAN ARAÇLARI İŞLE
            # ==================================================

            if (
                result.boxes is not None
                and result.boxes.id is not None
            ):

                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                det_scores = result.boxes.conf.cpu().tolist()

                for box, tid, cls_id, det_conf in zip(
                    boxes, ids, classes, det_scores
                ):

                    x1, y1, x2, y2 = map(int, box)

                    # Kutuyu görüntü sınırları içinde tut
                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Araç merkez noktası
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    vehicle_crop = frame[y1:y2, x1:x2]

                    if vehicle_crop.size == 0:
                        continue

                    current_count += 1

                    vehicle_type = VEHICLE_NAMES.get(
                        cls_id,
                        "Diger arac",
                    )

                    vehicle_color = VEHICLE_COLORS.get(
                        cls_id,
                        (255, 255, 255),
                    )

                    # ==================================================
                    # 6. PLAKA TESPİTİ VE OCR
                    # ==================================================

                    # Çizgiye yaklaşan araçlarda plaka ara.
                    near_line = (
                        abs(cy - line_y)
                        < int(height * PLATE_SEARCH_BAND)
                    )

                    last_check = last_plate_check.get(
                        tid,
                        -999999,
                    )

                    if (
                        near_line
                        and frame_no - last_check
                        >= PLATE_CHECK_INTERVAL
                    ):

                        last_plate_check[tid] = frame_no

                        plate_result = plate_model.predict(
                            vehicle_crop,
                            conf=PLATE_CONF,
                            imgsz=PLATE_IMGSZ,
                            device=device,
                            verbose=False,
                        )[0]

                        if (
                            plate_result.boxes is not None
                            and len(plate_result.boxes) > 0
                        ):

                            plate_boxes = (
                                plate_result.boxes.xyxy.cpu().numpy()
                            )

                            plate_scores = (
                                plate_result.boxes.conf.cpu().numpy()
                            )

                            # Güveni en yüksek plaka tespitini seç
                            best_idx = int(plate_scores.argmax())

                            px1, py1, px2, py2 = map(
                                int,
                                plate_boxes[best_idx],
                            )

                            crop_h, crop_w = vehicle_crop.shape[:2]

                            # Plaka kutusuna küçük bir pay bırak.
                            # Böylece kenardaki karakterler kesilmez.
                            pad_x = max(
                                1,
                                int((px2 - px1) * 0.04),
                            )

                            pad_y = max(
                                1,
                                int((py2 - py1) * 0.08),
                            )

                            px1 = max(0, px1 - pad_x)
                            py1 = max(0, py1 - pad_y)
                            px2 = min(crop_w, px2 + pad_x)
                            py2 = min(crop_h, py2 + pad_y)

                            raw_plate = vehicle_crop[
                                py1:py2,
                                px1:px2,
                            ]

                            plate_det_conf = float(
                                plate_scores[best_idx]
                            )

                            # Araç kırpımından tam video koordinatlarına
                            plate_box_on_frame = (
                                x1 + px1,
                                y1 + py1,
                                x1 + px2,
                                y1 + py2,
                            )

                            if raw_plate.size > 0:

                                # Orijinal ve iyileştirilmiş plaka,
                                # OCR sonucu ile birlikte döner.
                                plate_text, ocr_conf, enhanced = read_plate(
                                    raw_plate,
                                    reader,
                                )

                                if plate_text:
                                    quality = (
                                        0.30 * plate_det_conf
                                        + 0.70 * ocr_conf
                                    )
                                else:
                                    quality = 0.10 * plate_det_conf

                                previous_best = best_plates.get(
                                    tid,
                                    {"quality": -1},
                                )

                                # Daha kaliteli plaka görüntüsü geldiyse
                                # araç için en iyi sonucu güncelle.
                                if quality > previous_best["quality"]:

                                    raw_path = (
                                        RAW_PLATE_DIR
                                        / f"plate_{tid}.jpg"
                                    )

                                    enhanced_path = (
                                        ENHANCED_PLATE_DIR
                                        / f"plate_{tid}.jpg"
                                    )

                                    cv2.imwrite(
                                        str(raw_path),
                                        raw_plate,
                                    )

                                    cv2.imwrite(
                                        str(enhanced_path),
                                        enhanced,
                                    )

                                    best_plates[tid] = {
                                        "text": plate_text,
                                        "plate_confidence": plate_det_conf,
                                        "ocr_confidence": ocr_conf,
                                        "quality": quality,
                                        "raw_path": str(raw_path),
                                        "enhanced_path": str(enhanced_path),
                                    }

                                    current_plate_boxes.append(
                                        (
                                            plate_box_on_frame,
                                            plate_text,
                                        )
                                    )

                                    # Plaka daha sonra okunmuş olsa bile
                                    # önceki geçiş kayıtlarını güncelle.
                                    for event_idx in (
                                        event_indices_by_vehicle.get(
                                            tid, []
                                        )
                                    ):

                                        event = events[event_idx]

                                        event["plate"] = (
                                            plate_text or "OKUNAMADI"
                                        )

                                        event["plate_origin"] = (
                                            "Yabanci plaka varsayimi"
                                            if plate_text
                                            else "OKUNAMADI"
                                        )

                                        event[
                                            "plate_detection_confidence"
                                        ] = round(
                                            plate_det_conf,
                                            3,
                                        )

                                        event["ocr_confidence"] = round(
                                            ocr_conf,
                                            3,
                                        )

                                        event["raw_plate_image"] = str(
                                            raw_path
                                        )

                                        event["enhanced_plate_image"] = str(
                                            enhanced_path
                                        )

                    # ==================================================
                    # 7. SANAL ÇİZGİDEN GEÇİŞİ SAY
                    # ==================================================

                    direction = None

                    if tid in previous_positions:

                        prev_x, prev_y = previous_positions[tid]

                        # Yukarıdan aşağıya IN
                        if prev_y < line_y <= cy:
                            direction = "IN"

                        # Aşağıdan yukarıya OUT
                        elif prev_y > line_y >= cy:
                            direction = "OUT"

                    previous_positions[tid] = (cx, cy)

                    # Her araç ID'sini aynı yönde bir kez say
                    if direction is not None:

                        count_key = (tid, direction)

                        if count_key not in counted_keys:

                            counted_keys.add(count_key)

                            if direction == "IN":
                                in_count += 1
                            else:
                                out_count += 1

                            # Çizgiyi geçerken araç fotoğrafını kaydet
                            vehicle_path = (
                                VEHICLE_CROP_DIR
                                / f"vehicle_{tid}_{direction.lower()}.jpg"
                            )

                            cv2.imwrite(
                                str(vehicle_path),
                                vehicle_crop,
                            )

                            plate_info = best_plates.get(
                                tid,
                                {},
                            )

                            event = {
                                "time": format_video_time(
                                    elapsed_video
                                ),
                                "elapsed_seconds": round(
                                    elapsed_video,
                                    3,
                                ),
                                "frame": frame_no,
                                "vehicle_id": int(tid),
                                "vehicle_type": vehicle_type,
                                "direction": direction,
                                "vehicle_confidence": round(
                                    float(det_conf),
                                    3,
                                ),
                                "plate": (
                                    plate_info.get("text")
                                    or "OKUNAMADI"
                                ),
                                "plate_origin": (
                                    "Yabanci plaka varsayimi"
                                    if plate_info.get("text")
                                    else "OKUNAMADI"
                                ),
                                "plate_detection_confidence": round(
                                    plate_info.get(
                                        "plate_confidence",
                                        0.0,
                                    ),
                                    3,
                                ),
                                "ocr_confidence": round(
                                    plate_info.get(
                                        "ocr_confidence",
                                        0.0,
                                    ),
                                    3,
                                ),
                                "vehicle_image": str(vehicle_path),
                                "raw_plate_image": plate_info.get(
                                    "raw_path",
                                    "",
                                ),
                                "enhanced_plate_image": plate_info.get(
                                    "enhanced_path",
                                    "",
                                ),
                            }

                            events.append(event)

                            event_indices_by_vehicle.setdefault(
                                tid,
                                [],
                            ).append(len(events) - 1)

                    # ==================================================
                    # 8. ARAÇ KUTUSU VE PLAKA METNİ
                    # ==================================================

                    plate_info = best_plates.get(tid, {})
                    plate_text = plate_info.get("text", "")

                    label = f"ID:{tid} | {vehicle_type}"

                    if plate_text:
                        label += f" | Plaka:{plate_text}"
                    else:
                        label += " | Plaka araniyor"

                    label += f" | Det:{float(det_conf):.2f}"

                    if plate_info:
                        label += (
                            f" | OCR:"
                            f"{plate_info['ocr_confidence']:.2f}"
                        )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        vehicle_color,
                        2,
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        vehicle_color,
                        2,
                    )

            # ==================================================
            # 9. PLAKA KUTULARINI ÇİZ
            # ==================================================

            for plate_box, plate_text in current_plate_boxes:

                ax1, ay1, ax2, ay2 = plate_box

                cv2.rectangle(
                    frame,
                    (ax1, ay1),
                    (ax2, ay2),
                    (255, 255, 255),
                    2,
                )

                if plate_text:
                    cv2.putText(
                        frame,
                        plate_text,
                        (ax1, max(18, ay1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

            # ==================================================
            # 10. FPS VE IN/OUT GÖSTERGESİ
            # ==================================================

            elapsed_processing = time.time() - start_time

            processing_fps = (
                frame_no / max(elapsed_processing, 1e-6)
            )

            frame = draw_dashboard(
                frame=frame,
                line_y=line_y,
                in_count=in_count,
                out_count=out_count,
                total_count=in_count + out_count,
                current_count=current_count,
                processing_fps=processing_fps,
            )

            # Tamamlanan kareyi videoya yaz
            out.write(frame)

            # Her yaklaşık 3 saniyede ilerleme bilgisi
            if frame_no % max(
                1,
                int(video_fps * 3),
            ) == 0:

                if total_frames > 0:
                    progress = f"{frame_no}/{total_frames}"
                else:
                    progress = str(frame_no)

                print(
                    f"Kare: {progress} | "
                    f"Video süresi: {frame_no / video_fps:.1f} sn | "
                    f"IN: {in_count} | "
                    f"OUT: {out_count} | "
                    f"FPS: {processing_fps:.1f}"
                )

    finally:
        cap.release()
        out.release()

    # ==================================================
    # 11. CSV, JSON VE TXT SONUÇLARINI KAYDET
    # ==================================================

    save_results(events)

    print("\nİŞLEM TAMAMLANDI")
    print(f"İşlenen kare: {frame_no}")
    print(
        f"İşlenen süre: {frame_no / video_fps:.2f} saniye"
    )
    print(f"IN sayısı: {in_count}")
    print(f"OUT sayısı: {out_count}")
    print(f"Toplam geçiş: {in_count + out_count}")
    print(f"Video: {OUTPUT_VIDEO}")
    print(f"Plaka fotoğrafları: {RAW_PLATE_DIR}")
    print(f"İyileştirilmiş plakalar: {ENHANCED_PLATE_DIR}")


if __name__ == "__main__":
    main()
