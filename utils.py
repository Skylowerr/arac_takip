import cv2

from config import (
    RED,
    GREEN,
    YELLOW,
    WHITE
)


def format_video_time(seconds):
    """Saniyeyi okunabilir zaman metnine dönüştürür."""
    minutes = int(seconds // 60)
    seconds_left = seconds - minutes * 60
    hours = int(minutes // 60)
    minutes %= 60

    return (
        f"{hours:02d}:{minutes:02d}:"
        f"{seconds_left:05.2f}"
    )


def draw_counter_tag(frame, x, y, width, height, text, color):
    """Koyu arka planlı sayaç etiketi çizer."""
    h, w = frame.shape[:2]

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    x2 = min(w - 1, x + width)
    y2 = min(h - 1, y + height)

    cv2.rectangle(
        frame,
        (x, y),
        (x2, y2),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 10, min(y2 - 5, y + height - 9)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA
    )


def draw_dashboard(
    frame,
    line_y,
    in_count,
    out_count,
    total_count,
    current_count,
    processing_fps
):
    """Sanal çizgiyi, IN/OUT sayaçlarını ve FPS'yi ekrana çizer."""
    h, w = frame.shape[:2]

    # Tam genişlikte sayım çizgisi.
    cv2.line(
        frame,
        (0, line_y),
        (w - 1, line_y),
        RED,
        3
    )

    # IN / OUT sayaçlarını çizginin hemen üzerine yerleştir.
    tag_height = 36
    tag_width = min(190, max(125, w // 4))
    tag_y = max(5, line_y - tag_height - 2)

    draw_counter_tag(
        frame,
        15,
        tag_y,
        tag_width,
        tag_height,
        f"IN v  {in_count}",
        GREEN
    )

    draw_counter_tag(
        frame,
        w - tag_width - 15,
        tag_y,
        tag_width,
        tag_height,
        f"OUT ^  {out_count}",
        YELLOW
    )

    # Genel istatistikler.
    cv2.putText(
        frame,
        f"Toplam gecis: {total_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Anlik arac: {current_count}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"Isleme FPS: {processing_fps:.1f}",
        (max(20, w - 230), 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        YELLOW,
        2,
        cv2.LINE_AA
    )

    return frame
