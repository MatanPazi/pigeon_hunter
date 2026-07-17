import cv2
import platform
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

def is_raspberry_pi() -> bool:
    """Return True if running on any Raspberry Pi."""
    machine = platform.machine()
    return machine in ('armv6l', 'armv7l', 'aarch64')

def capture():
    from datetime import datetime
    import time
    from picamera2 import Picamera2

    INTERVAL_SEC = 1
    DURATION_SEC = 3600

    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (1296, 972)}
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    num_images = DURATION_SEC // INTERVAL_SEC

    for i in range(num_images):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = DATA_DIR / f"{timestamp}.jpg"

        picam2.capture_file(str(filename))

        # if i % 10 == 0:
        print(f"{i}/{num_images}")

        time.sleep(INTERVAL_SEC)

    picam2.stop()


def processing():
    import numpy as np

    # ---------- PARAMETERS ----------

    THRESHOLD = 10
    MIN_AREA = 500
    MAX_AREA = 20000

    kernel = np.ones((7,7), np.uint8)

    # ---------- LOAD IMAGES ----------

    background = cv2.imread("/home/matan/Matan/Repos/pigeon_hunter/data/balcony_smartphone_img.png")
    frame = cv2.imread("/home/matan/Matan/Repos/pigeon_hunter/data/balcony_with_pigeon_on_railing.png")

    # Save originals
    cv2.imwrite(str(DATA_DIR / "01_background.png"), background)
    cv2.imwrite(str(DATA_DIR / "02_input.png"), frame)

    # ---------- GRAYSCALE ----------

    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(str(DATA_DIR / "03_background_gray.png"), bg_gray)
    cv2.imwrite(str(DATA_DIR / "04_frame_gray.png"), frame_gray)

    # ---------- DIFFERENCE ----------

    diff = cv2.absdiff(frame_gray, bg_gray)

    cv2.imwrite(str(DATA_DIR / "05_difference.png"), diff)

    # ---------- THRESHOLD ----------

    _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    cv2.imwrite(str(DATA_DIR / "06_threshold.png"), thresh)

    # # ---------- OPEN (remove isolated pixels) ----------

    # opened = cv2.morphologyEx(
    #     thresh,
    #     cv2.MORPH_OPEN,
    #     kernel
    # )

    # cv2.imwrite(str(DATA_DIR / "07_open.png"), opened)

    # ---------- CLOSE (fill holes) ----------

    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
    )

    cv2.imwrite(str(DATA_DIR / "08_close.png"), closed)

    # ---------- FIND CONTOURS ----------

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = frame.copy()

    bird_found = False

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / h

        if 0.2 < aspect_ratio < 2:

            bird_found = True
            cv2.rectangle(
                output,
                (x, y),
                (x+w, y+h),
                (0,255,0),
                2
            )

            cv2.putText(
                output,
                f"A={int(area)}",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2
            )

    cv2.imwrite(str(DATA_DIR / "09_detection.png"), output)

# Simple usage
if __name__ == "__main__":
    if is_raspberry_pi():
        print("Running on Raspberry Pi")
        capture()
        # Grab image from Pi by PC:
        # scp matan@raspberrypi:~/Repos/pigeon_hunter/data/*.jpg ~/Matan/Repos/pigeon_hunter/data
    else:
        print("🖥️  Running on Linux PC (x86_64 or other)")
        processing()