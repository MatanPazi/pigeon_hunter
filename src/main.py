import cv2
import platform
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ---------- PARAMETERS ----------

THRESHOLD = 100
MIN_AREA = 250
MAX_AREA = 5000
ALPHA = 0.5          # ← Weighted update rate (smaller = slower adaptation)
kernel = np.ones((7,7), np.uint8)

def is_raspberry_pi() -> bool:
    """Return True if running on any Raspberry Pi."""
    machine = platform.machine()
    return machine in ('armv6l', 'armv7l', 'aarch64')

def capture():
    from datetime import datetime
    import time
    from picamera2 import Picamera2

    INTERVAL_SEC = 1
    DURATION_SEC = 18000

    picam2 = Picamera2()

    config = picam2.create_still_configuration(
        main={"size": (1296, 972)}
    )

    picam2.configure(config)
    picam2.start()

    time.sleep(2)

    num_images = DURATION_SEC // INTERVAL_SEC

    background = picam2.capture_array()
    background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)  # picam2 returns RGB. cv2 requires BGR
    bg_model = background.astype(np.float32)

    for i in range(num_images):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)   # Convert to BGR for OpenCV

        # === Processing ===
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(frame_gray, bg_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = frame_bgr.copy()
        contour_cntr = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            # === Rotated Bounding Box ===
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)                    # convert to integer coordinates

            # Get width and height from rotated rect
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

            # Filters
            if width < 15 or height < 15:           # too thin in any direction
                continue
            if aspect_ratio < 0.18 or aspect_ratio > 3.5:   # stricter for pigeons
                continue

            # Thin object filter (helps with flapping strips)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.04:              # very low = thin line-like
                    continue

            # === Valid detection ===
            bird_found = True

            # Draw rotated rectangle
            cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

            # Text label
            label = f"A={int(area)} AR={aspect_ratio:.2f}"

            # Calculate text position (inside the box, top-centered)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
            text_w, text_h = text_size

            # Center horizontally
            text_x = int(rect[0][0] - text_w / 2)   # use center of rotated rect
            text_y = int(rect[0][1] - (height / 2) + text_h + 8)  # near top of box

            # Safety: keep text inside frame
            text_y = max(text_y, text_h + 10)

            cv2.putText(
                output,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2
            )

            # Save both original and annotated
            cv2.imwrite(str(DATA_DIR / f"{timestamp}_detected_{contour_cntr:03d}.jpg"), output)
            print(f"{i}_detected!")
            contour_cntr += 1
            
        cv2.imwrite(str(DATA_DIR / f"{timestamp}_original.jpg"), frame_bgr)

        # === Weighted Background Update ===
        cv2.accumulateWeighted(frame, bg_model, ALPHA)
        background = bg_model.astype(np.uint8)   # update for next iteration        

        if i % 10 == 0:
            print(f"{i}/{num_images}")

        time.sleep(INTERVAL_SEC)

    picam2.stop()


def processing():

    # ---------- LOAD ALL IMAGES ----------
    image_files = sorted([f for f in DATA_DIR.glob("*.jpg") if f.is_file()])
    if len(image_files) < 2:
        print("❌ Need at least 2 .jpg images in the data folder.")
        return

    # First image = initial background
    background = cv2.imread(str(image_files[0]))
    if background is None:
        print("❌ Could not load background image.")
        return

    print(f"✅ Using {image_files[0].name} as initial background")
    # cv2.imwrite(str(DATA_DIR / "01_background.png"), background)

    # Background model for weighted averaging
    bg_model = background.astype(np.float32)

    for idx, frame_path in enumerate(image_files[1:], 1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Could not load {frame_path.name}")
            continue

        print(f"Processing {frame_path.name} ({idx}/{len(image_files)-1})")

        # cv2.imwrite(str(DATA_DIR / f"02_input_{idx:03d}.png"), frame)

        # Grayscale
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite(str(DATA_DIR / f"03_bg_gray_{idx:03d}.png"), bg_gray)
        # cv2.imwrite(str(DATA_DIR / f"04_frame_gray_{idx:03d}.png"), frame_gray)

        # Difference + Threshold
        diff = cv2.absdiff(frame_gray, bg_gray)

        # Add light blurring before thresholding (removes small noise)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        # cv2.imwrite(str(DATA_DIR / f"05_difference_{idx:03d}.png"), diff)

        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(str(DATA_DIR / f"06_threshold_{idx:03d}.png"), thresh)

        # Morphology
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        # cv2.imwrite(str(DATA_DIR / f"08_close_{idx:03d}.png"), closed)

        # Contours & Detection
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = frame.copy()
        bird_found = False

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            # === Rotated Bounding Box ===
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)                    # convert to integer coordinates

            # Get width and height from rotated rect
            width = rect[1][0]
            height = rect[1][1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

            # Filters
            if width < 15 or height < 15:           # too thin in any direction
                continue
            if aspect_ratio < 0.18 or aspect_ratio > 3.5:   # stricter for pigeons
                continue

            # Thin object filter (helps with flapping strips)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.04:              # very low = thin line-like
                    continue

            # === Valid detection ===
            bird_found = True

            # Draw rotated rectangle
            cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

            # Text label
            label = f"A={int(area)} AR={aspect_ratio:.2f}"

            # Calculate text position (inside the box, top-centered)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
            text_w, text_h = text_size

            # Center horizontally
            text_x = int(rect[0][0] - text_w / 2)   # use center of rotated rect
            text_y = int(rect[0][1] - (height / 2) + text_h + 8)  # near top of box

            # Safety: keep text inside frame
            text_y = max(text_y, text_h + 10)

            cv2.putText(
                output,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 0),
                2
            )

            print(f"✅ Detected contour in {image_files[0].name}")
            cv2.imwrite(str(DATA_DIR / f"{frame_path.name}_{idx:03d}_difference.png"), diff)
            cv2.imwrite(str(DATA_DIR / f"{frame_path.name}_{idx:03d}_threshold.png"), thresh)
            cv2.imwrite(str(DATA_DIR / f"{frame_path.name}_{idx:03d}_close.png"), closed)
            cv2.imwrite(str(DATA_DIR / f"{frame_path.name}_{idx:03d}_detection.png"), output)

        # === Weighted Background Update ===
        if not bird_found:
            cv2.accumulateWeighted(frame, bg_model, ALPHA)
            background = bg_model.astype(np.uint8)   # update for next iteration

    print("✅ All images processed! Check the `data/` folder.")

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
