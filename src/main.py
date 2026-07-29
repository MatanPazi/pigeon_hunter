import cv2
import platform
from pathlib import Path
import numpy as np
from datetime import datetime
import time

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ========================= CONFIG =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

THRESHOLD = 100
MIN_AREA = 250
MAX_AREA = 5000
ALPHA = 0.01
PERSISTENCE_FRAMES = 5
POS_TOL = 20
KERNEL = np.ones((7, 7), np.uint8)

def is_raspberry_pi() -> bool:
    """Return True if running on any Raspberry Pi."""
    machine = platform.machine()
    return machine in ('armv6l', 'armv7l', 'aarch64')


class BirdDetector:
    def __init__(self, persistence_frames: int = 3, position_tolerance: int = 60):
        self.persistence_frames = persistence_frames
        self.position_tolerance = position_tolerance
        self.recent_detections = []  # list of (center_x, center_y, area) from last few frames

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        background: np.ndarray,
        save_prefix: str = "",
        save_debug: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """
        Returns (annotated_output, bird_confirmed)
        bird_confirmed = True only after persistence check passes
        """
        bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(frame_gray, bg_gray)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thresh = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, KERNEL)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame_bgr.copy()
        current_candidates = []   # List of (center, area, contour, rect)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width < 15 or height < 15:
                continue

            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            if aspect_ratio < 0.18 or aspect_ratio > 3.5:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.04:
                    continue

            # Get center
            center = (int(rect[0][0]), int(rect[0][1]))
            current_candidates.append((center, area, contour, rect))

        # ==================== Persistence Check ====================
        confirmed_candidates  = self._get_confirmed_candidates(current_candidates)
        bird_confirmed = len(confirmed_candidates) > 0

        # === Draw only persistent detections ===
        if bird_confirmed and confirmed_candidates:
            for center, area, contour, rect in confirmed_candidates:
                # Draw the box
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.drawContours(output, [box], 0, (0, 255, 0), 2)

                # Text label
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

                label = f"A={int(area)} AR={aspect_ratio:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)[0]
                text_x = int(rect[0][0] - text_size[0] / 2)
                text_y = int(rect[0][1] - (height / 2) + text_size[1] + 8)
                text_y = max(text_y, text_size[1] + 10)

                cv2.putText(
                    output, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2
                )

        if save_debug and bird_confirmed:
            cv2.imwrite(f"{save_prefix}_detection.jpg", output)
            cv2.imwrite(f"{save_prefix}_diff.jpg", diff)
            cv2.imwrite(f"{save_prefix}_thresh.jpg", thresh)
            cv2.imwrite(f"{save_prefix}_closed.jpg", closed)            
            # add other debug saves as needed        
            print(f"Bird detected!")

        # Update history (use all candidates for learning)
        self.recent_detections.append([(c[0], c[1]) for c in current_candidates])
        if len(self.recent_detections) > 10:        # keep reasonable history
            self.recent_detections.pop(0)

        return output, bird_confirmed

    def _get_confirmed_candidates(self, current_candidates):
        """Return only candidates that have been seen in recent frames"""
        if not current_candidates:
            return []

        confirmed = []
        for center, area, contour, rect in current_candidates:
            cx, cy = center
            match_count = 0

            for past_frame in self.recent_detections[-self.persistence_frames:]:
                for past_center, _ in past_frame:
                    px, py = past_center
                    dist = ((cx - px)**2 + (cy - py)**2)**0.5
                    if dist < self.position_tolerance:
                        match_count += 1
                        break

            if match_count >= self.persistence_frames:
                confirmed.append((center, area, contour, rect))

        return confirmed

def pi_run():
    from picamera2 import Picamera2

    INTERVAL_SEC = 1
    DURATION_SEC = 18000

    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (1296, 972)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)

    # Init bird detector class
    detector = BirdDetector(persistence_frames=PERSISTENCE_FRAMES, position_tolerance=POS_TOL)

    # Initial background
    background = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
    bg_model = background.astype(np.float32)

    num_images = DURATION_SEC // INTERVAL_SEC

    for i in range(num_images):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        output, bird_found = detector.process_frame(
            frame_bgr, background,
            save_prefix=str(DATA_DIR / timestamp),
            save_debug=True
        )

        # save original
        cv2.imwrite(str(DATA_DIR / f"{timestamp}_original.jpg"), frame_bgr)
        
        if not bird_found:
            # Background update on if bird not found
            cv2.accumulateWeighted(frame_bgr, bg_model, ALPHA)
            background = bg_model.astype(np.uint8)   # update for next iteration

        if i % 10 == 0:
            print(f"{i}/{num_images}")

        time.sleep(INTERVAL_SEC)

    picam2.stop()
    print("Capture session finished.")


def pc_run():

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

    # Init bird detector class
    detector = BirdDetector(persistence_frames=PERSISTENCE_FRAMES, position_tolerance=POS_TOL)

    # Background model for weighted averaging
    bg_model = background.astype(np.float32)

    for idx, frame_path in enumerate(image_files[1:], 1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"⚠️ Could not load {frame_path.name}")
            continue

        print(f"Processing {frame_path.name} ({idx}/{len(image_files)-1})")

        save_prefix = str(DATA_DIR / f"{frame_path.stem}_proc_{idx:03d}")

        output, bird_found = detector.process_frame(
            frame, background,
            save_prefix=save_prefix,
            save_debug=True
        )

        if not bird_found:
            # === Weighted Background Update ===
            cv2.accumulateWeighted(frame, bg_model, ALPHA)
            background = bg_model.astype(np.uint8)   # update for next iteration

    print("✅ All images processed! Check the `data/` folder.")

# Simple usage
if __name__ == "__main__":
    if is_raspberry_pi():
        print("Running on Raspberry Pi")
        pi_run()
        # Grab image from Pi by PC:
        # DATE=$(date +%Y-%m-%d) && \
        # mkdir -p ~/Matan/Repos/pigeon_hunter/data/$DATE && \
        # scp matan@raspberrypi:~/Repos/pigeon_hunter/data/*.jpg \
        # ~/Matan/Repos/pigeon_hunter/data/$DATE/
    else:
        print("🖥️  Running on PC")
        pc_run()
