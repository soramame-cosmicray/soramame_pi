#!/usr/bin/env python3
"""
SORAMAME Raspberry Pi: minimal acquisition + candidate logging (public version)

This script is a cleaned, publishable subset of the original `pi_realtime_gps.py`.
It keeps only:
  - frame capture (Picamera2)
  - simple candidate extraction (threshold + contours)
  - optional ROI feature extraction
  - optional GPS logging (disabled by default)

It intentionally excludes:
  - any cloud/network upload
  - any device identifiers / credentials
  - any deployment-specific absolute paths

Author: SORAMAME project (public subset)
License: MIT (if you set repo license to MIT)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
from picamera2 import Picamera2

# -----------------------
# Configuration (edit here)
# -----------------------

# Use a repo-local data directory by default (do NOT commit large outputs).
DATA_DIR = Path("./data")  # created automatically
BADPIX_FILE = DATA_DIR / "badpixlist.dat"
LOG_DIR = DATA_DIR / "logs"

# Camera
WIDTH = 4056
HEIGHT = 3040
EXPOSURE_US = 1_000_000
ANALOG_GAIN = 1.0

# Candidate extraction
THRESH_VALUE = 3          # binary threshold (0-255)
MIN_AREA = 2.0            # minimum contour area in pixels^2
ROI_SIZE = 100            # crop size around centroid (pixels)
MIN_RECT_SIDE = 2.0       # filter by minAreaRect width/height

# GPS logging (privacy note: latitude/longitude may be sensitive)
ENABLE_GPS = False

# -----------------------
# Optional GPS support
# -----------------------

def get_gps_data(timeout_s: float = 1.0) -> Tuple[str, str, str]:
    """
    Returns (lat, lon, time_iso). Requires gpsd + python 'gps' module.
    If unavailable or timeout, raises an exception.
    """
    import gps  # type: ignore

    session = gps.gps(mode=gps.WATCH_ENABLE)
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("GPS read timeout")
        report = session.next()
        if isinstance(report, dict) and report.get("class") == "TPV":
            lat = str(getattr(report, "lat", "Unknown"))
            lon = str(getattr(report, "lon", "Unknown"))
            t = str(getattr(report, "time", "Unknown"))
            return lat, lon, t


# -----------------------
# Bad pixel handling
# -----------------------

def load_bad_pixels(path: Path) -> set[Tuple[int, int]]:
    """Load bad pixel coordinates from CSV-like file 'x,y' per line."""
    if not path.exists():
        return set()
    bad = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x_str, y_str = line.split(",")
            bad.add((int(x_str), int(y_str)))
        except Exception:
            # ignore malformed lines
            continue
    return bad


def is_defective_pixel(coordinate: Tuple[int, int], bad_pixels: set[Tuple[int, int]], radius: float = 2.0) -> bool:
    """Return True if coordinate is within 'radius' pixels of any known bad pixel."""
    x, y = coordinate
    r2 = radius * radius
    for bx, by in bad_pixels:
        dx = bx - x
        dy = by - y
        if dx * dx + dy * dy <= r2:
            return True
    return False


# -----------------------
# Candidate representation
# -----------------------

@dataclass
class Candidate:
    ts: str
    x: int
    y: int
    area: float
    rect_w: float
    rect_h: float
    angle: float
    rgb_sum: Optional[str] = None
    lat: str = "Unknown"
    lon: str = "Unknown"


# -----------------------
# Core processing
# -----------------------

def crop_square(img: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:
    half = size // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(img.shape[1], cx + half)
    y1 = min(img.shape[0], cy + half)
    return img[y0:y1, x0:x1]


def process_frame(gray: np.ndarray, ts: str, bad_pixels: set[Tuple[int, int]]) -> List[Candidate]:
    # threshold
    _, thr = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out: List[Candidate] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < MIN_AREA:
            continue

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        x = int(approx.ravel()[0])
        y = int(approx.ravel()[1])

        if is_defective_pixel((x, y), bad_pixels):
            continue

        # ROI and additional contour extraction
        roi2 = crop_square(gray, x, y, ROI_SIZE)
        if roi2.size == 0:
            continue

        _, thr2 = cv2.threshold(roi2, 10, 255, cv2.THRESH_BINARY)
        contours2, _ = cv2.findContours(thr2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # choose the largest local contour for shape features
        if not contours2:
            continue
        cnt2 = max(contours2, key=cv2.contourArea)

        rect = cv2.minAreaRect(cnt2)
        (center_x, center_y), (w, h), angle = rect

        if (w < MIN_RECT_SIDE) and (h < MIN_RECT_SIDE):
            continue

        # Optional feature: RGB sum requires a color ROI; here we keep grayscale sum as proxy
        gray_sum = int(np.sum(roi2))
        cand = Candidate(
            ts=ts,
            x=x,
            y=y,
            area=area,
            rect_w=float(w),
            rect_h=float(h),
            angle=float(angle),
            rgb_sum=str(gray_sum),
        )
        out.append(cand)

    return out


def append_log(cands: List[Candidate], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        for c in cands:
            # CSV-like row
            f.write(
                f"{c.ts},{c.x},{c.y},{c.area:.2f},{c.rect_w:.2f},{c.rect_h:.2f},{c.angle:.2f},{c.rgb_sum},{c.lat},{c.lon}\n"
            )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    bad_pixels = load_bad_pixels(BADPIX_FILE)

    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (WIDTH, HEIGHT), "format": "YUV420"})
    picam2.configure(camera_config)
    picam2.set_controls({"ExposureTime": EXPOSURE_US, "AnalogueGain": ANALOG_GAIN})
    picam2.start()

    try:
        while True:
            now = datetime.now()
            ts = now.strftime("%Y%m%d%H%M%S%f")
            day = now.strftime("%Y%m%d%H")

            yuv = picam2.capture_array()
            gray = yuv[:HEIGHT, :WIDTH]

            cands = process_frame(gray, ts, bad_pixels)

            if cands:
                if ENABLE_GPS:
                    try:
                        lat, lon, t_gps = get_gps_data()
                        for c in cands:
                            c.lat = lat
                            c.lon = lon
                            # (optional) could store t_gps separately
                    except Exception:
                        pass

                log_path = LOG_DIR / f"{day}.log"
                append_log(cands, log_path)

            # throttle a little to reduce CPU load if needed
            time.sleep(0.1)

    finally:
        picam2.close()


if __name__ == "__main__":
    main()
