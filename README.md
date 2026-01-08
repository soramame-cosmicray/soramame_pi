# SORAMAME Raspberry Pi acquisition (public subset)

This repository provides a **public, minimal subset** of the SORAMAME Raspberry Pi pipeline.
It focuses on **frame acquisition** and **event-candidate extraction** on a Raspberry Pi camera,
and logs detected candidates to local files.

This repo intentionally excludes any application-specific components such as cloud upload,
device identifiers, credentials, and deployment-specific settings.

## What this script does

`pi_realtime_gps_public.py` performs:

1. Captures frames using **Picamera2**
2. Converts/uses a grayscale image plane
3. Applies a binary threshold to find bright clusters
4. Extracts candidates via contour detection and basic shape features
5. Logs candidates as CSV-like rows to local files
6. (Optional) Logs GPS latitude/longitude when enabled (disabled by default)

## Files

- `pi_realtime_gps_public.py` : main script (public subset)
- `data/` : runtime output directory (created automatically; **not meant to be committed**)

## Dependencies

- Python 3.10+ (recommended)
- Raspberry Pi OS with **Picamera2**
- `numpy`
- `opencv-python` (or system `python3-opencv`)

Install (typical):

```bash
python3 -m pip install numpy opencv-python


Picamera2 is usually installed via Raspberry Pi OS packages.
Quick start
Connect and enable the Raspberry Pi camera module.
Run:
python3 pi_realtime_gps_public.py
Output logs will be written under:
./data/logs/
Output format
Each detected candidate is appended as one line (CSV-like):
timestamp,x,y,area,rect_w,rect_h,angle,roi_sum,lat,lon
timestamp: local timestamp in YYYYmmddHHMMSSffffff format
x, y: candidate position [pixels]
area: contour area [pixel^2]
rect_w, rect_h, angle: minAreaRect shape features of the largest local contour
roi_sum: sum of grayscale intensities in the ROI (proxy feature)
lat, lon: GPS position (or Unknown if GPS disabled/unavailable)
Configuration
Edit constants near the top of pi_realtime_gps_public.py, e.g.:
WIDTH, HEIGHT
EXPOSURE_US, ANALOG_GAIN
THRESH_VALUE, MIN_AREA, ROI_SIZE
ENABLE_GPS (default: False)
Bad pixel list
If ./data/badpixlist.dat exists, the script will use it to suppress known bad pixels.
Format: one x,y pair per line, e.g.
123,456
1000,42
GPS / privacy note
GPS logging is disabled by default (ENABLE_GPS=False).
If you enable it, be aware that latitude/longitude may be sensitive and should be handled
according to your privacy and data-sharing policies.
Limitations
Candidate detection depends strongly on sensor noise, shielding, exposure, and threshold settings.
This script logs candidate events and does not claim particle identification.
For quantitative comparisons, additional calibration and systematic studies are required.
License
MIT License (see LICENSE).

