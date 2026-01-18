#!/usr/bin/env python3
from picarx import Picarx

try:
    px = Picarx()
    px.set_cam_pan_angle(-45)  # Negative = LEFT
    print("Looking left")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
