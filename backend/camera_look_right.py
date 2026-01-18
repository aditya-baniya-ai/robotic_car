#!/usr/bin/env python3
from picarx import Picarx

try:
    px = Picarx()
    px.set_cam_pan_angle(45)  # Positive = RIGHT
    print("Looking right")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
