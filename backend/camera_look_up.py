#!/usr/bin/env python3
from picarx import Picarx

try:
    px = Picarx()
    px.set_cam_tilt_angle(20)  # Positive = up
    print("Looking up")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
