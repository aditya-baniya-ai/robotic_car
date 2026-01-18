#!/usr/bin/env python3
from picarx import Picarx

try:
    px = Picarx()
    px.set_cam_pan_angle(0)
    px.set_cam_tilt_angle(0)
    print("Camera centered")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
