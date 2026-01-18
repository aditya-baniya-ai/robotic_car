#!/usr/bin/env python3
import sys
from picarx import Picarx

if len(sys.argv) != 2:
    print("Usage: servo_tilt.py <angle>")
    sys.exit(1)

angle = int(sys.argv[1])
angle = max(0, min(180, angle))

try:
    px = Picarx()
    px.set_cam_tilt_angle(angle - 90)  # Convert 0-180 to -90 to +90
    print(f"Tilt: {angle}°")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
