from picarx import Picarx
import time, sys

speed = int(sys.argv[1]) if len(sys.argv) > 1 else 30
duration = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

speed = max(0, min(100, speed))
duration = max(0.1, min(10.0, duration))

px = Picarx()
px.set_dir_servo_angle(30)    # right
px.forward(speed)
time.sleep(duration)
px.forward(0)
px.set_dir_servo_angle(0)

print(f"OK right speed={speed} duration={duration}")
