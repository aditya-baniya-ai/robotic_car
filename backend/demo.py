import os
import asyncio
import logging
import json
import sys
import numpy as np
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext,
)

from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

load_dotenv(".env.local")

SYSTEM_PYTHON = "/usr/bin/python3"
FORWARD_SCRIPT = "/home/bobcat/picar_forward.py"
BACKWARD_SCRIPT = "/home/bobcat/picar_backward.py"
LEFT_SCRIPT = "/home/bobcat/picar_left.py"
RIGHT_SCRIPT = "/home/bobcat/picar_right.py"
STOP_SCRIPT = "/home/bobcat/picar_stop.py"
SERVO_PAN_SCRIPT = "/home/bobcat/servo_pan.py"
SERVO_TILT_SCRIPT = "/home/bobcat/servo_tilt.py"
CAMERA_CENTER_SCRIPT = "/home/bobcat/camera_center.py"
CAMERA_LOOK_LEFT_SCRIPT = "/home/bobcat/camera_look_left.py"
CAMERA_LOOK_RIGHT_SCRIPT = "/home/bobcat/camera_look_right.py"
CAMERA_LOOK_UP_SCRIPT = "/home/bobcat/camera_look_up.py"
CAMERA_LOOK_DOWN_SCRIPT = "/home/bobcat/camera_look_down.py"

# ArUco Marker to Object Mapping
MARKER_OBJECTS = {
    0: "monster_drink",
    1: "water_bottle",
    2: "cube",
    3: "laptop"
}


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You control a SunFounder PiCar-X with camera vision and ArUco marker detection.
You can move forward, backward, turn left, turn right, and stop.
You can control the camera servo to pan left/right and tilt up/down.
You can see through the camera and describe what you observe.
You can detect ArUco markers and navigate autonomously to them.
You track these objects with markers:
- Marker 0: monster_drink
- Marker 1: water_bottle
- Marker 2: cube
- Marker 3: laptop
You can open/close the camera and adjust zoom level.
Be concise and confirm actions.
When navigating to markers, use visual feedback to guide movements.
""".strip(),
        )
        self.camera_task: asyncio.Task | None = None
        self.current_room = None
        self.video_source: rtc.VideoSource | None = None
        self.video_track: rtc.LocalVideoTrack | None = None
        self.camera_zoom: float = 1.0
        self.latest_frame: np.ndarray | None = None
        self.latest_frame_with_boxes: np.ndarray | None = None
        self.vision_enabled: bool = True
        self.navigation_active: bool = False
        self.show_bounding_boxes: bool = True
        self.marker_objects: dict = MARKER_OBJECTS.copy()

    async def _run_script(self, script_path: str, *args: str) -> str:
        if not script_path or not os.path.exists(script_path):
            logger.error(f"Missing: {script_path}")
            return f"Missing: {script_path}"
        
        clean_env = os.environ.copy()
        clean_env.pop("VIRTUAL_ENV", None)
        clean_env.pop("PYTHONPATH", None)
        clean_env["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        
        proc = await asyncio.create_subprocess_exec(
            SYSTEM_PYTHON, script_path, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clean_env,
        )
        
        out, err = await proc.communicate()
        if proc.returncode == 0:
            return out.decode().strip() or "OK"
        
        err_msg = err.decode().strip() or "Error"
        logger.error(f"{script_path} failed: {err_msg}")
        return f"Error: {err_msg}"

    @function_tool
    async def move_forward(self, context: RunContext, speed: int = 30, duration: float = 1.0):
        """Move the robot forward."""
        speed = max(0, min(100, int(speed)))
        duration = max(0.1, min(10.0, float(duration)))
        return await self._run_script(FORWARD_SCRIPT, str(speed), str(duration))

    @function_tool
    async def move_backward(self, context: RunContext, speed: int = 30, duration: float = 1.0):
        """Move the robot backward."""
        speed = max(0, min(100, int(speed)))
        duration = max(0.1, min(10.0, float(duration)))
        return await self._run_script(BACKWARD_SCRIPT, str(speed), str(duration))

    @function_tool
    async def turn_left(self, context: RunContext, speed: int = 30, duration: float = 0.5):
        """Turn the robot left."""
        speed = max(0, min(100, int(speed)))
        duration = max(0.1, min(10.0, float(duration)))
        return await self._run_script(LEFT_SCRIPT, str(speed), str(duration))

    @function_tool
    async def turn_right(self, context: RunContext, speed: int = 30, duration: float = 0.5):
        """Turn the robot right."""
        speed = max(0, min(100, int(speed)))
        duration = max(0.1, min(10.0, float(duration)))
        return await self._run_script(RIGHT_SCRIPT, str(speed), str(duration))

    @function_tool
    async def stop_robot(self, context: RunContext):
        """Stop the robot."""
        return await self._run_script(STOP_SCRIPT)

    @function_tool
    async def camera_pan(self, context: RunContext, angle: int):
        """Pan camera to specific angle. 0=left, 90=center, 180=right"""
        angle = max(0, min(180, int(angle)))
        return await self._run_script(SERVO_PAN_SCRIPT, str(angle))

    @function_tool
    async def camera_tilt(self, context: RunContext, angle: int):
        """Tilt camera to specific angle. 0=down, 90=center, 180=up"""
        angle = max(0, min(180, int(angle)))
        return await self._run_script(SERVO_TILT_SCRIPT, str(angle))

    @function_tool
    async def camera_center(self, context: RunContext):
        """Center the camera."""
        return await self._run_script(CAMERA_CENTER_SCRIPT)

    @function_tool
    async def camera_look_left(self, context: RunContext):
        """Turn camera to look left."""
        return await self._run_script(CAMERA_LOOK_LEFT_SCRIPT)

    @function_tool
    async def camera_look_right(self, context: RunContext):
        """Turn camera to look right."""
        return await self._run_script(CAMERA_LOOK_RIGHT_SCRIPT)

    @function_tool
    async def camera_look_up(self, context: RunContext):
        """Tilt camera to look up."""
        return await self._run_script(CAMERA_LOOK_UP_SCRIPT)

    @function_tool
    async def camera_look_down(self, context: RunContext):
        """Tilt camera to look down."""
        return await self._run_script(CAMERA_LOOK_DOWN_SCRIPT)

    def _get_aruco_detector(self):
        """Get ArUco detector compatible with OpenCV 4.6+"""
        import cv2
        try:
            # OpenCV 4.7+
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            return detector, aruco_dict
        except AttributeError:
            # OpenCV 4.0-4.6 (fallback)
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()
            return (aruco_dict, aruco_params), aruco_dict

    def _detect_markers(self, gray, detector_obj):
        """Detect markers with version compatibility"""
        import cv2
        try:
            # OpenCV 4.7+ (detector_obj is ArucoDetector)
            return detector_obj.detectMarkers(gray)
        except AttributeError:
            # OpenCV 4.0-4.6 (detector_obj is tuple of dict and params)
            aruco_dict, aruco_params = detector_obj
            return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    def _draw_aruco_bounding_boxes(self, frame_rgb: np.ndarray, corners, ids) -> np.ndarray:
        """Draw bounding boxes and labels on detected ArUco markers."""
        import cv2
        
        # Create a copy to draw on
        display_frame = frame_rgb.copy()
        
        if ids is None or len(ids) == 0:
            return display_frame
        
        for i, marker_id in enumerate(ids.flatten()):
            # Get corners for this marker
            marker_corners = corners[i][0]
            
            # Convert to integer coordinates
            top_left = tuple(marker_corners[0].astype(int))
            top_right = tuple(marker_corners[1].astype(int))
            bottom_right = tuple(marker_corners[2].astype(int))
            bottom_left = tuple(marker_corners[3].astype(int))
            
            # Draw bounding box lines (green color)
            cv2.line(display_frame, top_left, top_right, (0, 255, 0), 3)
            cv2.line(display_frame, top_right, bottom_right, (0, 255, 0), 3)
            cv2.line(display_frame, bottom_right, bottom_left, (0, 255, 0), 3)
            cv2.line(display_frame, bottom_left, top_left, (0, 255, 0), 3)
            
            # Calculate center of marker
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))
            
            # Draw center point (red circle)
            cv2.circle(display_frame, (center_x, center_y), 6, (255, 0, 0), -1)
            
            # Get object name
            object_name = self.marker_objects.get(int(marker_id), f"ID:{marker_id}")
            
            # Calculate distance estimate
            marker_width_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
            estimated_distance_cm = (15.0 * 500) / marker_width_pixels
            
            # Prepare label text
            label = f"{object_name} [{int(estimated_distance_cm)}cm]"
            
            # Draw label background (filled rectangle)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_width, label_height = label_size
            
            # Position label above the marker
            label_x = center_x - label_width // 2
            label_y = top_left[1] - 10
            
            # Ensure label stays within frame bounds
            label_x = max(5, min(label_x, display_frame.shape[1] - label_width - 5))
            label_y = max(label_height + 5, label_y)
            
            # Draw semi-transparent background for label
            overlay = display_frame.copy()
            cv2.rectangle(overlay, 
                         (label_x - 5, label_y - label_height - 5),
                         (label_x + label_width + 5, label_y + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            # Draw label text (white color)
            cv2.putText(display_frame, label, 
                       (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display_frame

    @function_tool
    async def describe_view(self, context: RunContext):
        """Describe what the camera currently sees."""
        if self.latest_frame is None:
            return "Camera not active or no frame available"
        
        try:
            from PIL import Image
            import io
            import base64
            import openai as openai_lib
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "Vision system not configured: API key missing"
            
            # Frame is already in RGB format from camera stream
            img = Image.fromarray(self.latest_frame, 'RGB')
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Call OpenAI Vision API
            client = openai_lib.OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what you see in this image from a robot's camera perspective. Be brief and specific about objects, colors, and spatial relationships."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=150
            )
            
            description = response.choices[0].message.content
            logger.info(f"ðŸ‘ï¸ Vision: {description}")
            return description
            
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return f"Vision analysis failed: {str(e)}"

    @function_tool
    async def detect_aruco_markers(self, context: RunContext):
        """Detect all ArUco markers visible in current camera frame with estimated distances and positions."""
        if self.latest_frame is None:
            return "Camera not active or no frame available"
        
        try:
            import cv2
            
            # Get detector
            detector_obj, aruco_dict = self._get_aruco_detector()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = self._detect_markers(gray, detector_obj)
            
            if ids is None or len(ids) == 0:
                return "No ArUco markers detected in current view"
            
            # Calculate marker information
            marker_info = []
            frame_height, frame_width = gray.shape
            frame_center_x = frame_width / 2
            
            for i, marker_id in enumerate(ids.flatten()):
                # Calculate marker center
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])
                
                # Estimate distance based on marker size in pixels
                # Assuming 15cm real marker size
                marker_width_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
                # Simple distance estimation (calibrate this for your camera)
                estimated_distance_cm = (15.0 * 500) / marker_width_pixels
                
                # Determine position relative to frame center
                offset_x = center_x - frame_center_x
                if offset_x < -50:
                    position = "LEFT"
                elif offset_x > 50:
                    position = "RIGHT"
                else:
                    position = "CENTER"
                
                # Get object name from mapping
                object_name = self.marker_objects.get(int(marker_id), f"Unknown Marker {marker_id}")
                
                marker_info.append({
                    "id": int(marker_id),
                    "object": object_name,
                    "distance_cm": round(estimated_distance_cm, 1),
                    "position": position,
                    "center_x": round(center_x, 1),
                    "center_y": round(center_y, 1)
                })
            
            # Format response
            result = f"Detected {len(marker_info)} object(s): "
            details = []
            for m in marker_info:
                details.append(f"{m['object']} (Marker {m['id']}) at ~{m['distance_cm']}cm, position {m['position']}")
            result += "; ".join(details)
            
            logger.info(f"ðŸŽ¯ ArUco Detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"ArUco detection error: {e}")
            return f"ArUco detection failed: {str(e)}"

    @function_tool
    async def navigate_to_object(self, context: RunContext, object_name: str, target_distance_cm: int = 25):
        """Navigate to an object by its name (monster_drink, water_bottle, cube, or laptop)."""
        # Find marker ID for this object
        marker_id = None
        for mid, obj_name in self.marker_objects.items():
            if obj_name.lower() == object_name.lower():
                marker_id = mid
                break
        
        if marker_id is None:
            return f"Unknown object '{object_name}'. Available objects: {', '.join(self.marker_objects.values())}"
        
        return await self.navigate_to_marker(None, marker_id, target_distance_cm)

    @function_tool
    async def navigate_to_marker(self, context: RunContext, marker_id: int, target_distance_cm: int = 25):
        """Autonomously navigate to a specific ArUco marker using visual servoing."""
        if self.navigation_active:
            return "Navigation already in progress. Stop current navigation first."
        
        if self.latest_frame is None:
            return "Camera must be active to navigate"
        
        self.navigation_active = True
        object_name = self.marker_objects.get(marker_id, f"Marker {marker_id}")
        
        logger.info(f"ðŸš— Starting autonomous navigation to {object_name} (Marker {marker_id})")
        
        try:
            import cv2
            
            # Get detector
            detector_obj, aruco_dict = self._get_aruco_detector()
            
            max_iterations = 60
            consecutive_not_found = 0
            max_not_found = 5
            
            for iteration in range(max_iterations):
                if not self.navigation_active:
                    await self.stop_robot(None)
                    return "Navigation cancelled by user"
                
                # Detect markers in current frame
                frame_bgr = cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self._detect_markers(gray, detector_obj)
                
                # Check if target marker is visible
                target_found = False
                if ids is not None:
                    for i, detected_id in enumerate(ids.flatten()):
                        if detected_id == marker_id:
                            target_found = True
                            marker_corners = corners[i][0]
                            
                            # Calculate marker properties
                            center_x = np.mean(marker_corners[:, 0])
                            frame_height, frame_width = gray.shape
                            frame_center_x = frame_width / 2
                            
                            marker_width_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
                            estimated_distance_cm = (15.0 * 500) / marker_width_pixels
                            
                            offset_x = center_x - frame_center_x
                            
                            logger.info(f"ðŸŽ¯ Iter {iteration}: {object_name} at {estimated_distance_cm:.1f}cm, offset {offset_x:.1f}px")
                            
                            # Check if reached target
                            if estimated_distance_cm <= target_distance_cm:
                                await self.stop_robot(None)
                                self.navigation_active = False
                                return f"Successfully reached {object_name}! Final distance: {estimated_distance_cm:.1f}cm"
                            
                            # Determine action based on position and distance
                            if abs(offset_x) > 80:
                                # Need to turn to center marker
                                if offset_x < 0:
                                    await self.turn_left(None, 25, 0.3)
                                    logger.info("â†¶ Turning left to center marker")
                                else:
                                    await self.turn_right(None, 25, 0.3)
                                    logger.info("â†· Turning right to center marker")
                            else:
                                # Marker centered, move forward
                                move_duration = min(0.8, estimated_distance_cm / 40)
                                await self.move_forward(None, 25, move_duration)
                                logger.info(f"â†‘ Moving forward {move_duration:.1f}s")
                            
                            consecutive_not_found = 0
                            break
                
                if not target_found:
                    consecutive_not_found += 1
                    logger.warning(f"âš ï¸ {object_name} not visible ({consecutive_not_found}/{max_not_found})")
                    
                    if consecutive_not_found >= max_not_found:
                        # Try to find marker by panning camera or rotating
                        if consecutive_not_found == max_not_found:
                            await self.camera_pan(None, 60)  # Look left
                            await asyncio.sleep(0.5)
                        elif consecutive_not_found == max_not_found + 2:
                            await self.camera_pan(None, 120)  # Look right
                            await asyncio.sleep(0.5)
                        elif consecutive_not_found == max_not_found + 4:
                            await self.camera_center(None)
                            await self.turn_left(None, 25, 0.5)  # Rotate robot
                        elif consecutive_not_found >= max_not_found + 8:
                            await self.stop_robot(None)
                            await self.camera_center(None)
                            self.navigation_active = False
                            return f"Lost sight of {object_name}. Navigation aborted."
                
                await asyncio.sleep(0.5)  # Control loop rate
            
            # Max iterations reached
            await self.stop_robot(None)
            self.navigation_active = False
            return f"Navigation timeout after {max_iterations} iterations"
            
        except Exception as e:
            await self.stop_robot(None)
            self.navigation_active = False
            logger.error(f"Navigation error: {e}")
            return f"Navigation failed: {str(e)}"

    @function_tool
    async def stop_navigation(self, context: RunContext):
        """Stop autonomous navigation."""
        if self.navigation_active:
            self.navigation_active = False
            await self.stop_robot(None)
            return "Navigation stopped"
        return "No active navigation"

    @function_tool
    async def toggle_bounding_boxes(self, context: RunContext, enabled: bool = True):
        """Enable or disable ArUco marker bounding box visualization."""
        self.show_bounding_boxes = enabled
        status = "enabled" if enabled else "disabled"
        return f"Bounding box visualization {status}"

    @function_tool
    async def list_tracked_objects(self, context: RunContext):
        """List all objects currently tracked by ArUco markers."""
        objects_list = []
        for marker_id, obj_name in sorted(self.marker_objects.items()):
            objects_list.append(f"Marker {marker_id}: {obj_name}")
        return "Tracked objects: " + ", ".join(objects_list)

    @function_tool
    async def set_camera_zoom(self, context: RunContext, zoom_level: float = 1.0):
        """Adjust camera zoom. 1.0=widest, 2.0=2x, 4.0=4x"""
        zoom_level = max(1.0, min(4.0, float(zoom_level)))
        self.camera_zoom = zoom_level
        return f"Zoom: {zoom_level}x. Restart camera to apply."

    async def _stream_camera(self):
        if '/usr/lib/python3/dist-packages' not in sys.path:
            sys.path.insert(0, '/usr/lib/python3/dist-packages')
        
        try:
            from picamera2 import Picamera2
            import cv2
        except ImportError as e:
            logger.error(f"Import error: {e}")
            return
        
        picam = None
        try:
            capture_width, capture_height = 1280, 720
            output_width, output_height = 640, 480
            fps = 30
            
            picam = Picamera2()
            
            # Use BGR888 format which is standard for cameras
            config = picam.create_video_configuration(
                main={"size": (output_width, output_height), "format": "BGR888"},
                raw={"size": (capture_width, capture_height)},
                buffer_count=2,
            )
            picam.configure(config)
            
            sensor_width, sensor_height = picam.sensor_resolution
            zoom = self.camera_zoom
            crop_width = int(sensor_width / zoom)
            crop_height = int(sensor_height / zoom)
            crop_x = (sensor_width - crop_width) // 2
            crop_y = (sensor_height - crop_height) // 2
            
            picam.set_controls({
                "FrameDurationLimits": (33333, 33333),
                "ScalerCrop": (crop_x, crop_y, crop_width, crop_height),
            })
            picam.start()
            
            for _ in range(2):
                picam.capture_array()
                await asyncio.sleep(0.05)
            
            logger.info(f"ðŸ“· Streaming {output_width}x{output_height}@{fps}fps (BGR888 -> RGB) with bounding boxes")
            frame_count = 0
            
            # Setup ArUco detection with version compatibility
            detector_obj, aruco_dict = self._get_aruco_detector()
            
            while True:
                if not self.video_source:
                    break
                
                # Capture frame in BGR format
                frame_bgr = picam.capture_array()
                
                if len(frame_bgr.shape) == 3:
                    if frame_bgr.shape[2] == 4:
                        frame_bgr = frame_bgr[:, :, :3]
                    elif frame_bgr.shape[2] != 3:
                        continue
                
                # Convert BGR to RGB for correct colors
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                if not frame_rgb.flags['C_CONTIGUOUS']:
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                
                # Store clean frame for vision analysis
                if self.vision_enabled and frame_count % 15 == 0:
                    self.latest_frame = frame_rgb.copy()
                
                # Draw bounding boxes on display frame
                display_frame = frame_rgb
                if self.show_bounding_boxes:
                    # Detect markers
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = self._detect_markers(gray, detector_obj)
                    
                    if ids is not None and len(ids) > 0:
                        display_frame = self._draw_aruco_bounding_boxes(frame_rgb, corners, ids)
                        self.latest_frame_with_boxes = display_frame.copy()
                
                # Send frame to LiveKit
                video_frame = rtc.VideoFrame(
                    output_width, output_height,
                    rtc.VideoBufferType.RGB24,
                    display_frame.tobytes()
                )
                self.video_source.capture_frame(video_frame)
                
                frame_count += 1
                if frame_count % 150 == 0:
                    logger.info(f"ðŸ“¸ {frame_count} frames")
                
                await asyncio.sleep(1 / fps)
                
        except asyncio.CancelledError:
            logger.info("ðŸ“· Stopped")
        except Exception as e:
            logger.error(f"ðŸ“· Error: {e}")
        finally:
            if picam:
                try:
                    picam.stop()
                except:
                    pass

    @function_tool
    async def open_camera(self, context: RunContext):
        """Start streaming video."""
        if self.camera_task and not self.camera_task.done():
            return "Camera running"
        if not self.current_room:
            return "No room"
        
        try:
            width, height = 640, 480
            self.video_source = rtc.VideoSource(width, height)
            self.video_track = rtc.LocalVideoTrack.create_video_track("robot_camera", self.video_source)
            
            options = rtc.TrackPublishOptions(
                source=rtc.TrackSource.SOURCE_CAMERA,
                video_codec=rtc.VideoCodec.H264,
                video_encoding=rtc.VideoEncoding(max_bitrate=2_000_000, max_framerate=30),
            )
            
            await self.current_room.local_participant.publish_track(self.video_track, options)
            self.camera_task = asyncio.create_task(self._stream_camera())
            return "Camera started with ArUco detection and bounding boxes"
        except Exception as e:
            logger.error(f"Camera: {e}")
            return f"Error: {e}"

    @function_tool
    async def close_camera(self, context: RunContext):
        """Stop streaming video."""
        try:
            if self.camera_task and not self.camera_task.done():
                self.camera_task.cancel()
                try:
                    await self.camera_task
                except asyncio.CancelledError:
                    pass
            
            if self.video_track and self.current_room:
                await self.current_room.local_participant.unpublish_track(self.video_track.sid)
            
            self.video_source = None
            self.video_track = None
            self.camera_task = None
            self.latest_frame = None
            self.latest_frame_with_boxes = None
            return "Camera stopped"
        except Exception as e:
            return f"Error: {e}"

    async def do_open_camera(self):
        if not self.camera_task or self.camera_task.done():
            await self.open_camera(None)

    async def do_close_camera(self):
        await self.close_camera(None)


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    assistant = Assistant()
    assistant.current_room = ctx.room
    
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4o"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    await session.start(agent=assistant, room=ctx.room)
    await ctx.connect()
    
    await asyncio.sleep(0.5)
    try:
        await assistant.camera_center(None)
    except:
        pass
    
    await asyncio.sleep(0.5)
    try:
        await assistant.do_open_camera()
    except:
        pass
    
    def handle_car_control(pkt: rtc.DataPacket):
        try:
            if getattr(pkt, "topic", "") != "car-control":
                return
            
            payload = json.loads(bytes(pkt.data).decode("utf-8"))
            msg_type = payload.get("type")
            speed = int(payload.get("speed", 30))
            duration = float(payload.get("duration", 1.0))
            angle = int(payload.get("angle", 90))
            marker_id = int(payload.get("marker_id", 0))
            object_name = payload.get("object_name", "")
            
            tasks = {
                "forward": lambda: assistant.move_forward(None, speed, duration),
                "backward": lambda: assistant.move_backward(None, speed, duration),
                "left": lambda: assistant.turn_left(None, speed, duration),
                "right": lambda: assistant.turn_right(None, speed, duration),
                "stop": lambda: assistant.stop_robot(None),
                "open_camera": lambda: assistant.do_open_camera(),
                "close_camera": lambda: assistant.do_close_camera(),
                "servo_pan": lambda: assistant.camera_pan(None, angle),
                "servo_tilt": lambda: assistant.camera_tilt(None, angle),
                "camera_center": lambda: assistant.camera_center(None),
                "camera_look_left": lambda: assistant.camera_look_left(None),
                "camera_look_right": lambda: assistant.camera_look_right(None),
                "camera_look_up": lambda: assistant.camera_look_up(None),
                "camera_look_down": lambda: assistant.camera_look_down(None),
                "describe_view": lambda: assistant.describe_view(None),
                "detect_aruco": lambda: assistant.detect_aruco_markers(None),
                "navigate_to_marker": lambda: assistant.navigate_to_marker(None, marker_id),
                "navigate_to_object": lambda: assistant.navigate_to_object(None, object_name),
                "stop_navigation": lambda: assistant.stop_navigation(None),
                "list_objects": lambda: assistant.list_tracked_objects(None),
            }
            
            if msg_type in tasks:
                asyncio.create_task(tasks[msg_type]())
        except Exception as e:
            logger.error(f"Control: {e}")
    
    ctx.room.on("data_received", handle_car_control)
    logger.info("âœ… Agent ready with ArUco object tracking: monster_drink, water_bottle, cube, laptop")

if __name__ == "__main__":
    cli.run_app(server)