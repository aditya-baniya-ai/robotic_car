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


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You control a SunFounder PiCar-X with camera vision.
You can move forward, backward, turn left, turn right, and stop.
You can control the camera servo to pan left/right and tilt up/down.
You can see through the camera and describe what you observe.
You can open/close the camera and adjust zoom level.
Be concise and confirm actions.
""".strip(),
        )
        self.camera_task: asyncio.Task | None = None
        self.current_room = None
        self.video_source: rtc.VideoSource | None = None
        self.video_track: rtc.LocalVideoTrack | None = None
        self.camera_zoom: float = 1.0
        self.latest_frame: np.ndarray | None = None
        self.vision_enabled: bool = True

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
            logger.info(f"👁️ Vision: {description}")
            return description
            
        except Exception as e:
            logger.error(f"Vision error: {e}")
            return f"Vision analysis failed: {str(e)}"

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
            
            logger.info(f"📷 Streaming {output_width}x{output_height}@{fps}fps (BGR888 -> RGB)")
            frame_count = 0
            
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
                
                # Store latest frame for vision analysis (already in RGB)
                if self.vision_enabled and frame_count % 15 == 0:
                    self.latest_frame = frame_rgb.copy()
                
                # Send RGB frame to LiveKit
                video_frame = rtc.VideoFrame(
                    output_width, output_height,
                    rtc.VideoBufferType.RGB24,
                    frame_rgb.tobytes()
                )
                self.video_source.capture_frame(video_frame)
                
                frame_count += 1
                if frame_count % 150 == 0:
                    logger.info(f"📸 {frame_count} frames")
                
                await asyncio.sleep(1 / fps)
                
        except asyncio.CancelledError:
            logger.info("📷 Stopped")
        except Exception as e:
            logger.error(f"📷 Error: {e}")
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
            return "Camera started with vision"
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
            }
            
            if msg_type in tasks:
                asyncio.create_task(tasks[msg_type]())
        except Exception as e:
            logger.error(f"Control: {e}")
    
    ctx.room.on("data_received", handle_car_control)
    logger.info("✅ Agent ready with vision")

if __name__ == "__main__":
    cli.run_app(server)
