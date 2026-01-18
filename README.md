# Nexhacks PiCar LiveKit Agent

Hackathon project for **NExhacks**: a Raspberry Pi–powered robotic car that connects to LiveKit, streams live video, and accepts voice + UI commands in real time. The frontend provides a clean operator console with a robot camera view on the left and command controls on the right.

## What it does
- Two-way, low-latency voice conversation with a LiveKit agent.
- Live robot camera streaming from the Pi to the web UI.
- Remote control commands (drive, stop, pan/tilt camera) over LiveKit data channels.
- Optional vision description using an LLM (if `OPENAI_API_KEY` is configured).

## Architecture
1) **Raspberry Pi Agent** (`backend/agent.py`)
   - LiveKit Agents runtime
   - PiCar-X motion control via Python scripts
   - Picamera2 stream piped to LiveKit video track
2) **LiveKit Cloud**
   - Room/session orchestration, audio/video, and data channels
3) **Web Frontend** (`car-frontend/`)
   - Next.js control console
   - Connects to LiveKit using a token minted by `/api/connection-details`

## Commands supported
- Drive: `forward`, `backward`, `left`, `right`, `stop`
- Camera stream: `open_camera`, `close_camera`
- Camera movement: `servo_pan`, `servo_tilt`, `camera_center`
- Camera view: `camera_look_left`, `camera_look_right`, `camera_look_up`, `camera_look_down`
- Vision: `describe_view`

Commands are sent via LiveKit data channel topic `car-control`.

## Tech stack
- **Backend**: Python, LiveKit Agents, Picamera2, OpenCV, NumPy
- **Frontend**: Next.js (React), LiveKit JS SDK
- **Infra**: LiveKit Cloud

## Setup

### 1) LiveKit project
Create a LiveKit Cloud project and grab:
- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`

### 2) Raspberry Pi agent
On the Pi:
1. Copy the control scripts in `backend/` to your target path or update the constants in `backend/agent.py` to match your file locations.
2. Create `.env.local` next to `backend/agent.py` with at least:
   ```env
   LIVEKIT_URL=...
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   ```
3. If you want vision description, add:
   ```env
   OPENAI_API_KEY=...
   ```
4. Install dependencies (add your needed deps to `requirements.txt`):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
5. Start the agent:
   ```bash
   python backend/agent.py
   ```

### 3) Frontend
From your laptop:
1. Create `car-frontend/.env.local`:
   ```env
   LIVEKIT_URL=...
   LIVEKIT_API_KEY=...
   LIVEKIT_API_SECRET=...
   AGENT_NAME=pi-car
   ```
2. Install and run:
   ```bash
   cd car-frontend
   pnpm install
   pnpm dev
   ```
3. Open `http://localhost:3000` and press **Start call**.

## Repository structure
```
.
├── backend/                 # Raspberry Pi agent + control scripts
├── car-frontend/            # Next.js operator UI
├── requirements.txt         # Python deps (fill as needed)
└── README.md
```

## Notes for demo
- Make sure the Pi and frontend point to the same LiveKit project.
- If the UI connects but video is blank, use the **Open Camera** command.
- If commands do nothing, verify the data channel topic is `car-control`.

## Hackathon pitch (one-liner)
**A LiveKit-powered robotic car that talks, sees, and moves in real time—controlled by voice and a web console.**
