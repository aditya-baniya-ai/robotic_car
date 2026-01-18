'use client';

import { useCallback, useMemo, useState } from 'react';
import { useDataChannel, useSessionContext } from '@livekit/components-react';
import { Button } from '@/components/livekit/button';
import { cn } from '@/lib/utils';

const SPEED_MIN = 0;
const SPEED_MAX = 100;
const DURATION_MIN = 0.1;
const DURATION_MAX = 10;
const ANGLE_MIN = 0;
const ANGLE_MAX = 180;

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));

type CommandType =
  | 'forward'
  | 'backward'
  | 'left'
  | 'right'
  | 'stop'
  | 'open_camera'
  | 'close_camera'
  | 'servo_pan'
  | 'servo_tilt'
  | 'camera_center'
  | 'camera_look_left'
  | 'camera_look_right'
  | 'camera_look_up'
  | 'camera_look_down'
  | 'describe_view';

type CommandPayload = {
  type: CommandType;
  speed?: number;
  duration?: number;
  angle?: number;
};

export function CarControlPanel() {
  const session = useSessionContext();
  const { send, isSending } = useDataChannel('car-control');
  const [speed, setSpeed] = useState(30);
  const [duration, setDuration] = useState(1);
  const [angle, setAngle] = useState(90);
  const [lastCommand, setLastCommand] = useState<string | null>(null);
  const isConnected = session.isConnected;

  const disabled = !isConnected || isSending;

  const statusLabel = useMemo(() => (isConnected ? 'Connected' : 'Disconnected'), [isConnected]);

  const sendCommand = useCallback(
    async (type: CommandType, overrides: Partial<CommandPayload> = {}) => {
      if (!isConnected) {
        setLastCommand('Connect first');
        return;
      }

      const payload: CommandPayload = {
        type,
        speed: clamp(speed, SPEED_MIN, SPEED_MAX),
        duration: clamp(duration, DURATION_MIN, DURATION_MAX),
        angle: clamp(angle, ANGLE_MIN, ANGLE_MAX),
        ...overrides,
      };

      const encoded = new TextEncoder().encode(JSON.stringify(payload));
      await send(encoded, { reliable: true });
      setLastCommand(type.replace(/_/g, ' '));
    },
    [angle, duration, isConnected, send, speed]
  );

  return (
    <div className="border-input/50 bg-background/80 flex flex-col gap-4 rounded-3xl border p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Command Center
          </p>
          <p className="text-sm font-medium text-foreground">PiCar-X Controls</p>
        </div>
        <span
          className={cn(
            'rounded-full px-2 py-1 text-[10px] font-semibold uppercase tracking-wider',
            isConnected ? 'bg-emerald-500/15 text-emerald-600' : 'bg-amber-500/15 text-amber-600'
          )}
        >
          {statusLabel}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs">
        <Button size="sm" className="col-start-2" disabled={disabled} onClick={() => sendCommand('forward')}>
          Forward
        </Button>
        <Button size="sm" disabled={disabled} onClick={() => sendCommand('left')}>
          Left
        </Button>
        <Button size="sm" disabled={disabled} onClick={() => sendCommand('stop')} variant="destructive">
          Stop
        </Button>
        <Button size="sm" disabled={disabled} onClick={() => sendCommand('right')}>
          Right
        </Button>
        <Button size="sm" className="col-start-2" disabled={disabled} onClick={() => sendCommand('backward')}>
          Backward
        </Button>
      </div>

      <div className="grid grid-cols-3 gap-3 text-xs">
        <label className="flex flex-col gap-1">
          <span className="text-muted-foreground uppercase tracking-wide">Speed</span>
          <input
            type="number"
            min={SPEED_MIN}
            max={SPEED_MAX}
            value={speed}
            onChange={(e) => setSpeed(clamp(Number(e.target.value), SPEED_MIN, SPEED_MAX))}
            className="border-input/50 bg-background h-8 rounded-md border px-2 text-sm focus:outline-none"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-muted-foreground uppercase tracking-wide">Duration</span>
          <input
            type="number"
            min={DURATION_MIN}
            max={DURATION_MAX}
            step="0.1"
            value={duration}
            onChange={(e) => setDuration(clamp(Number(e.target.value), DURATION_MIN, DURATION_MAX))}
            className="border-input/50 bg-background h-8 rounded-md border px-2 text-sm focus:outline-none"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-muted-foreground uppercase tracking-wide">Angle</span>
          <input
            type="number"
            min={ANGLE_MIN}
            max={ANGLE_MAX}
            value={angle}
            onChange={(e) => setAngle(clamp(Number(e.target.value), ANGLE_MIN, ANGLE_MAX))}
            className="border-input/50 bg-background h-8 rounded-md border px-2 text-sm focus:outline-none"
          />
        </label>
      </div>

      <div className="grid gap-2 text-xs">
        <p className="text-muted-foreground text-xs font-semibold uppercase tracking-wider">
          Camera
        </p>
        <div className="grid grid-cols-2 gap-2">
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('open_camera')}>
            Open Camera
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('close_camera')}>
            Close Camera
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('camera_center')}>
            Center
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('describe_view')}>
            Describe View
          </Button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('camera_look_left')}>
            Look Left
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('camera_look_right')}>
            Look Right
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('camera_look_up')}>
            Look Up
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('camera_look_down')}>
            Look Down
          </Button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('servo_pan')}>
            Pan To Angle
          </Button>
          <Button size="sm" disabled={disabled} onClick={() => sendCommand('servo_tilt')}>
            Tilt To Angle
          </Button>
        </div>
      </div>

      {lastCommand && (
        <p className="text-muted-foreground text-xs">
          Last command: <span className="text-foreground">{lastCommand}</span>
        </p>
      )}
    </div>
  );
}
