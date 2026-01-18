'use client';

import { BarVisualizer, VideoTrack, useVoiceAssistant } from '@livekit/components-react';
import { cn } from '@/lib/utils';

export function RobotView() {
  const { state: agentState, audioTrack: agentAudioTrack, videoTrack: agentVideoTrack } =
    useVoiceAssistant();

  const isVideoReady = agentVideoTrack !== undefined;

  return (
    <div className="border-input/50 bg-muted/30 relative h-full w-full overflow-hidden rounded-3xl border shadow-xl">
      {isVideoReady ? (
        <VideoTrack
          trackRef={agentVideoTrack}
          className="h-full w-full object-cover"
          width={agentVideoTrack?.publication.dimensions?.width ?? 0}
          height={agentVideoTrack?.publication.dimensions?.height ?? 0}
        />
      ) : (
        <div className="flex h-full w-full flex-col items-center justify-center gap-4 p-6 text-center">
          <BarVisualizer
            barCount={5}
            state={agentState}
            options={{ minHeight: 6 }}
            trackRef={agentAudioTrack}
            className={cn('flex items-center justify-center gap-2')}
          >
            <span
              className={cn([
                'bg-muted-foreground/40 min-h-3 w-3 rounded-full',
                'origin-center transition-colors duration-250 ease-linear',
                'data-[lk-highlighted=true]:bg-foreground data-[lk-muted=true]:bg-muted',
              ])}
            />
          </BarVisualizer>
          <p className="text-muted-foreground text-sm">
            Waiting for the robot camera. Use the controls on the right to open the stream.
          </p>
        </div>
      )}

      <div className="bg-background/70 text-foreground absolute left-4 top-4 rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider">
        Robot View
      </div>
    </div>
  );
}
