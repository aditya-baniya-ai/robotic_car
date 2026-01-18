'use client';

import React, { useEffect, useRef, useState } from 'react';
import { useSessionContext, useSessionMessages } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { ChatTranscript } from '@/components/app/chat-transcript';
import { CarControlPanel } from '@/components/app/car-control-panel';
import { PreConnectMessage } from '@/components/app/preconnect-message';
import { RobotView } from '@/components/app/robot-view';
import {
  AgentControlBar,
  type ControlBarControls,
} from '@/components/livekit/agent-control-bar/agent-control-bar';
import { cn } from '@/lib/utils';
import { ScrollArea } from '../livekit/scroll-area/scroll-area';

interface FadeProps {
  top?: boolean;
  bottom?: boolean;
  className?: string;
}

export function Fade({ top = false, bottom = false, className }: FadeProps) {
  return (
    <div
      className={cn(
        'from-background pointer-events-none h-4 bg-linear-to-b to-transparent',
        top && 'bg-linear-to-b',
        bottom && 'bg-linear-to-t',
        className
      )}
    />
  );
}

interface SessionViewProps {
  appConfig: AppConfig;
}

export const SessionView = ({
  appConfig,
  ...props
}: React.ComponentProps<'section'> & SessionViewProps) => {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  const [chatOpen, setChatOpen] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const controls: ControlBarControls = {
    leave: true,
    microphone: true,
    chat: appConfig.supportsChatInput,
    camera: appConfig.supportsVideoInput,
    screenShare: appConfig.supportsScreenShare,
  };

  useEffect(() => {
    const lastMessage = messages.at(-1);
    const lastMessageIsLocal = lastMessage?.from?.isLocal === true;

    if (scrollAreaRef.current && lastMessageIsLocal) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <section className="bg-background relative z-10 h-full w-full overflow-hidden" {...props}>
      <div className="mx-auto flex h-full w-full max-w-6xl flex-col gap-5 p-4 md:flex-row md:gap-6 md:p-6">
        <div className="flex min-h-[320px] flex-1 flex-col">
          <RobotView />
        </div>

        <div className="flex w-full flex-col gap-4 md:w-[360px]">
          {appConfig.isPreConnectBufferEnabled && (
            <PreConnectMessage messages={messages} className="rounded-2xl border p-4" />
          )}

          <CarControlPanel />

          {chatOpen && (
            <div className="border-input/50 bg-background/80 flex min-h-[160px] flex-1 flex-col overflow-hidden rounded-2xl border">
              <div className="border-input/50 flex items-center justify-between border-b px-4 py-3">
                <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                  Transcript
                </span>
              </div>
              <div className="relative flex-1">
                <Fade top className="absolute inset-x-0 top-0 z-10 h-6" />
                <Fade bottom className="absolute inset-x-0 bottom-0 z-10 h-6" />
                <ScrollArea ref={scrollAreaRef} className="flex-1 px-4 py-3">
                  <ChatTranscript
                    messages={messages}
                    className="space-y-3 transition-opacity duration-300 ease-out"
                  />
                </ScrollArea>
              </div>
            </div>
          )}

          <AgentControlBar
            controls={controls}
            isConnected={session.isConnected}
            onDisconnect={session.end}
            onChatOpenChange={setChatOpen}
          />
        </div>
      </div>
    </section>
  );
};
