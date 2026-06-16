"use client";

import { useEffect, useRef, useState } from "react";
import { Loader2, Video, VideoOff } from "lucide-react";

import { api } from "@/lib/api";
import type { CameraStatus } from "@/lib/types";

/**
 * Polls the backend annotated snapshot (bounding boxes + recognition labels
 * drawn server-side) and renders it as a live feed. Works identically for a
 * webcam and an uploaded video file streamed through the camera service.
 */
export function DetectionFeed({
  onStatus,
  pollMs = 350,
}: {
  onStatus?: (s: CameraStatus) => void;
  pollMs?: number;
}) {
  const [frame, setFrame] = useState<string | null>(null);
  const [status, setStatus] = useState<CameraStatus | null>(null);
  const onStatusRef = useRef(onStatus);
  onStatusRef.current = onStatus;
  const lastUrl = useRef<string | null>(null);

  useEffect(() => {
    let closed = false;

    const poll = async () => {
      if (closed) return;
      try {
        const s = await api.get<CameraStatus>("camera/status");
        setStatus(s);
        onStatusRef.current?.(s);

        if (s.is_running) {
          const res = await fetch(`/api/backend/camera/snapshot?t=${Date.now()}`);
          if (res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            setFrame(url);
            if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
            lastUrl.current = url;
          }
        } else {
          setFrame(null);
        }
      } catch {
        /* transient — keep last frame */
      }
    };

    const id = setInterval(poll, pollMs);
    poll();
    return () => {
      closed = true;
      clearInterval(id);
      if (lastUrl.current) URL.revokeObjectURL(lastUrl.current);
    };
  }, [pollMs]);

  const running = status?.is_running;
  const isVideo = status?.source_kind === "video";

  return (
    <div className="relative aspect-video w-full overflow-hidden rounded-card border border-white/10 bg-black shadow-card">
      {frame ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={frame} alt="Detection feed" className="h-full w-full object-contain" />
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-3 text-text-muted">
          {running ? (
            <Loader2 className="h-9 w-9 animate-spin" />
          ) : (
            <VideoOff className="h-9 w-9" />
          )}
          <p className="text-sm">
            {running ? "Buffering frames…" : "No active stream."}
          </p>
        </div>
      )}

      {/* Status pill */}
      <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full border border-white/10 bg-black/60 px-3 py-1 text-xs font-medium backdrop-blur">
        <span
          className={`h-2 w-2 rounded-full ${
            running ? "animate-pulse bg-success" : "bg-danger"
          }`}
        />
        {running ? (isVideo ? "STREAMING VIDEO" : "LIVE") : "OFFLINE"}
        {status?.looping && running && (
          <span className="text-text-muted">· loop</span>
        )}
      </div>

      {/* FPS / source pill */}
      {running && (
        <div className="absolute right-3 top-3 flex items-center gap-1.5 rounded-full border border-white/10 bg-black/60 px-3 py-1 text-xs text-text-secondary backdrop-blur">
          <Video className="h-3 w-3" />
          {status?.fps ?? "—"} fps
        </div>
      )}
    </div>
  );
}
