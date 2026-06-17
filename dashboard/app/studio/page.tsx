"use client";

import { useCallback, useRef, useState } from "react";
import useSWR from "swr";
import {
  CheckCircle2,
  CloudUpload,
  Film,
  Loader2,
  Repeat,
  ScanFace,
  Square,
  UserCheck,
  UserPlus,
  Users,
  UsersRound,
} from "lucide-react";

import { api, fetcher } from "@/lib/api";
import type {
  ActivityResponse,
  CameraStatus,
  VideoStreamResponse,
  VisitorListResponse,
} from "@/lib/types";
import { DetectionFeed } from "@/components/detection-feed";
import { ActivityFeed } from "@/components/activity-feed";
import { StatCard } from "@/components/stat-card";
import {
  Badge,
  Button,
  Card,
  CardTitle,
  Input,
  PageHeader,
  Toggle,
} from "@/components/ui";

const ACCEPT = ".mp4,.avi,.mov,.mkv,.webm";

export default function VideoStudioPage() {
  const [file, setFile] = useState<File | null>(null);
  const [fps, setFps] = useState("2");
  const [loop, setLoop] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [status, setStatus] = useState<CameraStatus | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { data: activity } = useSWR<ActivityResponse>("activity?limit=10", fetcher, {
    refreshInterval: 3000,
  });
  // All-time registered visitor count (active, non-staff) — `total` from the
  // visitors list; limit=1 keeps the payload tiny since we only read the count.
  const { data: visitorList } = useSWR<VisitorListResponse>(
    "visitors?limit=1",
    fetcher,
    { refreshInterval: 5000 },
  );

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  async function startStream() {
    if (!file) return;
    setBusy(true);
    setMsg(null);
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("fps", fps);
      form.append("loop", String(loop));
      const res = await api.upload<VideoStreamResponse>("camera/upload-video", form);
      setMsg({
        kind: "ok",
        text: `Streaming "${res.filename}" (${res.size_mb} MB)${
          res.looping ? " · looping" : ""
        }.`,
      });
    } catch (e) {
      setMsg({ kind: "err", text: (e as Error).message });
    } finally {
      setBusy(false);
    }
  }

  async function stopStream() {
    setBusy(true);
    try {
      await api.post("camera/stop");
      setMsg(null);
    } finally {
      setBusy(false);
    }
  }

  const running = status?.is_running;
  const s = status;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Video Studio"
        subtitle="Upload a video and watch it run through live face & body detection."
        action={
          running ? (
            <Button variant="danger" onClick={stopStream} disabled={busy}>
              <Square className="h-4 w-4" /> Stop Stream
            </Button>
          ) : undefined
        }
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Feed + stats */}
        <div className="space-y-6 lg:col-span-2">
          <DetectionFeed onStatus={setStatus} />

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
            <StatCard
              label="Registered"
              value={visitorList?.total ?? "—"}
              hint="all-time total"
              icon={<UsersRound className="h-5 w-5" />}
              tone="primary"
            />
            <StatCard
              label="New"
              value={s?.new_visitors ?? "—"}
              hint="this session"
              icon={<UserPlus className="h-5 w-5" />}
              tone="success"
            />
            <StatCard
              label="Returning"
              value={s?.returning_visitors ?? "—"}
              hint="this session"
              icon={<UserCheck className="h-5 w-5" />}
              tone="warning"
            />
            <StatCard
              label="Persons"
              value={s?.persons_detected ?? "—"}
              icon={<Users className="h-5 w-5" />}
              tone="accent"
            />
            <StatCard
              label="Frames"
              value={s?.frames_processed ?? "—"}
              icon={<Film className="h-5 w-5" />}
              tone="primary"
            />
          </div>
        </div>

        {/* Upload panel */}
        <div className="space-y-6">
          <Card>
            <CardTitle>Upload Video</CardTitle>

            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
              className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-card border-2 border-dashed p-8 text-center transition ${
                dragOver
                  ? "border-primary bg-primary/10"
                  : "border-white/10 hover:border-white/20 hover:bg-white/5"
              }`}
            >
              <CloudUpload
                className={`h-8 w-8 ${dragOver ? "text-primary" : "text-text-muted"}`}
              />
              {file ? (
                <div className="space-y-1">
                  <p className="truncate text-sm font-medium text-text-primary">
                    {file.name}
                  </p>
                  <p className="text-xs text-text-muted">
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
              ) : (
                <>
                  <p className="text-sm text-text-secondary">
                    Drop a video here, or click to browse
                  </p>
                  <p className="text-xs text-text-muted">MP4, MOV, AVI, MKV, WEBM</p>
                </>
              )}
              <input
                ref={inputRef}
                type="file"
                accept={ACCEPT}
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>

            <div className="mt-4 space-y-4">
              <div className="flex items-center justify-between gap-3">
                <label className="text-sm text-text-secondary">Processing FPS</label>
                <div className="w-24">
                  <Input
                    type="number"
                    value={fps}
                    onChange={setFps}
                    min={1}
                    max={15}
                    step={1}
                  />
                </div>
              </div>
              <div className="flex items-center justify-between gap-3">
                <label className="flex items-center gap-2 text-sm text-text-secondary">
                  <Repeat className="h-4 w-4" /> Loop video
                </label>
                <Toggle checked={loop} onChange={setLoop} />
              </div>

              <Button
                className="w-full"
                onClick={startStream}
                disabled={!file || busy}
              >
                {busy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ScanFace className="h-4 w-4" />
                )}
                {running ? "Restart with this video" : "Start Detection"}
              </Button>

              {msg && (
                <div
                  className={`flex items-start gap-2 rounded-control p-3 text-xs ${
                    msg.kind === "ok"
                      ? "bg-success/10 text-success"
                      : "bg-danger/10 text-danger"
                  }`}
                >
                  {msg.kind === "ok" && <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />}
                  {msg.text}
                </div>
              )}
            </div>
          </Card>

          <Card>
            <CardTitle
              action={running ? <Badge tone="success">live</Badge> : undefined}
            >
              Live Detections
            </CardTitle>
            <ActivityFeed events={activity?.events ?? []} compact />
          </Card>
        </div>
      </div>
    </div>
  );
}
