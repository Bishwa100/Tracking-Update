"use client";

import { useState, useRef, useEffect } from "react";
import useSWR from "swr";
import { Play, RefreshCcw, Square } from "lucide-react";

import { api, fetcher, imageUrl } from "@/lib/api";
import type {
  CameraStatus,
  RoiResponse,
  RegionOfInterest,
  VisitorListResponse,
} from "@/lib/types";
import { Badge, Button, Card, CardTitle, ErrorState } from "@/components/ui";
import { uptime } from "@/lib/format";

export default function CameraPage() {
  const { data: status, error, mutate } = useSWR<CameraStatus>("camera/status", fetcher, {
    refreshInterval: 3000,
  });
  const { data: roiData } = useSWR<RoiResponse>("camera/roi", fetcher);
  // All-time registered visitor count (active, non-staff); limit=1 → just the total.
  const { data: visitorList } = useSWR<VisitorListResponse>(
    "visitors?limit=1",
    fetcher,
    { refreshInterval: 5000 },
  );

  const [source, setSource] = useState("0");
  const [fps, setFps] = useState("1.0");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [snapTick, setSnapTick] = useState(Date.now());

  // ROI drawing state
  const [roi, setRoi] = useState<RegionOfInterest | null>(null);
  const [savedRoi, setSavedRoi] = useState<RegionOfInterest | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [startPt, setStartPt] = useState<{ x: number; y: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Load existing ROI on mount
  useEffect(() => {
    if (roiData?.roi) {
      setSavedRoi(roiData.roi);
    }
  }, [roiData]);

  async function start() {
    setBusy(true);
    setMsg(null);
    try {
      await api.post("camera/start", { source, fps: parseFloat(fps) });
      await mutate();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setBusy(true);
    setMsg(null);
    try {
      await api.post("camera/stop");
      await mutate();
    } catch (e) {
      setMsg((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function getImgRect(img: HTMLImageElement) {
    const scaleX = img.clientWidth / img.naturalWidth;
    const scaleY = img.clientHeight / img.naturalHeight;
    const scale = Math.min(scaleX, scaleY);
    const rw = img.naturalWidth * scale;
    const rh = img.naturalHeight * scale;
    return {
      offsetX: (img.clientWidth - rw) / 2,
      offsetY: (img.clientHeight - rh) / 2,
      scale,
    };
  }

  function canvasToImage(cx: number, cy: number) {
    if (!imgRef.current) return { x: cx, y: cy };
    const { offsetX, offsetY, scale } = getImgRect(imgRef.current);
    return {
      x: Math.round((cx - offsetX) / scale),
      y: Math.round((cy - offsetY) / scale),
    };
  }

  function handleMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!canvasRef.current || !imgRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    setStartPt({ x: cx, y: cy });
    setDrawing(true);
  }

  function handleMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawing || !startPt || !canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    const x1 = Math.min(startPt.x, cx);
    const y1 = Math.min(startPt.y, cy);
    const x2 = Math.max(startPt.x, cx);
    const y2 = Math.max(startPt.y, cy);

    const topLeft = canvasToImage(x1, y1);
    const botRight = canvasToImage(x2, y2);
    setRoi({
      x1: topLeft.x,
      y1: topLeft.y,
      x2: botRight.x,
      y2: botRight.y,
    });
  }

  function handleMouseUp() {
    setDrawing(false);
  }

  async function saveRoi() {
    if (!roi) return;
    try {
      await api.post("camera/roi", { roi });
      setSavedRoi(roi);
      setRoi(null);
    } catch (e) {
      alert("Failed to save ROI: " + (e as Error).message);
    }
  }

  async function clearRoi() {
    try {
      await api.post("camera/roi", { roi: null });
      setSavedRoi(null);
      setRoi(null);
    } catch (e) {
      alert("Failed to clear ROI: " + (e as Error).message);
    }
  }

  // Redraw canvas when ROI state changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw saved ROI (solid blue)
    if (savedRoi && imgRef.current) {
      const { offsetX, offsetY, scale } = getImgRect(imgRef.current);
      const x1 = offsetX + savedRoi.x1 * scale;
      const y1 = offsetY + savedRoi.y1 * scale;
      const x2 = offsetX + savedRoi.x2 * scale;
      const y2 = offsetY + savedRoi.y2 * scale;
      ctx.strokeStyle = "rgb(59, 130, 246)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = "rgba(59, 130, 246, 0.05)";
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }

    // Draw in-progress ROI (dashed orange)
    if (roi && drawing && imgRef.current) {
      const { offsetX, offsetY, scale } = getImgRect(imgRef.current);
      const x1 = offsetX + roi.x1 * scale;
      const y1 = offsetY + roi.y1 * scale;
      const x2 = offsetX + roi.x2 * scale;
      const y2 = offsetY + roi.y2 * scale;
      ctx.strokeStyle = "rgb(255, 140, 0)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
  }, [roi, savedRoi, drawing]);

  // Sync canvas size with image
  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    const onLoad = () => {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
    };

    img.addEventListener("load", onLoad);
    onLoad();
    return () => img.removeEventListener("load", onLoad);
  }, [snapTick]);

  const running = !!status?.is_running;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Camera Management</h1>

      {error && <ErrorState message="Could not reach the backend camera API." />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardTitle>Camera Source</CardTitle>
          <div className="space-y-3">
            <label className="block text-sm">
              <span className="text-text-secondary">Source (0 = webcam, rtsp://…, or file path)</span>
              <input
                value={source}
                onChange={(e) => setSource(e.target.value)}
                disabled={running}
                className="mt-1 w-full rounded-control border border-card/60 bg-bg px-3 py-2 text-sm outline-none focus:border-primary disabled:opacity-50"
              />
            </label>
            <label className="block text-sm">
              <span className="text-text-secondary">FPS</span>
              <input
                type="number"
                step="0.5"
                min="0.5"
                value={fps}
                onChange={(e) => setFps(e.target.value)}
                disabled={running}
                className="mt-1 w-28 rounded-control border border-card/60 bg-bg px-3 py-2 text-sm outline-none focus:border-primary disabled:opacity-50"
              />
            </label>
            <div className="flex gap-2">
              <Button variant="success" onClick={start} disabled={busy || running}>
                <Play className="h-4 w-4" /> Start
              </Button>
              <Button variant="danger" onClick={stop} disabled={busy || !running}>
                <Square className="h-4 w-4" /> Stop
              </Button>
            </div>
            {msg && <p className="text-sm text-danger">{msg}</p>}
          </div>
        </Card>

        <Card>
          <CardTitle>Status</CardTitle>
          <dl className="space-y-2 text-sm">
            <Row label="State">
              {running ? <Badge tone="success">Running</Badge> : <Badge tone="danger">Stopped</Badge>}
            </Row>
            <Row label="Source">{status?.source ?? "—"}</Row>
            <Row label="FPS">{status?.fps ?? "—"}</Row>
            <Row label="Uptime">{status ? uptime(status.uptime_seconds) : "—"}</Row>
            <Row label="Frames processed">{status?.frames_processed ?? 0}</Row>
            <Row label="Frames skipped (dedup)">{status?.frames_skipped ?? 0}</Row>
            <Row label="Persons detected">{status?.persons_detected ?? 0}</Row>
            <Row label="Registered visitors">{visitorList?.total ?? "—"}</Row>
            <Row label="New / Returning (session)">
              {status ? `${status.new_visitors} / ${status.returning_visitors}` : "—"}
            </Row>
            {status?.last_error && <Row label="Last error"><span className="text-danger">{status.last_error}</span></Row>}
          </dl>
        </Card>
      </div>

      <Card>
        <div className="mb-3 flex items-center justify-between">
          <CardTitle>Latest Snapshot</CardTitle>
          <Button variant="ghost" onClick={() => setSnapTick(Date.now())}>
            <RefreshCcw className="h-4 w-4" /> Refresh
          </Button>
        </div>
        <div className="overflow-hidden rounded-card border border-card/60 bg-black">
          {running ? (
            <div className="relative inline-block w-full">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                ref={imgRef}
                src={`${imageUrl("camera/snapshot")}?t=${snapTick}`}
                alt="Latest camera snapshot"
                className="mx-auto max-h-[480px] object-contain w-full"
              />
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full cursor-crosshair"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              />
            </div>
          ) : (
            <p className="py-16 text-center text-sm text-text-secondary">
              Start the camera to see a snapshot.
            </p>
          )}
        </div>
        <div className="mt-4 flex gap-2">
          <Button variant="success" onClick={saveRoi} disabled={!roi}>
            Save Detection Zone
          </Button>
          <Button variant="danger" onClick={clearRoi} disabled={!savedRoi}>
            Clear Zone
          </Button>
        </div>
      </Card>
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-card/30 pb-1.5">
      <dt className="text-text-secondary">{label}</dt>
      <dd className="font-medium">{children}</dd>
    </div>
  );
}
