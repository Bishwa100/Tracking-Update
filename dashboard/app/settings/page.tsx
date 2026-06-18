"use client";

import { useMemo, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { Cpu, Info, Loader2, RotateCcw, Save, SlidersHorizontal, Zap } from "lucide-react";

import { api, fetcher } from "@/lib/api";
import type { AdminSettings, DeviceStatus } from "@/lib/types";
import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorState,
  Input,
  PageHeader,
  Select,
  Skeleton,
  Toggle,
} from "@/components/ui";

// Group runtime-patchable keys into sections for a readable form.
const GROUPS: { title: string; keys: string[] }[] = [
  {
    title: "Recognition Thresholds",
    keys: [
      "RETURNING_FACE_THRESHOLD",
      "NEW_VISITOR_MAX_SIMILARITY",
      "REJECT_SIMILARITY",
      "AMBIGUITY_MARGIN",
      "STRONG_MATCH_THRESHOLD",
    ],
  },
  {
    title: "Visit Sessions",
    keys: ["VISIT_COOLDOWN_MINUTES", "SEATED_COOLDOWN_MINUTES", "MAX_VISIT_DURATION_HOURS"],
  },
  {
    title: "Temporal Consistency",
    keys: [
      "TEMPORAL_WINDOW_SECONDS",
      "TEMPORAL_MAX_PIXEL_DISTANCE",
      "TEMPORAL_MIN_SIMILARITY",
    ],
  },
  {
    title: "Detection & Quality",
    keys: [
      "FACE_CONF_SKIP_BODY",
      "MIN_FACE_DET_SCORE",
      "FACE_QUALITY_CUTOFF",
      "YOLO_PERSON_CONFIDENCE",
    ],
  },
  {
    title: "Preprocessing & Pose",
    keys: [
      "FACE_PREPROCESSING_CLAHE",
      "FACE_PREPROCESSING_GAMMA",
      "CLAHE_CLIP_LIMIT",
      "POSE_AWARE_GALLERY",
    ],
  },
  {
    title: "Mask & Auto-Tuning",
    keys: ["MASK_DETECTION_ENABLED", "MASKED_FACE_THRESHOLD_OFFSET", "AUTO_TUNING_ENABLED"],
  },
  {
    title: "Multi-Angle Identity",
    keys: [
      "GREY_ZONE_POLICY",
      "POSE_CONTINUOUS_SEARCH",
      "ADAPTIVE_VISITOR_THRESHOLDS",
      "IDENTITY_TOP_K",
      "TRACKLET_ENABLED",
      "TRACKLET_WINDOW_SECONDS",
      "TRACKLET_MAX_PIXEL_DISTANCE",
      "TRACKLET_MIN_OBSERVATIONS_NEW",
    ],
  },
  {
    title: "Cross-Camera",
    keys: [
      "CROSS_CAMERA_ENABLED",
      "CROSS_CAMERA_LOOKBACK_SECONDS",
      "CROSS_CAMERA_REVIEW_THRESHOLD",
      "CROSS_CAMERA_AUTO_THRESHOLD",
      "CROSS_CAMERA_AUTO_MERGE_THRESHOLD",
    ],
  },
];

// String-enum settings rendered as a dropdown instead of a number input.
const SELECT_OPTIONS: Record<string, { value: string; label: string }[]> = {
  GREY_ZONE_POLICY: [
    { value: "review", label: "review — hold + audit (safe default)" },
    { value: "tracklet", label: "tracklet — confirm across frames" },
    { value: "register", label: "register — legacy (create new)" },
  ],
};

function prettyLabel(key: string): string {
  return key
    .toLowerCase()
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

const DEVICE_OPTIONS: { value: string; label: string; needsGpu?: boolean }[] = [
  { value: "cpu", label: "CPU" },
  { value: "cuda", label: "GPU", needsGpu: true },
  { value: "auto", label: "Auto" },
];

function DeviceCard() {
  const { mutate } = useSWRConfig();
  const { data, error, isLoading } = useSWR<DeviceStatus>("admin/device", fetcher);
  const [switching, setSwitching] = useState<string | null>(null);

  async function select(value: string) {
    if (switching || data?.requested === value) return;
    setSwitching(value);
    try {
      await api.post("admin/device", { device: value });
      await mutate("admin/device");
    } catch (e) {
      alert(`Device switch failed: ${(e as Error).message}`);
    } finally {
      setSwitching(null);
    }
  }

  if (error) {
    return (
      <Card>
        <CardTitle>Processing Device</CardTitle>
        <ErrorState message="Could not load device status. Is ADMIN_API_KEY configured?" />
      </Card>
    );
  }
  if (isLoading || !data) {
    return <Skeleton className="h-32" />;
  }

  const onGpu = data.current_device === "cuda";
  return (
    <Card>
      <CardTitle
        action={
          <Badge tone={onGpu ? "success" : "neutral"}>
            {onGpu ? (
              <>
                <Zap className="h-3 w-3" /> GPU
              </>
            ) : (
              <>
                <Cpu className="h-3 w-3" /> CPU
              </>
            )}
          </Badge>
        }
      >
        <span className="flex items-center gap-2">
          {onGpu ? <Zap className="h-4 w-4" /> : <Cpu className="h-4 w-4" />} Processing
          Device
        </span>
      </CardTitle>

      <p className="mb-3 text-sm text-text-secondary">
        Choose where detection & recognition run. Switching reloads all models
        in-process (a few seconds) — no restart needed.
      </p>

      <div className="flex flex-wrap gap-2">
        {DEVICE_OPTIONS.map((opt) => {
          const active = data.requested === opt.value;
          const disabled =
            (opt.needsGpu && !data.cuda_available) || switching !== null;
          return (
            <Button
              key={opt.value}
              variant={active ? "primary" : "ghost"}
              size="sm"
              disabled={disabled}
              onClick={() => select(opt.value)}
            >
              {switching === opt.value && <Loader2 className="h-3 w-3 animate-spin" />}
              {opt.label}
            </Button>
          );
        })}
        {switching && (
          <span className="flex items-center gap-1 text-xs text-text-muted">
            <Loader2 className="h-3 w-3 animate-spin" /> Reloading models…
          </span>
        )}
      </div>

      <div className="mt-3 space-y-1 text-xs text-text-muted">
        {data.cuda_available ? (
          <p>
            GPU detected:{" "}
            <span className="text-text-secondary">{data.gpu_name ?? "CUDA device"}</span>
            {data.gpu_memory_mb
              ? ` · ${(data.gpu_memory_mb / 1024).toFixed(0)} GB`
              : ""}
            {onGpu && data.gpu_memory_used_mb != null
              ? ` · ${data.gpu_memory_used_mb} MB in use`
              : ""}
          </p>
        ) : (
          <p>
            No usable GPU in this environment. Install a CUDA torch build +
            <code className="px-1 text-primary">onnxruntime-gpu</code> to enable the
            GPU option.
          </p>
        )}
        <p>
          Active device:{" "}
          <span className="text-text-secondary">{data.current_device}</span>
          {data.requested === "auto" ? " (auto)" : ""}
        </p>
      </div>
    </Card>
  );
}

export default function SettingsPage() {
  const { mutate } = useSWRConfig();
  const { data, error, isLoading } = useSWR<AdminSettings>("admin/settings", fetcher);

  // Local pending edits keyed by setting name.
  const [edits, setEdits] = useState<AdminSettings>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const merged = useMemo(() => ({ ...(data ?? {}), ...edits }), [data, edits]);
  const dirtyKeys = Object.keys(edits).filter((k) => edits[k] !== data?.[k]);
  const dirty = dirtyKeys.length > 0;

  function setVal(key: string, value: number | boolean | string) {
    setEdits((e) => ({ ...e, [key]: value }));
    setSaved(false);
  }

  async function save() {
    if (!dirty) return;
    setSaving(true);
    try {
      const updates: AdminSettings = {};
      for (const k of dirtyKeys) updates[k] = merged[k];
      await api.patch("admin/settings", { updates });
      await mutate("admin/settings");
      setEdits({});
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (e) {
      alert(`Save failed: ${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Settings"
        subtitle="Tune recognition live — changes apply instantly, no restart."
        action={
          <div className="flex items-center gap-2">
            {dirty && (
              <Button variant="ghost" size="sm" onClick={() => setEdits({})}>
                <RotateCcw className="h-4 w-4" /> Discard
              </Button>
            )}
            <Button onClick={save} disabled={!dirty || saving}>
              <Save className="h-4 w-4" />
              {saving ? "Saving…" : saved ? "Saved ✓" : `Save${dirty ? ` (${dirtyKeys.length})` : ""}`}
            </Button>
          </div>
        }
      />

      <div className="flex items-start gap-2 rounded-card border border-primary/25 bg-primary/10 p-4 text-sm text-text-secondary">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
        <p>
          These thresholds are applied in-process immediately and persisted to the{" "}
          <code className="text-primary">runtime_settings</code> table. Calibrate on your
          own camera footage — see the{" "}
          <span className="text-text-primary">Detection Quality</span> chart on Analytics.
        </p>
      </div>

      <DeviceCard />

      {error ? (
        <ErrorState message="Could not load admin settings. Is ADMIN_API_KEY configured?" />
      ) : isLoading ? (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-56" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          {GROUPS.map((group) => {
            const present = group.keys.filter((k) => k in merged);
            if (present.length === 0) return null;
            return (
              <Card key={group.title}>
                <CardTitle
                  action={
                    present.some((k) => dirtyKeys.includes(k)) ? (
                      <Badge tone="warning">edited</Badge>
                    ) : undefined
                  }
                >
                  {group.title}
                </CardTitle>
                <div className="space-y-3">
                  {present.map((key) => {
                    const val = merged[key];
                    const isBool = typeof val === "boolean";
                    const isString = typeof val === "string";
                    const selectOpts = SELECT_OPTIONS[key];
                    const isDirty = dirtyKeys.includes(key);
                    return (
                      <div
                        key={key}
                        className={`flex items-center justify-between gap-4 rounded-control px-1 py-1.5 ${
                          isDirty ? "bg-warning/5" : ""
                        }`}
                      >
                        <label className="text-sm text-text-secondary">
                          {prettyLabel(key)}
                        </label>
                        {isBool ? (
                          <Toggle
                            checked={val as boolean}
                            onChange={(v) => setVal(key, v)}
                          />
                        ) : isString && selectOpts ? (
                          <div className="w-56">
                            <Select
                              value={val as string}
                              options={selectOpts}
                              onChange={(v) => setVal(key, v)}
                            />
                          </div>
                        ) : (
                          <div className="w-28">
                            <Input
                              type="number"
                              step={0.01}
                              value={val as number}
                              onChange={(v) => setVal(key, Number(v))}
                            />
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </Card>
            );
          })}
        </div>
      )}

      <Card>
        <CardTitle>
          <span className="flex items-center gap-2">
            <SlidersHorizontal className="h-4 w-4" /> All Runtime Values
          </span>
        </CardTitle>
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs md:grid-cols-3">
          {Object.entries(merged)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, v]) => (
              <div
                key={k}
                className="flex items-center justify-between border-b border-white/5 py-1.5"
              >
                <span className="truncate text-text-muted">{k}</span>
                <span className="font-medium text-text-secondary">{String(v)}</span>
              </div>
            ))}
        </div>
      </Card>
    </div>
  );
}
