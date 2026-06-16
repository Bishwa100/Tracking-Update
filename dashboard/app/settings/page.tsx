"use client";

import { useMemo, useState } from "react";
import useSWR, { useSWRConfig } from "swr";
import { Info, RotateCcw, Save, SlidersHorizontal } from "lucide-react";

import { api, fetcher } from "@/lib/api";
import type { AdminSettings } from "@/lib/types";
import {
  Badge,
  Button,
  Card,
  CardTitle,
  ErrorState,
  Input,
  PageHeader,
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
];

function prettyLabel(key: string): string {
  return key
    .toLowerCase()
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
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

  function setVal(key: string, value: number | boolean) {
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
