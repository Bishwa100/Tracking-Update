"use client";

import { useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import useSWR, { useSWRConfig } from "swr";
import { ArrowLeft, GitMerge, Save, ShieldCheck, ShieldX, Trash2, UserCog } from "lucide-react";

import { api, fetcher } from "@/lib/api";
import type { VisitorDetail, VisitListResponse } from "@/lib/types";
import { formatDateTime, formatDuration, relativeTime } from "@/lib/format";
import { VisitorAvatar } from "@/components/visitor-table";
import { MonthlyBar } from "@/components/charts";
import { Badge, Button, Card, CardTitle, ErrorState, Spinner } from "@/components/ui";

const CONSENT_TONE: Record<string, "success" | "neutral" | "danger"> = {
  explicit: "success",
  implicit: "neutral",
  opted_out: "danger",
};
const CONSENT_LABEL: Record<string, string> = {
  explicit: "Explicit Consent",
  implicit: "Implicit Consent",
  opted_out: "Opted Out",
};

interface GalleryInsights {
  gallery_size: number;
  pose_coverage: Record<string, number>;
  camera_coverage: Record<string, number>;
  adaptive_thresholds: {
    expected_match_similarity: number | null;
    match_similarity_std: number | null;
    personal_returning_threshold: number | null;
    personal_new_threshold: number | null;
  };
  merges: {
    id: string;
    source_visitor_id: string | null;
    reason: string | null;
    similarity: number | null;
    merged_by: string | null;
    created_at: string | null;
  }[];
}

function monthlyBuckets(visits: { entered_at: string }[]) {
  const counts = new Map<string, number>();
  for (const v of visits) {
    const d = new Date(v.entered_at);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, value]) => {
      const [, m] = key.split("-");
      const label = new Date(2000, Number(m) - 1).toLocaleString(undefined, { month: "short" });
      return { label, value };
    });
}

export default function VisitorProfilePage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { mutate } = useSWRConfig();

  const { data: visitor, error } = useSWR<VisitorDetail>(`visitors/${id}`, fetcher);
  const { data: visitsData } = useSWR<VisitListResponse>(
    `visitors/${id}/visits?limit=200`,
    fetcher,
  );
  const { data: insights } = useSWR<GalleryInsights>(
    `visitors/${id}/gallery-insights`,
    fetcher,
  );

  const [name, setName] = useState<string | null>(null);
  const [notes, setNotes] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [mergeId, setMergeId] = useState("");
  const [busyMsg, setBusyMsg] = useState<string | null>(null);

  if (error) return <ErrorState message="Visitor not found." />;
  if (!visitor) return <Spinner />;

  const displayName = name ?? visitor.name ?? "";
  const displayNotes = notes ?? visitor.notes ?? "";
  const visits = visitsData?.visits ?? [];
  const durations = visits.filter((v) => v.duration_minutes != null);
  const avgDuration = durations.length
    ? Math.round(durations.reduce((s, v) => s + (v.duration_minutes || 0), 0) / durations.length)
    : null;

  async function save() {
    setSaving(true);
    try {
      await api.put(`visitors/${id}`, { name: displayName || null, notes: displayNotes || null });
      await mutate(`visitors/${id}`);
    } finally {
      setSaving(false);
    }
  }

  async function toggleStaff() {
    await api.post(`admin/visitors/${id}/mark-staff`, { is_staff: !visitor!.is_staff });
    await mutate(`visitors/${id}`);
  }

  async function remove() {
    if (!confirm("Soft-delete this visitor? They will be excluded from listings.")) return;
    await api.del(`visitors/${id}`);
    router.push("/visitors");
  }

  async function merge() {
    if (!mergeId.trim()) return;
    if (!confirm(`Merge this visitor INTO ${mergeId}? This cannot be undone.`)) return;
    setBusyMsg("Merging…");
    try {
      await api.post(`admin/visitors/${id}/merge`, { target_visitor_id: mergeId.trim() });
      router.push(`/visitors/${mergeId.trim()}`);
    } catch (e) {
      setBusyMsg(`Merge failed: ${(e as Error).message}`);
    }
  }

  async function setConsent(status: string) {
    if (status === "opted_out" && !confirm("Opt this visitor out? They will no longer be recognised and their embeddings will be purged.")) {
      return;
    }
    await api.post(`visitors/${id}/consent`, { consent_status: status, method: "staff" });
    await mutate(`visitors/${id}`);
  }

  return (
    <div className="space-y-6">
      <Link href="/visitors" className="inline-flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary">
        <ArrowLeft className="h-4 w-4" /> Back to Directory
      </Link>

      {/* Identity */}
      <Card>
        <div className="flex flex-col gap-5 md:flex-row">
          <VisitorAvatar visitor={visitor} size={96} />
          <div className="flex-1 space-y-3">
            <div className="flex items-center gap-2">
              <input
                value={displayName}
                onChange={(e) => setName(e.target.value)}
                placeholder={`Visitor ${visitor.id.slice(0, 8)}`}
                className="rounded-control border border-card/60 bg-bg px-3 py-1.5 text-lg font-semibold outline-none focus:border-primary"
              />
              {visitor.is_staff && <Badge tone="accent">Staff</Badge>}
              {visitor.consent_status && (
                <Badge tone={CONSENT_TONE[visitor.consent_status] ?? "neutral"}>
                  {visitor.consent_status === "opted_out" ? (
                    <ShieldX className="h-3 w-3" />
                  ) : (
                    <ShieldCheck className="h-3 w-3" />
                  )}
                  {CONSENT_LABEL[visitor.consent_status] ?? visitor.consent_status}
                </Badge>
              )}
            </div>
            <textarea
              value={displayNotes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Notes…"
              rows={2}
              className="w-full rounded-control border border-card/60 bg-bg px-3 py-2 text-sm outline-none focus:border-primary"
            />
            <div className="flex flex-wrap gap-2">
              <Button onClick={save} disabled={saving}>
                <Save className="h-4 w-4" /> {saving ? "Saving…" : "Save"}
              </Button>
              <Button variant="ghost" onClick={toggleStaff}>
                {visitor.is_staff ? "Unmark staff" : "Mark as staff"}
              </Button>
            </div>
          </div>
        </div>
      </Card>

      {/* Stats */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <Card><p className="text-xs text-text-secondary">Total Visits</p><p className="mt-1 text-2xl font-semibold">{visitor.visit_count}</p></Card>
        <Card><p className="text-xs text-text-secondary">Last Visit</p><p className="mt-1 text-2xl font-semibold">{relativeTime(visitor.last_seen_at)}</p></Card>
        <Card><p className="text-xs text-text-secondary">First Visit</p><p className="mt-1 text-2xl font-semibold">{relativeTime(visitor.first_seen_at)}</p></Card>
        <Card><p className="text-xs text-text-secondary">Avg Duration</p><p className="mt-1 text-2xl font-semibold">{formatDuration(avgDuration)}</p></Card>
      </div>

      {/* Frequency chart */}
      <Card>
        <CardTitle>Visit Frequency (by month)</CardTitle>
        {visits.length ? (
          <MonthlyBar data={monthlyBuckets(visits)} />
        ) : (
          <p className="py-6 text-center text-sm text-text-secondary">No visits recorded.</p>
        )}
      </Card>

      {/* Gallery & matching (Phase 3/4) */}
      {insights && (
        <Card>
          <CardTitle>
            <span className="flex items-center gap-2">
              <UserCog className="h-4 w-4" /> Gallery &amp; Matching
            </span>
          </CardTitle>
          <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
            <div>
              <p className="mb-2 text-xs uppercase tracking-wide text-text-secondary">
                Pose coverage ({insights.gallery_size} faces)
              </p>
              <div className="flex flex-wrap gap-1.5">
                {Object.keys(insights.pose_coverage).length === 0 ? (
                  <span className="text-sm text-text-muted">—</span>
                ) : (
                  Object.entries(insights.pose_coverage).map(([bin, n]) => (
                    <Badge key={bin} tone="neutral">
                      {bin}: {n}
                    </Badge>
                  ))
                )}
              </div>
            </div>
            <div>
              <p className="mb-2 text-xs uppercase tracking-wide text-text-secondary">
                Seen on cameras
              </p>
              <div className="flex flex-wrap gap-1.5">
                {Object.keys(insights.camera_coverage).length === 0 ? (
                  <span className="text-sm text-text-muted">—</span>
                ) : (
                  Object.entries(insights.camera_coverage).map(([cam, n]) => (
                    <Badge key={cam} tone="accent">
                      {cam}: {n}
                    </Badge>
                  ))
                )}
              </div>
            </div>
            <div>
              <p className="mb-2 text-xs uppercase tracking-wide text-text-secondary">
                Adaptive threshold
              </p>
              <dl className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <dt className="text-text-secondary">Returning</dt>
                  <dd className="font-medium">
                    {insights.adaptive_thresholds.personal_returning_threshold?.toFixed(3) ?? "global"}
                  </dd>
                </div>
                <div className="flex justify-between">
                  <dt className="text-text-secondary">Mean ± std</dt>
                  <dd className="font-medium">
                    {insights.adaptive_thresholds.expected_match_similarity != null
                      ? `${insights.adaptive_thresholds.expected_match_similarity.toFixed(2)} ± ${(insights.adaptive_thresholds.match_similarity_std ?? 0).toFixed(2)}`
                      : "—"}
                  </dd>
                </div>
              </dl>
            </div>
          </div>
          {insights.merges.length > 0 && (
            <div className="mt-4 border-t border-card/40 pt-3">
              <p className="mb-2 text-xs uppercase tracking-wide text-text-secondary">
                Merge history
              </p>
              <ul className="space-y-1 text-xs text-text-secondary">
                {insights.merges.map((m) => (
                  <li key={m.id} className="flex items-center gap-2">
                    <GitMerge className="h-3 w-3 text-text-muted" />
                    <span>{m.reason ?? "merge"}</span>
                    {m.similarity != null && (
                      <span className="text-text-muted">sim {m.similarity.toFixed(3)}</span>
                    )}
                    {m.merged_by && <Badge tone="neutral">{m.merged_by}</Badge>}
                    {m.created_at && (
                      <span className="ml-auto text-text-muted">{relativeTime(m.created_at)}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>
      )}

      {/* History */}
      <Card>
        <CardTitle>Visit History</CardTitle>
        <ul className="divide-y divide-card/40 text-sm">
          {visits.map((v, i) => (
            <li key={v.id} className="flex items-center justify-between py-2.5">
              <div className="flex items-center gap-3">
                <span className="w-10 text-text-secondary">#{visits.length - i}</span>
                <span>{formatDateTime(v.entered_at)}</span>
                <span className="text-text-secondary">→ {v.is_active ? "still here" : formatDateTime(v.left_at)}</span>
              </div>
              <div className="flex items-center gap-3">
                {v.is_active && <Badge tone="success">Active</Badge>}
                <span className="text-text-secondary">{v.detection_count} det.</span>
                <span className="font-medium">{formatDuration(v.duration_minutes)}</span>
              </div>
            </li>
          ))}
          {visits.length === 0 && (
            <li className="py-4 text-center text-text-secondary">No visits.</li>
          )}
        </ul>
      </Card>

      {/* Consent & privacy */}
      <Card>
        <CardTitle>
          <span className="flex items-center gap-2">
            <UserCog className="h-4 w-4" /> Consent &amp; Privacy
          </span>
        </CardTitle>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm text-text-secondary">Current status:</span>
          <Badge tone={CONSENT_TONE[visitor.consent_status ?? "implicit"] ?? "neutral"}>
            {CONSENT_LABEL[visitor.consent_status ?? "implicit"] ?? "Implicit"}
          </Badge>
          {visitor.opted_out_at && (
            <span className="text-xs text-text-muted">
              opted out {relativeTime(visitor.opted_out_at)}
            </span>
          )}
          <div className="ml-auto flex gap-2">
            <Button variant="ghost" size="sm" onClick={() => setConsent("explicit")}>
              <ShieldCheck className="h-4 w-4" /> Mark Consented
            </Button>
            <Button variant="danger" size="sm" onClick={() => setConsent("opted_out")}>
              <ShieldX className="h-4 w-4" /> Opt Out
            </Button>
          </div>
        </div>
      </Card>

      {/* Danger zone */}
      <Card>
        <CardTitle>Admin</CardTitle>
        <div className="flex flex-wrap items-end gap-3">
          <div className="flex items-end gap-2">
            <label className="flex flex-col gap-1 text-xs text-text-secondary">
              Merge into visitor ID
              <input
                value={mergeId}
                onChange={(e) => setMergeId(e.target.value)}
                placeholder="target uuid"
                className="w-72 rounded-control border border-card/60 bg-bg px-3 py-2 text-sm outline-none focus:border-primary"
              />
            </label>
            <Button variant="ghost" onClick={merge}>
              <GitMerge className="h-4 w-4" /> Merge
            </Button>
          </div>
          <Button variant="danger" onClick={remove}>
            <Trash2 className="h-4 w-4" /> Delete
          </Button>
        </div>
        {busyMsg && <p className="mt-2 text-sm text-text-secondary">{busyMsg}</p>}
      </Card>
    </div>
  );
}
