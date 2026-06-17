"use client";

import { useState } from "react";
import Link from "next/link";
import useSWR, { useSWRConfig } from "swr";
import {
  AlertTriangle,
  Check,
  Combine,
  Copy,
  ShieldAlert,
  Sparkles,
  UserX,
} from "lucide-react";

import { api, fetcher, imageUrl } from "@/lib/api";
import type { ReviewFlag } from "@/lib/types";
import { relativeTime } from "@/lib/format";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  ErrorState,
  PageHeader,
  Skeleton,
} from "@/components/ui";

type Tone = "warning" | "danger" | "accent";

const TONE_TILE: Record<Tone, string> = {
  warning: "bg-warning/15 text-warning",
  danger: "bg-danger/15 text-danger",
  accent: "bg-accent/15 text-accent-bright",
};

const FLAG_META: Record<
  string,
  { tone: Tone; icon: typeof AlertTriangle; label: string }
> = {
  new_low_quality: { tone: "warning", icon: AlertTriangle, label: "Low Quality" },
  probable_duplicate: { tone: "accent", icon: Copy, label: "Probable Duplicate" },
  high_ambiguity: { tone: "warning", icon: ShieldAlert, label: "High Ambiguity" },
  opted_out_match: { tone: "danger", icon: UserX, label: "Opted-Out Match" },
};

function ComparePhoto({
  visitorId,
  label,
  name,
}: {
  visitorId: string;
  label: string;
  name?: string | null;
}) {
  return (
    <Link
      href={`/visitors/${visitorId}`}
      className="group flex flex-col items-center gap-1.5"
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imageUrl(`/api/visitors/${visitorId}/thumbnail`)}
        alt={name || label}
        width={112}
        height={112}
        className="h-28 w-28 rounded-control border border-card/60 object-cover transition group-hover:border-primary"
        style={{ width: 112, height: 112 }}
      />
      <span className="text-[10px] uppercase tracking-wide text-text-muted">
        {label}
      </span>
      <span className="max-w-[7rem] truncate text-xs font-medium text-text-secondary group-hover:text-primary">
        {name || `Visitor ${visitorId.slice(0, 8)}`}
      </span>
    </Link>
  );
}

function DuplicateCompare({ flag }: { flag: ReviewFlag }) {
  if (!flag.matched_visitor_id) return null;
  return (
    <div className="mt-3 flex items-center gap-4 rounded-control bg-card/30 p-3">
      <ComparePhoto visitorId={flag.visitor_id} label="New" />
      <div className="flex flex-col items-center gap-1 text-center">
        <Copy className="h-4 w-4 text-accent-bright" />
        {flag.similarity != null && (
          <span className="text-xs font-semibold text-text-primary">
            {(flag.similarity * 100).toFixed(1)}%
          </span>
        )}
        <span className="text-[10px] uppercase tracking-wide text-text-muted">
          match
        </span>
      </div>
      <ComparePhoto
        visitorId={flag.matched_visitor_id}
        label="Existing"
        name={flag.matched_visitor_name}
      />
    </div>
  );
}

export default function ReviewQueuePage() {
  const { mutate } = useSWRConfig();
  const { data, error, isLoading } = useSWR<ReviewFlag[]>(
    "admin/review-queue?limit=99",
    fetcher,
    { refreshInterval: 15000 },
  );
  const [resolving, setResolving] = useState<string | null>(null);
  const [cleaning, setCleaning] = useState<string | null>(null);
  const [cleanResult, setCleanResult] = useState<Record<string, string>>({});
  const [merging, setMerging] = useState(false);
  const [mergeResult, setMergeResult] = useState<string | null>(null);

  const flags = data ?? [];

  async function resolve(id: string) {
    setResolving(id);
    try {
      await api.post(`admin/review-queue/${id}/resolve`);
      await mutate("admin/review-queue?limit=99");
    } finally {
      setResolving(null);
    }
  }

  async function cleanFaces(flagId: string, visitorId: string) {
    setCleaning(flagId);
    try {
      const res = await api.post<{ removed: number; kept: number }>(
        `admin/visitors/${visitorId}/clean-faces`,
      );
      setCleanResult((prev) => ({
        ...prev,
        [flagId]: res.removed
          ? `Removed ${res.removed} unclear face${res.removed === 1 ? "" : "s"} · ${res.kept} kept`
          : `No unclear faces · ${res.kept} kept`,
      }));
    } catch {
      setCleanResult((prev) => ({ ...prev, [flagId]: "Clean failed" }));
    } finally {
      setCleaning(null);
    }
  }

  async function autoMergeDuplicates() {
    setMerging(true);
    setMergeResult(null);
    try {
      const res = await api.post<{ merged: number; skipped: number }>(
        "admin/review-queue/auto-merge-duplicates",
      );
      setMergeResult(
        res.merged
          ? `Merged ${res.merged} duplicate${res.merged === 1 ? "" : "s"} into a single user${res.skipped ? ` · ${res.skipped} skipped` : ""}`
          : "No duplicates could be merged",
      );
      await mutate("admin/review-queue?limit=99");
    } catch {
      setMergeResult("Auto-merge failed");
    } finally {
      setMerging(false);
    }
  }

  const byType = flags.reduce<Record<string, number>>((acc, f) => {
    acc[f.flag_type] = (acc[f.flag_type] || 0) + 1;
    return acc;
  }, {});

  const duplicateCount = byType["probable_duplicate"] || 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Review Queue"
        subtitle="Flagged registrations and probable duplicates awaiting a decision."
        action={
          <Badge tone={flags.length ? "warning" : "success"}>
            {flags.length} pending
          </Badge>
        }
      />

      {Object.keys(byType).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(byType).map(([type, n]) => {
            const meta = FLAG_META[type] ?? {
              tone: "warning" as const,
              icon: AlertTriangle,
              label: type,
            };
            const Icon = meta.icon;
            return (
              <Badge key={type} tone={meta.tone}>
                <Icon className="h-3 w-3" /> {meta.label}: {n}
              </Badge>
            );
          })}
        </div>
      )}

      {duplicateCount > 0 && (
        <Card className="!p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-control bg-accent/15 text-accent-bright">
                <Combine className="h-5 w-5" />
              </div>
              <div className="space-y-0.5">
                <p className="text-sm font-medium text-text-primary">
                  {duplicateCount} probable duplicate{duplicateCount === 1 ? "" : "s"}
                </p>
                <p className="text-xs text-text-muted">
                  Merge each into the existing visitor it matches, collapsing them into
                  a single user.
                </p>
                {mergeResult && (
                  <p className="text-xs text-text-secondary">{mergeResult}</p>
                )}
              </div>
            </div>
            <Button
              variant="primary"
              size="sm"
              onClick={autoMergeDuplicates}
              disabled={merging}
            >
              <Combine className="h-4 w-4" />
              {merging ? "Merging…" : "Auto-merge duplicates"}
            </Button>
          </div>
        </Card>
      )}

      {error ? (
        <ErrorState message="Could not load the review queue. Is ADMIN_API_KEY configured?" />
      ) : isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
      ) : flags.length === 0 ? (
        <EmptyState
          icon={<Check className="h-8 w-8" />}
          message="All clear — no items need review."
        />
      ) : (
        <div className="space-y-3">
          {flags.map((flag) => {
            const meta = FLAG_META[flag.flag_type] ?? {
              tone: "warning" as const,
              icon: AlertTriangle,
              label: flag.flag_type,
            };
            const Icon = meta.icon;
            return (
              <Card key={flag.id} className="!p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-3">
                    <div
                      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-control ${TONE_TILE[meta.tone]}`}
                    >
                      <Icon className="h-5 w-5" />
                    </div>
                    <div className="min-w-0 space-y-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={meta.tone}>{meta.label}</Badge>
                        <span className="text-xs text-text-muted">
                          {relativeTime(flag.created_at)}
                        </span>
                      </div>
                      <p className="text-sm text-text-secondary">{flag.detail}</p>
                      {flag.matched_visitor_id && (
                        <p className="text-xs text-text-secondary">
                          Similar to{" "}
                          <Link
                            href={`/visitors/${flag.matched_visitor_id}`}
                            className="text-primary hover:underline"
                          >
                            {flag.matched_visitor_name ||
                              `visitor ${flag.matched_visitor_id.slice(0, 8)}`}
                          </Link>
                          {flag.similarity != null && (
                            <span className="ml-1 text-text-muted">
                              · {(flag.similarity * 100).toFixed(1)}% match
                            </span>
                          )}
                        </p>
                      )}
                      <Link
                        href={`/visitors/${flag.visitor_id}`}
                        className="inline-block text-xs text-primary hover:underline"
                      >
                        View visitor {flag.visitor_id.slice(0, 8)} →
                      </Link>
                      <DuplicateCompare flag={flag} />
                    </div>
                  </div>
                  <div className="flex shrink-0 flex-col items-end gap-1.5">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => cleanFaces(flag.id, flag.visitor_id)}
                      disabled={cleaning === flag.id}
                    >
                      <Sparkles className="h-4 w-4" />
                      {cleaning === flag.id ? "…" : "Auto-clean faces"}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => resolve(flag.id)}
                      disabled={resolving === flag.id}
                    >
                      <Check className="h-4 w-4" />
                      {resolving === flag.id ? "…" : "Resolve"}
                    </Button>
                    {cleanResult[flag.id] && (
                      <span className="max-w-[10rem] text-right text-[10px] text-text-muted">
                        {cleanResult[flag.id]}
                      </span>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
