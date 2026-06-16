"use client";

import { useState } from "react";
import Link from "next/link";
import useSWR, { useSWRConfig } from "swr";
import {
  AlertTriangle,
  Check,
  Copy,
  ShieldAlert,
  UserX,
} from "lucide-react";

import { api, fetcher } from "@/lib/api";
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

export default function ReviewQueuePage() {
  const { mutate } = useSWRConfig();
  const { data, error, isLoading } = useSWR<ReviewFlag[]>(
    "admin/review-queue?limit=99",
    fetcher,
    { refreshInterval: 15000 },
  );
  const [resolving, setResolving] = useState<string | null>(null);

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

  const byType = flags.reduce<Record<string, number>>((acc, f) => {
    acc[f.flag_type] = (acc[f.flag_type] || 0) + 1;
    return acc;
  }, {});

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
                      <Link
                        href={`/visitors/${flag.visitor_id}`}
                        className="inline-block text-xs text-primary hover:underline"
                      >
                        View visitor {flag.visitor_id.slice(0, 8)} →
                      </Link>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => resolve(flag.id)}
                    disabled={resolving === flag.id}
                  >
                    <Check className="h-4 w-4" />
                    {resolving === flag.id ? "…" : "Resolve"}
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
