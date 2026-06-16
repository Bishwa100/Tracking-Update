"use client";

import { useState } from "react";
import useSWR from "swr";
import { DoorOpen, LogIn, RefreshCw, Users } from "lucide-react";

import { fetcher } from "@/lib/api";
import type { AnalyticsSummary, ActivityResponse, CameraStatus } from "@/lib/types";
import { DetectionFeed } from "@/components/detection-feed";
import { ActivityFeed } from "@/components/activity-feed";
import { StatCard } from "@/components/stat-card";
import { Card, CardTitle, PageHeader } from "@/components/ui";

function startOfTodayISO() {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.toISOString();
}

export default function LiveMonitorPage() {
  const [status, setStatus] = useState<CameraStatus | null>(null);
  const since = startOfTodayISO();

  const { data: summary } = useSWR<AnalyticsSummary>(
    `analytics/summary?since=${since}`,
    fetcher,
    { refreshInterval: 15000 },
  );
  const { data: activity } = useSWR<ActivityResponse>("activity?limit=12", fetcher, {
    refreshInterval: 5000,
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Live Monitor"
        subtitle="Real-time feed with on-frame recognition labels."
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <DetectionFeed onStatus={setStatus} />
        </div>

        <div className="grid grid-cols-2 gap-4 lg:grid-cols-1">
          <StatCard
            label="Persons Detected"
            value={status?.persons_detected ?? "—"}
            icon={<Users className="h-6 w-6" />}
            tone="primary"
          />
          <StatCard
            label="Today's Visits"
            value={summary?.total_visits ?? "—"}
            icon={<DoorOpen className="h-6 w-6" />}
            tone="success"
          />
          <StatCard
            label="New Today"
            value={summary?.new_visitors ?? "—"}
            icon={<LogIn className="h-6 w-6" />}
            tone="accent"
          />
          <StatCard
            label="Returning Today"
            value={summary?.returning_visitors ?? "—"}
            icon={<RefreshCw className="h-6 w-6" />}
            tone="warning"
          />
        </div>
      </div>

      <Card>
        <CardTitle>Recent Activity</CardTitle>
        <ActivityFeed events={activity?.events ?? []} />
      </Card>
    </div>
  );
}
