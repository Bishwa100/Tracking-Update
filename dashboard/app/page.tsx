"use client";

import useSWR from "swr";

import { fetcher } from "@/lib/api";
import type { ActivityResponse } from "@/lib/types";
import { DetectionFeed } from "@/components/detection-feed";
import { ActivityFeed } from "@/components/activity-feed";
import { Card, CardTitle, PageHeader } from "@/components/ui";

export default function LiveMonitorPage() {
  const { data: activity } = useSWR<ActivityResponse>("activity?limit=12", fetcher, {
    refreshInterval: 5000,
  });

  return (
    <div className="space-y-6">
      <PageHeader
        title="Live Monitor"
        subtitle="Real-time feed with on-frame recognition labels."
      />

      <DetectionFeed />

      <Card>
        <CardTitle>Recent Activity</CardTitle>
        <ActivityFeed events={activity?.events ?? []} />
      </Card>
    </div>
  );
}
