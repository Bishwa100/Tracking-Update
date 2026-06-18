"use client";

import { useState } from "react";
import useSWR from "swr";
import { Flame, ShieldCheck } from "lucide-react";

import { fetcher } from "@/lib/api";
import type {
  ConfidenceWeightedSummary,
  DetectionQuality,
  FrequencyDistribution,
  HourlyBreakdown,
  TopVisitor,
} from "@/lib/types";
import {
  DailyVisitsArea,
  DetectionQualityBar,
  FrequencyBar,
  HourlyStackedBar,
  NewVsReturningDonut,
} from "@/components/charts";
import { StatCard } from "@/components/stat-card";
import { Button, Card, CardTitle, PageHeader } from "@/components/ui";
import { formatDuration } from "@/lib/format";

type RangeKey = "today" | "week" | "month";

interface PipelineQuality {
  total_detections: number;
  grey_zone: number;
  grey_zone_rate: number;
  ambiguous: number;
  ambiguous_rate: number;
  temporal_recoveries: number;
  cross_camera_recoveries: number;
  tracklet_recoveries: number;
  new_registrations: number;
}

function sinceFor(key: RangeKey): string {
  const d = new Date();
  if (key === "today") d.setHours(0, 0, 0, 0);
  else if (key === "week") d.setDate(d.getDate() - 7);
  else d.setDate(d.getDate() - 30);
  return d.toISOString();
}

export default function AnalyticsPage() {
  const [range, setRange] = useState<RangeKey>("month");
  const since = sinceFor(range);

  const { data: summary } = useSWR<ConfidenceWeightedSummary>(
    `analytics/confidence-weighted?since=${since}`,
    fetcher,
  );
  const { data: quality } = useSWR<DetectionQuality>(
    `analytics/detection-quality?since=${since}`,
    fetcher,
  );
  const { data: freq } = useSWR<FrequencyDistribution>("analytics/frequency", fetcher);
  const { data: hourly } = useSWR<HourlyBreakdown>(`analytics/hourly?since=${since}`, fetcher);
  const { data: top } = useSWR<TopVisitor[]>("analytics/top-visitors?limit=5", fetcher);
  const { data: pipeline } = useSWR<PipelineQuality>(
    `analytics/pipeline-quality?since=${since}`,
    fetcher,
  );

  const cw = summary?.confidence_weighted;
  const freqData = freq
    ? Object.entries(freq.distribution).map(([bucket, count]) => ({
        bucket: bucket === "1" ? "1 visit" : `${bucket} visits`,
        count,
      }))
    : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analytics"
        action={
          <div className="flex gap-1.5 rounded-control bg-white/5 p-1">
            {(["today", "week", "month"] as RangeKey[]).map((k) => (
              <Button
                key={k}
                variant={range === k ? "primary" : "ghost"}
                size="sm"
                onClick={() => setRange(k)}
              >
                {k === "today" ? "Today" : k === "week" ? "Week" : "Month"}
              </Button>
            ))}
          </div>
        }
      />

      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        <StatCard label="Unique Visitors" value={summary?.total_unique_visitors ?? "—"} />
        <StatCard
          label="Effective Unique"
          value={cw ? cw.effective_unique.toFixed(0) : "—"}
          hint={cw ? `conf-weighted · avg ${(cw.avg_confidence * 100).toFixed(0)}%` : undefined}
          icon={<ShieldCheck className="h-5 w-5" />}
          tone="accent"
        />
        <StatCard
          label="Return Rate"
          value={summary ? `${Math.round(summary.return_rate * 100)}%` : "—"}
          tone="success"
        />
        <StatCard
          label="Avg Duration"
          value={summary ? formatDuration(Math.round(summary.average_duration_minutes)) : "—"}
          tone="warning"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardTitle>Daily Visits</CardTitle>
          <DailyVisitsArea data={summary?.visits_by_day ?? []} />
        </Card>
        <Card>
          <CardTitle>New vs Returning</CardTitle>
          <NewVsReturningDonut
            newCount={summary?.new_visitors ?? 0}
            returningCount={summary?.returning_visitors ?? 0}
          />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card>
          <CardTitle>Detection Quality</CardTitle>
          <DetectionQualityBar
            high={quality?.bands.high ?? 0}
            medium={quality?.bands.medium ?? 0}
            low={quality?.bands.low ?? 0}
          />
          <p className="mt-2 text-center text-xs text-text-muted">
            {quality ? `${quality.total_detections.toLocaleString()} detections` : ""}
          </p>
        </Card>
        <Card className="lg:col-span-2">
          <CardTitle>Hourly (New + Returning)</CardTitle>
          <HourlyStackedBar data={hourly?.hourly ?? []} />
        </Card>
      </div>

      {pipeline && (
        <Card>
          <CardTitle>Pipeline Health (decision quality)</CardTitle>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <StatCard
              label="Grey-zone held"
              value={`${(pipeline.grey_zone_rate * 100).toFixed(1)}%`}
              hint={`${pipeline.grey_zone.toLocaleString()} of ${pipeline.total_detections.toLocaleString()}`}
              tone="warning"
            />
            <StatCard
              label="Ambiguous"
              value={`${(pipeline.ambiguous_rate * 100).toFixed(1)}%`}
              hint={`${pipeline.ambiguous.toLocaleString()} skipped`}
            />
            <StatCard
              label="Re-acquired"
              value={(
                pipeline.temporal_recoveries +
                pipeline.cross_camera_recoveries +
                pipeline.tracklet_recoveries
              ).toLocaleString()}
              hint={`${pipeline.cross_camera_recoveries} cross-camera`}
              tone="accent"
            />
            <StatCard
              label="New registrations"
              value={pipeline.new_registrations.toLocaleString()}
              tone="success"
            />
          </div>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card>
          <CardTitle>Top Regulars</CardTitle>
          <ol className="space-y-2 text-sm">
            {(top ?? []).map((v, i) => (
              <li key={v.visitor_id} className="flex items-center justify-between">
                <a
                  href={`/visitors/${v.visitor_id}`}
                  className="flex items-center gap-2 hover:text-primary-bright"
                >
                  <span className="text-text-muted">{i + 1}.</span>
                  {v.name || `Visitor ${v.visitor_id.slice(0, 8)}`}
                </a>
                <span className="inline-flex items-center gap-1 font-medium">
                  {v.visit_count}
                  {v.visit_count >= 10 && <Flame className="h-3.5 w-3.5 text-warning" />}
                </span>
              </li>
            ))}
            {(top ?? []).length === 0 && (
              <li className="py-4 text-center text-text-secondary">No data yet.</li>
            )}
          </ol>
        </Card>
        <Card className="lg:col-span-2">
          <CardTitle>Visit Frequency Distribution</CardTitle>
          <FrequencyBar data={freqData} />
        </Card>
      </div>
    </div>
  );
}
