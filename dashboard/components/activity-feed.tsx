import Link from "next/link";
import { AlertTriangle, UserPlus, UserCheck, Clock, PauseCircle, Camera } from "lucide-react";

import type { ActivityEvent } from "@/lib/types";
import { formatTime, shortId } from "@/lib/format";
import { Badge } from "@/components/ui";

function EventIcon({ ev }: { ev: ActivityEvent }) {
  if (ev.is_ambiguous) return <AlertTriangle className="h-4 w-4 text-warning" />;
  if (ev.match_source === "grey_zone")
    return <PauseCircle className="h-4 w-4 text-text-muted" />;
  if (ev.is_new_visitor) return <UserPlus className="h-4 w-4 text-success" />;
  if (ev.match_source === "cross_camera")
    return <Camera className="h-4 w-4 text-accent-bright" />;
  if (ev.match_source === "temporal" || ev.match_source === "tracklet")
    return <Clock className="h-4 w-4 text-primary" />;
  return <UserCheck className="h-4 w-4 text-primary" />;
}

function label(ev: ActivityEvent) {
  if (ev.is_ambiguous) return <Badge tone="warning">Ambiguous</Badge>;
  if (ev.match_source === "grey_zone") return <Badge tone="neutral">Held</Badge>;
  if (ev.is_new_visitor) return <Badge tone="success">New</Badge>;
  if (ev.match_source === "cross_camera")
    return <Badge tone="accent">Cross-camera</Badge>;
  if (ev.match_source === "temporal" || ev.match_source === "tracklet")
    return <Badge tone="primary">Re-acquired</Badge>;
  return <Badge tone="primary">Recognized</Badge>;
}

export function ActivityFeed({
  events,
  compact = false,
}: {
  events: ActivityEvent[];
  compact?: boolean;
}) {
  if (events.length === 0) {
    return <p className="py-6 text-center text-sm text-text-secondary">No activity yet.</p>;
  }
  return (
    <ul className="divide-y divide-card/40">
      {events.map((ev) => {
        const sim = ev.face_similarity ?? ev.body_similarity;
        const name = ev.visitor_name || (ev.visitor_id ? `Visitor ${shortId(ev.visitor_id)}` : "Unknown");
        const inner = (
          <div className="flex items-center gap-3 py-2.5">
            <EventIcon ev={ev} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="truncate text-sm font-medium text-text-primary">{name}</span>
                {label(ev)}
              </div>
              {!compact && (
                <p className="text-xs text-text-secondary">
                  {ev.camera_id ? `Camera ${ev.camera_id} · ` : ""}
                  {ev.match_source && ev.match_source !== "none"
                    ? `via ${ev.match_source}`
                    : ev.is_ambiguous
                    ? "skipped (margin too narrow)"
                    : ""}
                </p>
              )}
            </div>
            <div className="text-right">
              {sim != null && (
                <p className="text-xs font-medium text-text-primary">{sim.toFixed(2)}</p>
              )}
              <p className="text-xs text-text-secondary">{formatTime(ev.detected_at)}</p>
            </div>
          </div>
        );
        return (
          <li key={ev.id}>
            {ev.visitor_id ? (
              <Link href={`/visitors/${ev.visitor_id}`} className="block hover:bg-card/20">
                {inner}
              </Link>
            ) : (
              inner
            )}
          </li>
        );
      })}
    </ul>
  );
}
