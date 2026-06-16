"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import {
  Activity,
  BarChart3,
  Camera,
  ClipboardCheck,
  LayoutDashboard,
  Clapperboard,
  Settings,
  Users,
  UtensilsCrossed,
} from "lucide-react";

import { fetcher } from "@/lib/api";
import type { HealthResponse, ReviewFlag } from "@/lib/types";

const NAV = [
  { href: "/", label: "Live Monitor", icon: LayoutDashboard },
  { href: "/studio", label: "Video Studio", icon: Clapperboard },
  { href: "/visitors", label: "Visitors", icon: Users },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/activity", label: "Activity", icon: Activity },
  { href: "/review", label: "Review Queue", icon: ClipboardCheck, badgeKey: "review" },
  { href: "/camera", label: "Camera", icon: Camera },
  { href: "/settings", label: "Settings", icon: Settings },
];

function Dot({ ok }: { ok: boolean }) {
  return (
    <span className="relative flex h-2 w-2">
      {ok && (
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-60" />
      )}
      <span
        className={`relative inline-flex h-2 w-2 rounded-full ${
          ok ? "bg-success" : "bg-danger"
        }`}
      />
    </span>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const { data: health } = useSWR<HealthResponse>("health", fetcher, {
    refreshInterval: 10000,
  });
  const { data: flags } = useSWR<ReviewFlag[]>("admin/review-queue?limit=99", fetcher, {
    refreshInterval: 20000,
  });
  const reviewCount = flags?.length ?? 0;

  return (
    <aside className="sticky top-0 flex h-screen w-64 flex-col border-r border-white/5 bg-surface/40 backdrop-blur-xl">
      <div className="flex items-center gap-2.5 px-5 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-control bg-gradient-primary shadow-glow">
          <UtensilsCrossed className="h-5 w-5 text-white" />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold">Restaurant Tracker</p>
          <p className="text-[11px] text-text-muted">Visitor Intelligence</p>
        </div>
      </div>

      <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-2">
        {NAV.map(({ href, label, icon: Icon, badgeKey }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`group relative flex items-center gap-3 rounded-control px-3 py-2 text-sm transition-all ${
                active
                  ? "bg-white/5 font-medium text-text-primary"
                  : "text-text-secondary hover:bg-white/5 hover:text-text-primary"
              }`}
            >
              {active && (
                <span className="absolute left-0 top-1/2 h-5 w-1 -translate-y-1/2 rounded-r-full bg-gradient-primary" />
              )}
              <Icon
                className={`h-4 w-4 transition-colors ${
                  active ? "text-primary-bright" : "text-text-muted group-hover:text-text-secondary"
                }`}
              />
              <span className="flex-1">{label}</span>
              {badgeKey === "review" && reviewCount > 0 && (
                <span className="inline-flex h-5 min-w-5 items-center justify-center rounded-full bg-warning/20 px-1.5 text-[11px] font-semibold text-warning">
                  {reviewCount}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      <div className="space-y-2 border-t border-white/5 px-5 py-4 text-xs text-text-secondary">
        <p className="font-semibold uppercase tracking-wide text-text-muted">System Status</p>
        <div className="flex items-center gap-2">
          <Dot ok={!!health?.camera_running} />
          Camera: {health?.camera_running ? "Running" : "Stopped"}
        </div>
        <div className="flex items-center gap-2">
          <Dot ok={!!health?.models_loaded} />
          Models: {health?.models_loaded ? "Loaded" : "Loading"}
        </div>
        <div className="flex items-center gap-2">
          <Dot ok={health?.database === "connected"} />
          DB: {health?.database ?? "—"}
        </div>
      </div>
    </aside>
  );
}
