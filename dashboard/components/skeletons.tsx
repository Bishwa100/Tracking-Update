"use client";

/**
 * Enhanced loading skeletons for each page.
 * Each skeleton mirrors the real page layout so the transition from
 * skeleton → content feels seamless. Uses the existing `.skeleton`
 * CSS class and `animate-fade-in` Tailwind animation.
 */

// ─── Dashboard (home page) ──────────────────────────────────────────
export function DashboardSkeleton() {
  // 2 stat cards (grid-cols-2) + video feed placeholder + activity list
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page header skeleton */}
      <div>
        <div className="skeleton h-8 w-48" />
        <div className="skeleton mt-2 h-4 w-72" />
      </div>
      {/* Stat cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="glass rounded-card p-5">
            <div className="skeleton h-3 w-24" />
            <div className="skeleton mt-3 h-8 w-16" />
            <div className="skeleton mt-2 h-3 w-32" />
          </div>
        ))}
      </div>
      {/* Video feed */}
      <div className="skeleton aspect-video w-full rounded-card" />
      {/* Activity feed */}
      <div className="glass rounded-card p-5">
        <div className="skeleton mb-4 h-4 w-32" />
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex items-center gap-3 py-2.5">
            <div className="skeleton h-4 w-4 rounded-full" />
            <div className="flex-1">
              <div className="skeleton h-4 w-32" />
              <div className="skeleton mt-1 h-3 w-48" />
            </div>
            <div className="skeleton h-3 w-16" />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Analytics ───────────────────────────────────────────────────────
export function AnalyticsSkeleton() {
  // 4 stat cards (grid-cols-4) + 2 chart placeholders + table
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div className="skeleton h-8 w-32" />
        <div className="skeleton h-9 w-48 rounded-control" />
      </div>
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="glass rounded-card p-5">
            <div className="skeleton h-3 w-20" />
            <div className="skeleton mt-3 h-8 w-16" />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="glass rounded-card p-5">
          <div className="skeleton mb-4 h-4 w-28" />
          <div className="skeleton h-[260px] w-full rounded-control" />
        </div>
        <div className="glass rounded-card p-5">
          <div className="skeleton mb-4 h-4 w-36" />
          <div className="skeleton h-[260px] w-full rounded-control" />
        </div>
      </div>
    </div>
  );
}

// ─── Visitors ────────────────────────────────────────────────────────
export function VisitorsSkeleton() {
  // Search bar + table with 8 rows
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="skeleton h-8 w-36" />
      <div className="glass rounded-card p-5">
        <div className="flex gap-3">
          <div className="skeleton h-10 flex-1 rounded-control" />
          <div className="skeleton h-10 w-32 rounded-control" />
          <div className="skeleton h-10 w-32 rounded-control" />
        </div>
      </div>
      <div className="glass rounded-card p-5">
        {[...Array(8)].map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-3 border-b border-card/40 py-3 last:border-0"
          >
            <div className="skeleton h-10 w-10 rounded-full" />
            <div className="flex-1">
              <div className="skeleton h-4 w-32" />
            </div>
            <div className="skeleton h-4 w-12" />
            <div className="skeleton h-4 w-20" />
            <div className="skeleton h-4 w-20" />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Activity ────────────────────────────────────────────────────────
export function ActivitySkeleton() {
  // Filter bar + 10 activity rows
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="skeleton h-8 w-44" />
      <div className="glass rounded-card p-5">
        <div className="flex gap-2">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton h-8 w-24 rounded-control" />
          ))}
        </div>
      </div>
      <div className="glass rounded-card p-5">
        {[...Array(10)].map((_, i) => (
          <div key={i} className="flex items-center gap-3 py-2.5">
            <div className="skeleton h-4 w-4 rounded-full" />
            <div className="flex-1">
              <div className="skeleton h-4 w-28" />
              <div className="skeleton mt-1 h-3 w-44" />
            </div>
            <div className="skeleton h-3 w-16" />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Camera ──────────────────────────────────────────────────────────
export function CameraSkeleton() {
  // Camera controls + video feed + status panel
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="skeleton h-8 w-40" />
      <div className="glass rounded-card p-5">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="skeleton h-10 rounded-control" />
          ))}
        </div>
      </div>
      <div className="skeleton aspect-video w-full rounded-card" />
      <div className="glass rounded-card p-5">
        <div className="skeleton mb-4 h-4 w-28" />
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
          {[...Array(8)].map((_, i) => (
            <div key={i}>
              <div className="skeleton h-3 w-20" />
              <div className="skeleton mt-1 h-5 w-16" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Settings ────────────────────────────────────────────────────────
export function SettingsSkeleton() {
  // Settings groups
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="skeleton h-8 w-28" />
      {[...Array(3)].map((_, i) => (
        <div key={i} className="glass rounded-card p-5">
          <div className="skeleton mb-4 h-4 w-36" />
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {[...Array(4)].map((_, j) => (
              <div key={j}>
                <div className="skeleton h-3 w-32" />
                <div className="skeleton mt-2 h-10 w-full rounded-control" />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
