import { ReactNode } from "react";

type Tone = "primary" | "success" | "warning" | "accent" | "danger";

const toneTile: Record<Tone, string> = {
  primary: "bg-primary/15 text-primary-bright",
  success: "bg-success/15 text-success",
  warning: "bg-warning/15 text-warning",
  accent: "bg-accent/15 text-accent-bright",
  danger: "bg-danger/15 text-danger",
};

export function StatCard({
  label,
  value,
  hint,
  icon,
  tone = "primary",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  icon?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className="glass glass-hover group animate-fade-in rounded-card p-5 shadow-card">
      <div className="flex items-start justify-between">
        <div className="min-w-0">
          <p className="text-xs font-medium uppercase tracking-wide text-text-secondary">
            {label}
          </p>
          <p className="mt-2 text-3xl font-semibold tracking-tight text-text-primary">
            {value}
          </p>
          {hint && <p className="mt-1 text-xs text-text-muted">{hint}</p>}
        </div>
        {icon && (
          <div
            className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-control transition-transform group-hover:scale-105 ${toneTile[tone]}`}
          >
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
