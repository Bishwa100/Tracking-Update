"use client";

import { ReactNode } from "react";

export function Card({
  children,
  className = "",
  glow = false,
}: {
  children: ReactNode;
  className?: string;
  glow?: boolean;
}) {
  return (
    <div
      className={`glass animate-fade-in rounded-card p-5 shadow-card ${
        glow ? "shadow-glow" : ""
      } ${className}`}
    >
      {children}
    </div>
  );
}

export function CardTitle({
  children,
  action,
}: {
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="mb-4 flex items-center justify-between">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-text-secondary">
        {children}
      </h2>
      {action}
    </div>
  );
}

type BadgeTone = "primary" | "success" | "warning" | "danger" | "accent" | "neutral";

const toneClasses: Record<BadgeTone, string> = {
  primary: "bg-primary/15 text-primary-bright ring-1 ring-inset ring-primary/25",
  success: "bg-success/15 text-success ring-1 ring-inset ring-success/25",
  warning: "bg-warning/15 text-warning ring-1 ring-inset ring-warning/25",
  danger: "bg-danger/15 text-danger ring-1 ring-inset ring-danger/25",
  accent: "bg-accent/15 text-accent-bright ring-1 ring-inset ring-accent/25",
  neutral: "bg-card/40 text-text-secondary ring-1 ring-inset ring-white/5",
};

export function Badge({
  children,
  tone = "neutral",
  className = "",
}: {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${toneClasses[tone]} ${className}`}
    >
      {children}
    </span>
  );
}

type ButtonVariant = "primary" | "danger" | "ghost" | "success" | "accent";

const buttonClasses: Record<ButtonVariant, string> = {
  primary:
    "bg-gradient-primary text-white shadow-glow hover:brightness-110 active:brightness-95",
  accent: "bg-accent text-white hover:bg-accent/90",
  success: "bg-success text-white hover:bg-success/90",
  danger: "bg-danger/90 text-white hover:bg-danger",
  ghost: "bg-white/5 text-text-primary ring-1 ring-inset ring-white/10 hover:bg-white/10",
};

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled = false,
  type = "button",
  className = "",
  size = "md",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: ButtonVariant;
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
  size?: "sm" | "md";
}) {
  const sizeCls = size === "sm" ? "px-3 py-1.5 text-xs" : "px-4 py-2 text-sm";
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex items-center justify-center gap-2 rounded-control font-medium transition-all duration-150 disabled:cursor-not-allowed disabled:opacity-50 ${sizeCls} ${buttonClasses[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export function Input({
  value,
  onChange,
  placeholder,
  type = "text",
  className = "",
  ...rest
}: {
  value: string | number;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  className?: string;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      {...rest}
      className={`w-full rounded-control border border-white/10 bg-bg/60 px-3 py-2 text-sm text-text-primary outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20 ${className}`}
    />
  );
}

export function Select({
  value,
  onChange,
  options,
  className = "",
}: {
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`w-full rounded-control border border-white/10 bg-bg/60 px-3 py-2 text-sm text-text-primary outline-none transition focus:border-primary focus:ring-2 focus:ring-primary/20 ${className}`}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value} className="bg-bg text-text-primary">
          {o.label}
        </option>
      ))}
    </select>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-text-secondary">
      <div className="h-4 w-4 animate-spin rounded-full border-2 border-card border-t-primary" />
      {label || "Loading…"}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} />;
}

export function EmptyState({
  message,
  icon,
}: {
  message: string;
  icon?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-card border border-dashed border-white/10 p-10 text-center text-sm text-text-secondary">
      {icon && <div className="text-text-muted">{icon}</div>}
      {message}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-card border border-danger/30 bg-danger/10 p-4 text-sm text-danger">
      {message}
    </div>
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="inline-flex items-center gap-2"
    >
      <span
        className={`relative h-5 w-9 rounded-full transition-colors ${
          checked ? "bg-primary" : "bg-card/60"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-transform ${
            checked ? "translate-x-4" : "translate-x-0.5"
          }`}
        />
      </span>
      {label && <span className="text-sm text-text-secondary">{label}</span>}
    </button>
  );
}

export function PageHeader({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 text-sm text-text-secondary">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}
