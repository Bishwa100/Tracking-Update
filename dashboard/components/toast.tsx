"use client";

import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  CheckCircle2,
  XCircle,
  Info,
  AlertTriangle,
  X,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type ToastType = "success" | "error" | "info" | "warning";

interface ToastItem {
  id: string;
  type: ToastType;
  message: string;
  /** Tracks whether the toast is exiting (for the slide-out animation). */
  exiting: boolean;
}

interface ToastAPI {
  success: (message: string) => void;
  error: (message: string) => void;
  info: (message: string) => void;
  warning: (message: string) => void;
}

/* ------------------------------------------------------------------ */
/*  Design tokens per toast type                                       */
/* ------------------------------------------------------------------ */

const iconMap: Record<ToastType, typeof CheckCircle2> = {
  success: CheckCircle2,
  error: XCircle,
  info: Info,
  warning: AlertTriangle,
};

const colorClasses: Record<
  ToastType,
  { badge: string; progress: string }
> = {
  success: {
    badge: "bg-success/15 text-success border-success/25",
    progress: "bg-success",
  },
  error: {
    badge: "bg-danger/15 text-danger border-danger/25",
    progress: "bg-danger",
  },
  info: {
    badge: "bg-primary/15 text-primary-bright border-primary/25",
    progress: "bg-primary-bright",
  },
  warning: {
    badge: "bg-warning/15 text-warning border-warning/25",
    progress: "bg-warning",
  },
};

/* Auto-dismiss duration in ms. */
const DISMISS_MS = 4000;
/* Duration of the exit animation in ms — must match the CSS class. */
const EXIT_ANIMATION_MS = 300;
/* Maximum number of visible toasts. */
const MAX_VISIBLE = 5;

/* ------------------------------------------------------------------ */
/*  Context                                                            */
/* ------------------------------------------------------------------ */

const ToastContext = createContext<ToastAPI | null>(null);

/**
 * Hook to access the toast API.
 * Must be used inside a `<ToastProvider>`.
 */
export function useToast(): ToastAPI {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within a <ToastProvider>");
  }
  return ctx;
}

/* ------------------------------------------------------------------ */
/*  Single Toast                                                       */
/* ------------------------------------------------------------------ */

function Toast({
  item,
  onClose,
}: {
  item: ToastItem;
  onClose: (id: string) => void;
}) {
  const Icon = iconMap[item.type];
  const colors = colorClasses[item.type];

  return (
    <div
      role="alert"
      className={`pointer-events-auto relative flex w-full max-w-[400px] items-start gap-3 overflow-hidden rounded-card border border-white/10 bg-surface/90 p-4 shadow-card backdrop-blur-xl ${
        item.exiting ? "animate-toast-out" : "animate-toast-in"
      }`}
    >
      {/* Icon badge */}
      <span
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full border ${colors.badge}`}
      >
        <Icon size={14} />
      </span>

      {/* Message */}
      <p className="flex-1 pt-0.5 text-sm leading-snug text-text-primary">
        {item.message}
      </p>

      {/* Close button */}
      <button
        type="button"
        onClick={() => onClose(item.id)}
        className="shrink-0 rounded-control p-1 text-text-muted transition hover:bg-white/10 hover:text-text-primary"
        aria-label="Close notification"
      >
        <X size={14} />
      </button>

      {/* Progress bar — only runs while NOT exiting. */}
      {!item.exiting && (
        <span
          className={`absolute bottom-0 left-0 h-[2px] ${colors.progress} animate-toast-progress`}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Provider                                                           */
/* ------------------------------------------------------------------ */

let _idCounter = 0;

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  /* Refs to track dismiss timers so we can clean them up. */
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(
    new Map(),
  );

  /* ---- helpers --------------------------------------------------- */

  const removeToast = useCallback((id: string) => {
    /* Clear any pending timer. */
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const startExit = useCallback(
    (id: string) => {
      setToasts((prev) =>
        prev.map((t) => (t.id === id ? { ...t, exiting: true } : t)),
      );
      /* After exit animation finishes, fully remove the toast. */
      setTimeout(() => removeToast(id), EXIT_ANIMATION_MS);
    },
    [removeToast],
  );

  const addToast = useCallback(
    (type: ToastType, message: string) => {
      const id = `toast-${++_idCounter}`;
      const item: ToastItem = { id, type, message, exiting: false };

      setToasts((prev) => {
        const next = [...prev, item];
        /* Evict the oldest toast(s) when we exceed MAX_VISIBLE. */
        while (next.length > MAX_VISIBLE) {
          const oldest = next.shift();
          if (oldest) {
            const t = timersRef.current.get(oldest.id);
            if (t) {
              clearTimeout(t);
              timersRef.current.delete(oldest.id);
            }
          }
        }
        return next;
      });

      /* Schedule auto-dismiss. */
      const timer = setTimeout(() => startExit(id), DISMISS_MS);
      timersRef.current.set(id, timer);
    },
    [startExit],
  );

  /* ---- stable API object ----------------------------------------- */

  const api = useRef<ToastAPI>({
    success: (msg) => addToast("success", msg),
    error: (msg) => addToast("error", msg),
    info: (msg) => addToast("info", msg),
    warning: (msg) => addToast("warning", msg),
  });

  /* Keep the ref callbacks up-to-date without breaking identity. */
  useEffect(() => {
    api.current.success = (msg) => addToast("success", msg);
    api.current.error = (msg) => addToast("error", msg);
    api.current.info = (msg) => addToast("info", msg);
    api.current.warning = (msg) => addToast("warning", msg);
  }, [addToast]);

  /* ---- cleanup on unmount ---------------------------------------- */

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      timers.forEach((t) => clearTimeout(t));
      timers.clear();
    };
  }, []);

  return (
    <ToastContext.Provider value={api.current}>
      {children}

      {/* Toast container — bottom-right, stacked vertically. */}
      <div
        aria-live="polite"
        className="pointer-events-none fixed bottom-6 right-6 z-50 flex flex-col items-end gap-3"
      >
        {toasts.map((t) => (
          <Toast key={t.id} item={t} onClose={startExit} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}
