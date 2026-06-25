"use client";

import { usePathname } from "next/navigation";
import { ReactNode } from "react";

/**
 * Lightweight page transition wrapper.
 * Re-keys on pathname change to re-trigger the existing
 * `animate-fade-in` CSS animation (opacity 0→1, translateY(6px)→0, 0.3s ease-out).
 * No external animation libraries required.
 */
export function PageTransition({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  // Re-trigger animation by changing key when pathname changes
  return (
    <div key={pathname} className="animate-fade-in">
      {children}
    </div>
  );
}
