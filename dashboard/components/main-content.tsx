"use client";

import type { ReactNode } from "react";
import { useSidebar } from "@/components/sidebar-context";

/** Main content wrapper that adjusts padding based on sidebar state */
export function MainContent({ children }: { children: ReactNode }) {
  const { collapsed } = useSidebar();

  return (
    <main className="flex-1 overflow-x-hidden">
      <div
        className={`mx-auto max-w-7xl p-6 lg:p-8 transition-all duration-300`}
      >
        {children}
      </div>
    </main>
  );
}
