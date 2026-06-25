"use client";

import { Menu, UtensilsCrossed } from "lucide-react";
import { useSidebar } from "@/components/sidebar-context";

/** Mobile-only header bar with hamburger menu button */
export function MobileHeader() {
  const { setMobileOpen } = useSidebar();

  return (
    <header className="sticky top-0 z-40 flex items-center gap-3 border-b border-white/5 bg-surface/60 px-4 py-3 backdrop-blur-xl md:hidden">
      <button
        onClick={() => setMobileOpen(true)}
        className="rounded-control p-1.5 text-text-muted hover:bg-white/5 hover:text-text-primary transition-colors"
        aria-label="Open menu"
      >
        <Menu className="h-5 w-5" />
      </button>
      <div className="flex items-center gap-2">
        <div className="flex h-7 w-7 items-center justify-center rounded-control bg-gradient-primary shadow-glow">
          <UtensilsCrossed className="h-4 w-4 text-white" />
        </div>
        <span className="text-sm font-semibold">Restaurant Tracker</span>
      </div>
    </header>
  );
}
