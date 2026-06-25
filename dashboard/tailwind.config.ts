import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Modern slate-based dark theme with deeper backgrounds.
        bg: "#0A0F1E",        // deeper than slate-900
        surface: "#131A2B",   // glass card base
        "surface-2": "#1B2438",
        card: "#2A3650",      // borders / dividers
        primary: "#3B82F6",   // blue-500
        "primary-bright": "#60A5FA",
        success: "#10B981",   // emerald-500
        warning: "#F59E0B",   // amber-500
        danger: "#EF4444",    // red-500
        accent: "#8B5CF6",    // violet-500
        "accent-bright": "#A78BFA",
        "text-primary": "#F8FAFC",   // slate-50
        "text-secondary": "#94A3B8", // slate-400
        "text-muted": "#64748B",     // slate-500
      },
      borderRadius: {
        card: "16px",
        control: "10px",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(59,130,246,0.15), 0 8px 30px -8px rgba(59,130,246,0.25)",
        "glow-accent": "0 0 0 1px rgba(139,92,246,0.15), 0 8px 30px -8px rgba(139,92,246,0.25)",
        card: "0 1px 2px rgba(0,0,0,0.3), 0 8px 24px -12px rgba(0,0,0,0.5)",
      },
      backgroundImage: {
        "gradient-primary": "linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%)",
        "gradient-surface": "linear-gradient(180deg, rgba(27,36,56,0.6) 0%, rgba(19,26,43,0.6) 100%)",
        "grid-faint":
          "radial-gradient(circle at 1px 1px, rgba(148,163,184,0.06) 1px, transparent 0)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
        "pulse-ring": {
          "0%": { boxShadow: "0 0 0 0 rgba(16,185,129,0.5)" },
          "70%": { boxShadow: "0 0 0 8px rgba(16,185,129,0)" },
          "100%": { boxShadow: "0 0 0 0 rgba(16,185,129,0)" },
        },
        /* Toast slide-up + fade-in entrance. */
        "toast-in": {
          "0%": { opacity: "0", transform: "translateY(16px) scale(0.96)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        /* Toast slide-right + fade-out exit. */
        "toast-out": {
          "0%": { opacity: "1", transform: "translateX(0) scale(1)" },
          "100%": { opacity: "0", transform: "translateX(80px) scale(0.96)" },
        },
        /* Toast progress bar countdown. */
        "toast-progress": {
          "0%": { width: "100%" },
          "100%": { width: "0%" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.3s ease-out",
        shimmer: "shimmer 1.5s infinite",
        "pulse-ring": "pulse-ring 2s infinite",
        "toast-in": "toast-in 0.3s ease-out forwards",
        "toast-out": "toast-out 0.3s ease-in forwards",
        "toast-progress": "toast-progress 4s linear forwards",
      },
    },
  },
  plugins: [],
};

export default config;
