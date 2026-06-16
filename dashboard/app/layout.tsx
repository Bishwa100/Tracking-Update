import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";
import { Sidebar } from "@/components/sidebar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Restaurant Visitor Tracker",
  description: "Auto-registering visitor detection, recognition, and analytics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans">
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 overflow-x-hidden">
            <div className="mx-auto max-w-7xl p-6 lg:p-8">{children}</div>
          </main>
        </div>
      </body>
    </html>
  );
}
