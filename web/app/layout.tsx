import type { Metadata } from "next";
import "@fontsource-variable/source-serif-4";
// KaTeX styles for math in model replies (rendered by components/chat/Markdown).
import "katex/dist/katex.min.css";
import "./globals.css";
import { MockModeBanner } from "@/components/MockModeBanner";

export const metadata: Metadata = {
  title: {
    default: "ChatISA: AI tools for Miami University students",
    template: "%s · ChatISA",
  },
  description:
    "Free, sponsored access to leading AI models through six learning modules, built by Miami University's Farmer School of Business.",
  icons: { icon: "/brand/beveled-m.png" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    // Browser extensions commonly add attributes to <html> and <body> before
    // React hydrates (for example screen recorders). suppressHydrationWarning
    // ignores attribute differences on these two elements only; mismatches in
    // our own markup are still reported.
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <body
        className="flex min-h-full flex-col bg-warm-white text-ink"
        suppressHydrationWarning
      >
        <a href="#main" className="skip-link">
          Skip to main content
        </a>
        <MockModeBanner />
        {children}
      </body>
    </html>
  );
}
