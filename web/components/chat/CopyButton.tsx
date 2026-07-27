"use client";

import { useState } from "react";

/**
 * Copies `text` to the clipboard and announces success. Shared by plain code
 * blocks and the runnable/editable ones, so an edited snippet copies exactly
 * what the student changed, not the original.
 */
export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={copy}
        className="rounded-card border border-medium-tan bg-paper px-2 py-1 text-xs font-bold text-ink hover:border-miami-red hover:text-miami-red"
      >
        Copy code
      </button>
      {/* Status is announced, not conveyed by color alone. */}
      <span role="status" className="text-xs text-dark-tan">
        {copied ? "Copied to clipboard" : ""}
      </span>
    </div>
  );
}
