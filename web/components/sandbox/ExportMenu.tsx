"use client";

import { useEffect, useRef, useState } from "react";
import type { ExportFormat } from "@/lib/sandbox/export";

/**
 * A small menu button that exports one named tabular object as CSV or TSV. The
 * menu is two buttons; Escape closes it and returns focus to the trigger, an
 * outside click dismisses it, and every item is keyboard reachable.
 */
export function ExportMenu({
  name,
  onExport,
}: {
  name: string;
  onExport: (format: ExportFormat) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  function choose(format: ExportFormat) {
    setOpen(false);
    triggerRef.current?.focus();
    onExport(format);
  }

  return (
    <div ref={rootRef} className="relative inline-block">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Export ${name}`}
        title={`Export ${name} as CSV or TSV`}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
        className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent)]"
      >
        Export
      </button>
      {open ? (
        <div
          role="menu"
          aria-label={`Export ${name} format`}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setOpen(false);
              triggerRef.current?.focus();
            }
          }}
          className="absolute right-0 z-10 mt-1 min-w-[10rem] rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] py-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => choose("csv")}
            className="block w-full px-3 py-1 text-left text-xs font-bold text-[var(--sb-text)] hover:bg-[var(--sb-header)] hover:text-[var(--sb-accent)]"
          >
            Export as CSV
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => choose("tsv")}
            className="block w-full px-3 py-1 text-left text-xs font-bold text-[var(--sb-text)] hover:bg-[var(--sb-header)] hover:text-[var(--sb-accent)]"
          >
            Export as TSV
          </button>
        </div>
      ) : null}
    </div>
  );
}
