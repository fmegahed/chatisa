import Link from "next/link";
import type { ModuleInfo } from "@/lib/modules";

/**
 * Temporary stand-in page while a module is rebuilt slice by slice.
 * Replaced by the real module UI in its implementation slice.
 */
export function ModulePlaceholder({ module: mod }: { module: ModuleInfo }) {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">{mod.name}</h1>
      <p className="mt-4 text-lg leading-relaxed">{mod.description}</p>
      <div
        role="status"
        className="mt-8 rounded-card border border-medium-tan bg-light-tan p-5"
      >
        <p className="font-bold">This module is coming soon.</p>
        <p className="mt-1">
          {mod.name} is part of the new ChatISA and will be available here
          soon. Until then, the current ChatISA remains available as usual.
        </p>
      </div>
      <p className="mt-8">
        <Link
          href="/"
          className="font-bold text-accent-red underline underline-offset-2"
        >
          Back to all modules
        </Link>
      </p>
    </div>
  );
}
