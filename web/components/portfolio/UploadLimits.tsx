"use client";

import { MAX_CHARS_PER_FILE } from "@/lib/portfolio/files";
import { MAX_NOTEBOOK_BYTES, MAX_TEXT_BYTES } from "@/lib/portfolio/intake";
import { PUSH_LIMITS } from "@/lib/scout/github";

const mb = (n: number) => `${Math.round(n / (1024 * 1024))} MB`;
const kb = (n: number) => `${Math.round(n / 1000)} KB`;

/**
 * The limits, said before the first file is picked rather than discovered
 * one "too large" at a time. Every number is derived from the constant the
 * code enforces, so this copy cannot drift from the behaviour.
 */
export function UploadLimits(props: { perProject?: number }) {
  return (
    <p className="mt-2 rounded-card bg-light-tan p-3 text-sm">
      <strong>Limits.</strong> Up to {mb(PUSH_LIMITS.fileBytes)} per file and {mb(PUSH_LIMITS.totalBytes)} for the
      whole site, at most {PUSH_LIMITS.files} files{props.perProject ? ` (${props.perProject} per project)` : ""}.
      Files over {mb(PUSH_LIMITS.fileBytes)} are kept out of the publish. When writing the page, the AI reads the
      first {MAX_CHARS_PER_FILE.toLocaleString()} characters of each text file (code, notebooks, Word, PowerPoint);
      text files over {kb(MAX_TEXT_BYTES)} and notebooks over {mb(MAX_NOTEBOOK_BYTES)} are published as they are but not
      read. PDFs and images are published, not read.
    </p>
  );
}
