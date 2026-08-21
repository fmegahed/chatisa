"use client";

import { measure } from "@/lib/portfolio/files";
import { PUSH_LIMITS, type PushFile } from "@/lib/scout/github";

const mb = (n: number) => `${(n / 1_000_000).toFixed(2)} MB`;
/** Derived, so the copy cannot drift from the limit the push enforces. */
const kb = (n: number) => `${Math.round(n / 1_000)} KB`;

/** What the repository will weigh, said before the push rather than after. */
export function SizeMeter(props: { files: PushFile[] }) {
  const m = measure(props.files);
  return (
    <div
      role="status"
      className={`mt-3 rounded-card border p-3 ${m.ok ? "border-medium-tan bg-light-tan" : "border-miami-red bg-paper"}`}
    >
      <p>
        <strong>{m.count}</strong> of {PUSH_LIMITS.files} files, <strong>{mb(m.totalBytes)}</strong> of{" "}
        {mb(PUSH_LIMITS.totalBytes)}
      </p>
      {m.over.length > 0 ? (
        <p className="mt-1 font-bold text-miami-red">
          Too large to publish ({kb(PUSH_LIMITS.fileBytes)} limit per file):{" "}
          {m.over.map((o) => o.path).join(", ")}
        </p>
      ) : null}
      {m.count > PUSH_LIMITS.files || m.totalBytes > PUSH_LIMITS.totalBytes ? (
        <p className="mt-1 font-bold text-miami-red">
          Remove or untick some files to fit the repository limits.
        </p>
      ) : null}
    </div>
  );
}
