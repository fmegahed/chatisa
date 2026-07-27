"use client";

import dynamic from "next/dynamic";
import type { ModelOption } from "@/lib/config/models";

/**
 * The Sandbox renders client-only. Its resizable panels compute layout from the
 * DOM, so server-rendering them causes a hydration mismatch; and its runtimes
 * are browser-only anyway. So it is loaded with ssr disabled, behind a short
 * placeholder.
 */
const Sandbox = dynamic(
  () => import("./Sandbox").then((m) => m.Sandbox),
  {
    ssr: false,
    loading: () => (
      <div className="mx-auto max-w-2xl px-4 py-16">
        <h1 className="text-3xl">Coding Studio</h1>
        <p className="mt-4 text-dark-tan">Loading the workspace...</p>
      </div>
    ),
  },
);

export function SandboxClient(props: {
  models: ModelOption[];
  defaultModelId: string;
  userEmail: string;
}) {
  return (
    <Sandbox
      models={props.models}
      defaultModelId={props.defaultModelId}
      userEmail={props.userEmail}
    />
  );
}
