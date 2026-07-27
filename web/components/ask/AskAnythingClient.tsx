"use client";

import dynamic from "next/dynamic";
import type { ModelOption } from "@/lib/config/models";

/**
 * Ask Anything renders client-only. Its state seeds from localStorage (the
 * device-side chat store) and a random chat id, both of which differ between a
 * server render and the browser, so server-rendering it causes a hydration
 * mismatch. Loaded with ssr disabled behind a short placeholder, the same
 * pattern as the Coding Studio (SandboxClient).
 */
const AskAnything = dynamic(
  () => import("./AskAnything").then((m) => m.AskAnything),
  {
    ssr: false,
    loading: () => (
      <p className="text-dark-tan">Loading your chats...</p>
    ),
  },
);

export function AskAnythingClient(props: {
  models: ModelOption[];
  defaultModelId: string;
}) {
  return (
    <AskAnything models={props.models} defaultModelId={props.defaultModelId} />
  );
}
