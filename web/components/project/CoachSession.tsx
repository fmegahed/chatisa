// components/project/CoachSession.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { Markdown } from "@/components/chat/Markdown";
import { ModelChooser } from "@/components/ModelChooser";
import { ScopingDeliverable } from "@/components/project/ScopingDeliverable";
import { GenericDeliverable } from "@/components/project/GenericDeliverable";
import type { ScopingContent } from "@/lib/project/scoping";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";
import type { ModelOption } from "@/lib/config/models";

type CoachSessionProps = {
  projectId: string;
  projectName: string;
  coachType: string;
  coachTitle: string;
  models: ModelOption[];
  defaultModelId: string;
  initialContent: unknown;
  initialMessages: UIMessage[];
  initialLastUpdatedBy: string | null;
} & ({ kind: "scoping" } | { kind: "generic"; spec: CoachSpec });

export function CoachSession(props: CoachSessionProps) {
  const { projectId, projectName, coachType, coachTitle, models, defaultModelId } = props;
  const base = `/api/project-assistant/${projectId}/coach/${coachType}`;
  const [modelId, setModelId] = useState(defaultModelId);
  const [input, setInput] = useState("");
  const [content, setContent] = useState<unknown>(props.initialContent);
  const [lastUpdatedBy, setLastUpdatedBy] = useState<string | null>(props.initialLastUpdatedBy);
  const [saveError, setSaveError] = useState<string | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [transport] = useState(() => new DefaultChatTransport({ api: base }));
  const { messages, sendMessage, status, stop, error, clearError } = useChat({
    messages: props.initialMessages,
    transport,
    onFinish() {
      // The coach's tool calls changed the worksheet on the server; pull it
      // back so the panel reflects them, and persist the transcript.
      void refetchDeliverable();
      void fetch(`${base}/deliverable`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: messages }),
      });
    },
  });

  const busy = status === "submitted" || status === "streaming";

  async function refetchDeliverable() {
    try {
      const res = await fetch(`${base}/deliverable`);
      if (!res.ok) return;
      const data = (await res.json()) as { contentJson: string; lastUpdatedBy: string | null };
      setContent(JSON.parse(data.contentJson));
      setLastUpdatedBy(data.lastUpdatedBy);
    } catch {
      // A failed refetch leaves the last known worksheet in place.
    }
  }

  /** Direct edits save on a short debounce, last-save-wins. */
  function onWorksheetChange(next: unknown) {
    setContent(next);
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      setSaveError(null);
      fetch(`${base}/deliverable`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: next }),
      })
        .then((res) => {
          if (!res.ok) throw new Error("save failed");
          return res.json();
        })
        .then((data: { lastUpdatedBy: string | null }) => setLastUpdatedBy(data.lastUpdatedBy))
        .catch(() => setSaveError("Your last edit did not save. Check your connection."));
    }, 600);
  }

  useEffect(() => () => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
  }, []);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    clearError();
    sendMessage({ text }, { body: { modelId } });
    setInput("");
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <Link href={`/project-assistant/${projectId}`} className="text-sm underline">
        Back to project
      </Link>
      <h1 className="mt-3 text-3xl">{coachTitle}</h1>
      <p className="mt-1 text-dark-tan">{projectName}</p>

      <div className="mt-6 grid gap-8 lg:grid-cols-2">
        {/* Chat */}
        <div className="flex flex-col gap-4">
          <ModelChooser
            options={models}
            value={modelId}
            onChange={setModelId}
            help="Switching applies to your next message."
          />
          <div role="log" aria-label="Coach conversation" aria-busy={busy} className="flex flex-col gap-4">
            {messages.length === 0 ? (
              <div className="rounded-card border border-medium-tan bg-paper p-5">
                <h2 className="text-xl">Start scoping</h2>
                <p className="mt-2">
                  Describe your project in a sentence or two. The coach will walk you
                  through the worksheet, one question at a time, and fill it as you go.
                </p>
              </div>
            ) : null}
            {messages.map((m) => {
              const text = m.parts
                .filter((p) => p.type === "text")
                .map((p) => ("text" in p ? p.text : ""))
                .join("");
              if (!text) return null;
              const isUser = m.role === "user";
              return (
                <article
                  key={m.id}
                  className={
                    isUser
                      ? "self-end rounded-card border border-medium-tan bg-light-tan p-4 md:max-w-[85%]"
                      : "rounded-card border border-medium-tan bg-paper p-4"
                  }
                >
                  <h3 className="mb-1 text-sm font-bold text-dark-tan">{isUser ? "You" : "Coach"}</h3>
                  {isUser ? <p className="whitespace-pre-wrap">{text}</p> : <Markdown>{text}</Markdown>}
                </article>
              );
            })}
          </div>

          <p role="status" className="text-sm text-dark-tan">
            {status === "submitted" ? "Sending." : status === "streaming" ? "The coach is responding." : ""}
          </p>
          {error ? (
            <div role="alert" className="rounded-card border-2 border-miami-red bg-paper p-4">
              <p className="font-bold text-miami-red">That response failed</p>
              <p className="mt-1">{error.message || "The coach could not respond. Your message was kept."}</p>
            </div>
          ) : null}

          <form onSubmit={submit} className="flex flex-col gap-2">
            <label htmlFor="coach-input" className="text-sm font-bold">
              Your message
            </label>
            <textarea
              id="coach-input"
              rows={3}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) submit(e);
              }}
              className="w-full rounded-card border border-medium-tan bg-paper p-3"
            />
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={busy || input.trim().length === 0}
                className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
              >
                Send message
              </button>
              {busy ? (
                <button
                  type="button"
                  onClick={stop}
                  className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
                >
                  Stop
                </button>
              ) : null}
            </div>
          </form>
        </div>

        {/* Live deliverable */}
        <div className="lg:border-l lg:border-medium-tan lg:pl-8">
          {saveError ? (
            <p role="alert" className="mb-3 text-miami-red">
              {saveError}
            </p>
          ) : null}
          <div className="mb-4">
            <a
              href={`${base}/export`}
              className="inline-block rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
            >
              Download Word
            </a>
          </div>
          {props.kind === "scoping" ? (
            <ScopingDeliverable
              content={content as ScopingContent}
              onChange={onWorksheetChange}
              lastUpdatedBy={lastUpdatedBy}
            />
          ) : (
            <GenericDeliverable
              spec={props.spec}
              content={content as GenericContent}
              onChange={onWorksheetChange}
              lastUpdatedBy={lastUpdatedBy}
            />
          )}
        </div>
      </div>
    </div>
  );
}
