"use client";

import { useEffect, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { Markdown } from "@/components/chat/Markdown";
import { ModelChooser } from "@/components/ModelChooser";
import type { ModelOption } from "@/lib/config/models";

/**
 * The Sandbox side chat: coding help that can see the student's current script,
 * last result, and variables (types and columns only, never data values). It
 * reuses the streaming /api/chat route with the sandbox_chat module, sending
 * that context per message so it stays out of the saved conversation.
 */
export function SandboxChat(props: {
  models: ModelOption[];
  defaultModelId: string;
  /** Current work, gathered at send time. */
  getContext: () => string;
  onClose: () => void;
}) {
  // Remember the chosen model across reloads (client-only render, so
  // localStorage is available), falling back if it is no longer offered.
  const [modelId, setModelId] = useState(() => {
    try {
      const saved = window.localStorage.getItem("sb-chat-model");
      if (saved && props.models.some((m) => m.id === saved)) return saved;
    } catch {
      // fall through
    }
    return props.defaultModelId;
  });
  const chooseModel = (id: string) => {
    setModelId(id);
    try {
      window.localStorage.setItem("sb-chat-model", id);
    } catch {
      // best-effort
    }
  };
  const [input, setInput] = useState("");
  const [transport] = useState(
    () => new DefaultChatTransport({ api: "/api/chat" }),
  );
  const endRef = useRef<HTMLDivElement | null>(null);

  const { messages, sendMessage, status, stop, error } = useChat({ transport });
  const busy = status === "submitted" || status === "streaming";

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, busy]);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    sendMessage(
      { text },
      {
        body: {
          module: "sandbox_chat",
          modelId,
          context: props.getContext(),
        },
      },
    );
    setInput("");
  }

  const noModels = props.models.length === 0;

  return (
    <aside
      aria-label="Sandbox assistant"
      className="flex w-full flex-col overflow-hidden border-l border-[var(--sb-border)] bg-[var(--sb-panel)] text-[var(--sb-text)]"
    >
      <header className="flex items-center justify-between border-b border-[var(--sb-border)] bg-[var(--sb-header)] px-3 py-2">
        <h2 className="text-sm font-bold">Assistant</h2>
        <button
          type="button"
          onClick={props.onClose}
          className="rounded-card border border-[var(--sb-border)] px-2 py-1 text-xs font-bold hover:border-[var(--sb-accent)]"
        >
          Close
        </button>
      </header>

      {noModels ? (
        <p className="p-3 text-sm text-[var(--sb-muted)]">
          No models are configured on this server, so the assistant is
          unavailable.
        </p>
      ) : (
        <>
          <div className="border-b border-[var(--sb-border)] p-2">
            <ModelChooser
              options={props.models}
              value={modelId}
              onChange={chooseModel}
              help="The assistant can see your script, last result, and variables (not your data's values)."
            />
          </div>

          <div
            tabIndex={0}
            role="log"
            aria-label="Conversation"
            className="min-h-0 flex-1 overflow-auto p-3"
          >
            {messages.length === 0 ? (
              <p className="text-sm text-[var(--sb-muted)]">
                Ask about your code, an error you hit, or how to work with your
                data. I can see what you are working on.
              </p>
            ) : null}
            {messages.map((message) => {
              const text = message.parts
                .filter((p) => p.type === "text")
                .map((p) => ("text" in p ? p.text : ""))
                .join("");
              const isUser = message.role === "user";
              return (
                <div
                  key={message.id}
                  className={`mb-3 rounded-card border border-[var(--sb-border)] p-2 ${isUser ? "bg-[var(--sb-header)]" : "bg-[var(--sb-bg)]"}`}
                >
                  <p className="mb-1 text-xs font-bold text-[var(--sb-muted)]">
                    {isUser ? "You" : "Assistant"}
                  </p>
                  {isUser ? (
                    <p className="whitespace-pre-wrap text-sm">{text}</p>
                  ) : (
                    <Markdown>{text}</Markdown>
                  )}
                </div>
              );
            })}
            {error ? (
              <p role="alert" className="text-sm text-[var(--sb-accent)]">
                Something went wrong. Try again.
              </p>
            ) : null}
            <div ref={endRef} />
          </div>

          <form
            onSubmit={submit}
            className="border-t border-[var(--sb-border)] p-2"
          >
            <label htmlFor="sb-chat-input" className="sr-only">
              Your message
            </label>
            <textarea
              id="sb-chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  submit(e);
                }
              }}
              rows={2}
              placeholder="Ask about your code, an error, or your data."
              className="w-full resize-none rounded-card border border-[var(--sb-border)] bg-[var(--sb-bg)] p-2 text-sm text-[var(--sb-text)] focus:border-[var(--sb-accent)] focus:outline-none"
            />
            <div className="mt-1 flex items-center justify-between">
              <span className="text-xs text-[var(--sb-muted)]">
                Enter to send
              </span>
              {busy ? (
                <button
                  type="button"
                  onClick={stop}
                  className="rounded-card border border-[var(--sb-border)] px-3 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
                >
                  Stop
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={input.trim().length === 0}
                  className="rounded-card bg-[var(--sb-accent)] px-3 py-1 text-sm font-bold text-white disabled:cursor-not-allowed disabled:opacity-60"
                >
                  Send
                </button>
              )}
            </div>
          </form>
        </>
      )}
    </aside>
  );
}
