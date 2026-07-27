"use client";

import { memo, useEffect, useRef, useState } from "react";
import { useChat } from "@ai-sdk/react";
import {
  DefaultChatTransport,
  lastAssistantMessageIsCompleteWithToolCalls,
  type UIMessage,
} from "ai";
import { Markdown } from "./Markdown";
import { ModelChooser } from "@/components/ModelChooser";
import type { ModelOption } from "@/lib/config/models";

/** Kept as an alias so pages importing this name still compile; the shape now
 * comes from buildModelOptions so every module presents models identically. */
export type ChatModelOption = ModelOption;

/** A tool call handed to the host's executor (Ask Anything runs it in-browser). */
export interface ChatToolCall {
  toolName: string;
  toolCallId: string;
  input: unknown;
}

/** What a host's prepare() turns one chosen file into: the message parts to
 * send (file parts or data-attachment parts) plus chip copy. */
export interface PreparedAttachment {
  parts: unknown[];
  detail?: string;
}

/** Host-provided attachment support: which files the picker offers and how a
 * chosen file becomes message parts. prepare() throwing (with a student
 * readable message) marks the chip as failed without blocking the send. */
export interface AttachmentsConfig {
  accept: string;
  prepare: (file: File) => Promise<PreparedAttachment>;
}

interface PendingAttachment {
  id: string;
  label: string;
  status: "reading" | "ready" | "error";
  detail?: string;
  error?: string;
  parts?: unknown[];
}

interface ChatProps {
  moduleKey: string;
  moduleName: string;
  placeholder: string;
  models: ChatModelOption[];
  defaultModelId: string;
  initialMessages?: UIMessage[];
  /** Endpoint override: Ask Anything posts to its tools-bearing sibling route. */
  api?: string;
  /** Executes one tool call in the host (the browser) and resolves its output
   * object. When present, tool results are posted back automatically so the
   * model can continue (the agentic loop). */
  onToolCall?: (call: ChatToolCall) => Promise<unknown>;
  /** Renders one tool part of an assistant message (Ask Anything's tool card). */
  toolRenderer?: (part: Record<string, unknown>, key: string) => React.ReactNode;
  /** Enables the composer's Attach button (Ask Anything's files-in). */
  attachments?: AttachmentsConfig;
  /** Called whenever the transcript changes (streaming included), so a host
   * page can persist it (Ask Anything stores chats in localStorage). */
  onMessagesChange?: (messages: UIMessage[]) => void;
  /** Called when the student picks a different model, so a host page can
   * remember the choice per chat. */
  onModelChange?: (modelId: string) => void;
  /** Replaces the default "Start the conversation" copy (which is written for
   * the Coding Tutor) when a host module has its own register. */
  emptyState?: { heading: string; body: string };
}

/** True for the message parts that represent a tool call's lifecycle. */
function isToolPart(part: { type: string }): boolean {
  return part.type === "dynamic-tool" || part.type.startsWith("tool-");
}

/** An attachment chip in a transcript bubble: file name plus a short detail. */
function AttachmentChip({ name, detail }: { name: string; detail?: string }) {
  return (
    <span className="inline-flex max-w-full items-center gap-1 rounded-card border border-medium-tan bg-light-tan px-2 py-1 text-xs font-bold">
      <span aria-hidden="true">&#128206;</span>
      <span className="truncate">{name}</span>
      {detail ? <span className="font-normal text-dark-tan">({detail})</span> : null}
    </span>
  );
}

/**
 * Renders one attachment part of a transcript message: images as thumbnails,
 * PDFs and extracted files as labeled chips. A file part whose data URL was
 * offloaded to device storage and could not be rehydrated (deleted store,
 * private mode) falls back to a chip, never a broken image.
 */
function renderAttachmentPart(
  part: Record<string, unknown>,
  key: string,
): React.ReactNode {
  if (part.type === "file") {
    const url = typeof part.url === "string" ? part.url : "";
    const mediaType = typeof part.mediaType === "string" ? part.mediaType : "";
    const filename =
      typeof part.filename === "string" ? part.filename : "attachment";
    if (mediaType.startsWith("image/") && /^(data|blob|https):/.test(url)) {
      return (
        // Client-produced data URLs cannot go through next/image.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={key}
          src={url}
          alt={`Attached image: ${filename}`}
          className="max-h-64 max-w-full rounded-card border border-medium-tan"
        />
      );
    }
    return (
      <AttachmentChip
        key={key}
        name={filename}
        detail={mediaType === "application/pdf" ? "PDF" : undefined}
      />
    );
  }
  const data = (part as { data?: { name?: string; detail?: string } }).data;
  return (
    <AttachmentChip key={key} name={data?.name ?? "attachment"} detail={data?.detail} />
  );
}

/**
 * One message bubble. Memoised so that while a new reply streams, the already
 * finished messages are not re-rendered on every token. That keeps a running
 * in-browser code block (which can take many seconds) from re-parsing its
 * markdown underneath itself. A completed message's props are stable, so it
 * skips re-render; only the streaming message, whose text changes, re-renders.
 */
const ChatMessage = memo(function ChatMessage({
  id,
  role,
  parts,
  notice,
  toolRenderer,
}: {
  id: string;
  role: string;
  parts: UIMessage["parts"];
  notice?: string;
  toolRenderer?: (part: Record<string, unknown>, key: string) => React.ReactNode;
}) {
  const isUser = role === "user";
  const headingId = `speaker-${id}`;
  // Parts render in order: consecutive text parts join into one Markdown run,
  // and tool parts (when the host renders them) appear between the runs, so a
  // reply that thinks, runs code, and concludes reads in its true sequence.
  const segments: { key: string; node: React.ReactNode }[] = [];
  let textRun = "";
  let runIndex = 0;
  const flush = () => {
    if (!textRun) return;
    const content = textRun;
    segments.push({
      key: `text-${runIndex++}`,
      node: isUser ? (
        <p className="whitespace-pre-wrap">{content}</p>
      ) : (
        <Markdown>{content}</Markdown>
      ),
    });
    textRun = "";
  };
  for (const [i, part] of parts.entries()) {
    if (part.type === "text" && "text" in part) {
      textRun += part.text;
    } else if (toolRenderer && isToolPart(part)) {
      flush();
      const key = `tool-${(part as { toolCallId?: string }).toolCallId ?? i}`;
      segments.push({
        key,
        node: toolRenderer(part as unknown as Record<string, unknown>, key),
      });
    } else if (part.type === "file" || part.type === "data-attachment") {
      flush();
      const key = `attach-${i}`;
      segments.push({
        key,
        node: renderAttachmentPart(part as unknown as Record<string, unknown>, key),
      });
    }
  }
  flush();
  return (
    <article
      // Named by its own visible heading, so each message announces as "You" or
      // "ChatISA" without colliding with the composer label.
      aria-labelledby={headingId}
      className={
        isUser
          ? "self-end rounded-card border border-medium-tan bg-light-tan p-4 md:max-w-[85%]"
          : "rounded-card border border-medium-tan bg-paper p-4"
      }
    >
      <h3 id={headingId} className="mb-1 text-sm font-bold text-dark-tan">
        {isUser ? "You" : "ChatISA"}
      </h3>
      {segments.map((s) => (
        <div key={s.key} className="[&+div]:mt-3">
          {s.node}
        </div>
      ))}
      {/*
        A reply that arrived empty, or stopped mid-sentence, is explained rather
        than left as a blank or truncated bubble. The server decides the wording
        so it stays consistent with every other failure message.
      */}
      {!isUser && notice ? (
        <p className="mt-2 border-l-4 border-medium-tan pl-3 text-sm text-dark-tan">
          {notice}
        </p>
      ) : null}
    </article>
  );
});

export function Chat({
  moduleKey,
  moduleName,
  placeholder,
  models,
  defaultModelId,
  initialMessages = [],
  api,
  onToolCall,
  toolRenderer,
  attachments,
  onMessagesChange,
  onModelChange,
  emptyState,
}: ChatProps) {
  const [modelId, setModelId] = useState(defaultModelId);
  const [input, setInput] = useState("");
  const logRef = useRef<HTMLDivElement>(null);
  const errorRef = useRef<HTMLDivElement>(null);
  // Preserves the student's text if a send fails, so work is never lost.
  const [lastFailedInput, setLastFailedInput] = useState<string | null>(null);
  // Files chosen but not yet sent. Each prepares asynchronously (extraction,
  // downscaling, a dataset import) and its chip reflects the progress.
  const [pending, setPending] = useState<PendingAttachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const anyReading = pending.some((p) => p.status === "reading");

  function onFilesChosen(list: FileList | null) {
    if (!list || !attachments) return;
    for (const file of Array.from(list)) {
      const id = crypto.randomUUID();
      setPending((p) => [...p, { id, label: file.name, status: "reading" }]);
      attachments
        .prepare(file)
        .then((prep) =>
          setPending((p) =>
            p.map((x) =>
              x.id === id
                ? { ...x, status: "ready", parts: prep.parts, detail: prep.detail }
                : x,
            ),
          ),
        )
        .catch((err: unknown) =>
          setPending((p) =>
            p.map((x) =>
              x.id === id
                ? {
                    ...x,
                    status: "error",
                    error: err instanceof Error ? err.message : String(err),
                  }
                : x,
            ),
          ),
        );
    }
    // Reset so choosing the same file again re-fires the change event.
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  // The latest model id and tool executor, readable from callbacks created at
  // mount (the transport and useChat options are built once). Assigned in
  // effects, not during render, per the react-hooks/refs rule; both are read
  // only at request time, well after the effects have run.
  const modelIdRef = useRef(modelId);
  useEffect(() => {
    modelIdRef.current = modelId;
  }, [modelId]);
  const toolExecRef = useRef(onToolCall);
  useEffect(() => {
    toolExecRef.current = onToolCall;
  }, [onToolCall]);
  // addToolResult comes from the hook below; the onToolCall option (passed TO
  // the hook) reaches it through this ref once the hook has returned.
  const addToolResultRef = useRef<
    ((args: {
      state: "output-available";
      tool: never;
      toolCallId: string;
      output: unknown;
    }) => Promise<void>) | null
  >(null);

  // Created once. Routing fields ride on every request, including the automatic
  // continuation after a tool result, which sendMessage options do not cover.
  // The prepare callback captures modelIdRef but reads it only when a request
  // is sent, never during render, so the refs rule is quieted here.
  const [transport] = useState(
    // eslint-disable-next-line react-hooks/refs
    () =>
      new DefaultChatTransport({
        api: api ?? "/api/chat",
        prepareSendMessagesRequest: ({ id, messages, body }) => ({
          body: {
            id,
            messages,
            ...(body ?? {}),
            module: moduleKey,
            modelId: modelIdRef.current,
          },
        }),
      }),
  );

  // The onToolCall option captures toolExecRef/addToolResultRef; both are read
  // only when the model calls a tool (long after render), covered by the same
  // rationale as the transport's quieted refs rule above.
  const chatHelpers = useChat({
      messages: initialMessages,
      transport,
      // Providers stream tool-call input as rapid delta bursts (a hosted
      // PowerPoint build sends hundreds with no delay between them). Unthrottled,
      // each delta re-renders and re-runs the persist effect, and the nested
      // updates exceed React's depth limit ("Maximum update depth exceeded",
      // reproduced 2026-07-24 with a burst-streaming mock). One update per
      // animation frame is plenty for reading a stream.
      throttle: 50,
      onError() {
        setLastFailedInput((prev) => prev ?? input);
      },
      // The agentic loop (Ask Anything): the model's tool call is executed in
      // the browser by the host's executor, its output is posted back, and once
      // every call in the reply has an output the next request fires by itself.
      ...(onToolCall
        ? {
            sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
            onToolCall: async ({ toolCall }: { toolCall: unknown }) => {
              const call = toolCall as {
                toolName?: string;
                toolCallId: string;
                input?: unknown;
              };
              const exec = toolExecRef.current;
              if (!exec || !call.toolName) return;
              let output: unknown;
              try {
                output = await exec({
                  toolName: call.toolName,
                  toolCallId: call.toolCallId,
                  input: call.input,
                });
              } catch (err) {
                // The executor is expected to encode failures in its output;
                // this catch is the belt-and-braces so the loop never hangs.
                output = {
                  status: "error",
                  error: err instanceof Error ? err.message : String(err),
                };
              }
              // NOT awaited, deliberately: the store awaits onToolCall inside
              // its stream-processing job, and addToolResult queues on the same
              // job executor, so awaiting it here deadlocks the loop (the part
              // stays input-available forever). Fire it and return.
              void addToolResultRef.current?.({
                state: "output-available",
                tool: call.toolName as never,
                toolCallId: call.toolCallId,
                output,
              });
            },
          }
        : {}),
    });
  const { messages, sendMessage, status, stop, error, clearError, addToolResult } =
    chatHelpers;
  useEffect(() => {
    addToolResultRef.current = addToolResult as never;
  }, [addToolResult]);

  // Reports every transcript change (streaming included) to the host page, so
  // Ask Anything can persist the chat to the device as it grows. The callback
  // rides in a ref so the effect fires only when the MESSAGES change; a parent
  // that re-creates the callback while persisting must not re-trigger it (that
  // exact feedback loop hit "maximum update depth" in development).
  const onMessagesChangeRef = useRef(onMessagesChange);
  useEffect(() => {
    onMessagesChangeRef.current = onMessagesChange;
  }, [onMessagesChange]);
  useEffect(() => {
    onMessagesChangeRef.current?.(messages);
  }, [messages]);

  /** Routing and policy fields the server needs for this turn. */
  function sendOptions() {
    return { body: { module: moduleKey, modelId } };
  }

  const busy = status === "submitted" || status === "streaming";

  // Move focus to the error so keyboard and screen-reader users find it.
  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);

  function submit(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    const readyParts = pending
      .filter((p) => p.status === "ready")
      .flatMap((p) => p.parts ?? []);
    // A message needs words or at least one ready file; nothing sends while a
    // file is still being read.
    if ((!text && readyParts.length === 0) || busy || anyReading) return;
    clearError();
    setLastFailedInput(null);
    if (readyParts.length > 0) {
      const parts = [
        ...readyParts,
        ...(text ? [{ type: "text", text }] : []),
      ];
      sendMessage({ parts } as Parameters<typeof sendMessage>[0], sendOptions());
    } else {
      sendMessage({ text }, sendOptions());
    }
    setInput("");
    setPending([]);
  }

  function retry() {
    const text = lastFailedInput?.trim();
    clearError();
    if (text) {
      setLastFailedInput(null);
      sendMessage({ text }, sendOptions());
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="w-full">
          <ModelChooser
            options={models}
            value={modelId}
            onChange={(id) => {
              setModelId(id);
              onModelChange?.(id);
            }}
            help="Switching applies to your next message. Earlier replies stay as they are."
          />
        </div>
      </div>

      <div
        ref={logRef}
        role="log"
        aria-label={`${moduleName} conversation`}
        aria-busy={busy}
        className="flex flex-col gap-4"
      >
        {messages.length === 0 ? (
          <div className="rounded-card border border-medium-tan bg-paper p-5">
            <h2 className="text-xl">
              {emptyState?.heading ?? "Start the conversation"}
            </h2>
            <p className="mt-2">
              {emptyState?.body ??
                "Ask a question about your code or an analytics concept. Answers include examples in both R and Python."}
            </p>
          </div>
        ) : null}

        {messages.map((message) => {
          const notice = (message.metadata as { notice?: string } | undefined)
            ?.notice;
          return (
            <ChatMessage
              key={message.id}
              id={message.id}
              role={message.role}
              parts={message.parts}
              notice={notice}
              toolRenderer={toolRenderer}
            />
          );
        })}
      </div>

      {/* Streaming and error state are announced, never color-only. */}
      <p role="status" className="text-sm text-dark-tan">
        {status === "submitted"
          ? "Sending your message."
          : status === "streaming"
            ? "ChatISA is responding."
            : ""}
      </p>

      {error ? (
        <div
          ref={errorRef}
          tabIndex={-1}
          role="alert"
          className="rounded-card border-2 border-miami-red bg-paper p-4"
        >
          <h2 className="font-bold text-miami-red">That response failed</h2>
          <p className="mt-1">
            {error.message ||
              "The model could not complete that response. Your message was kept."}
          </p>
          <button
            type="button"
            onClick={retry}
            className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
          >
            Try again
          </button>
        </div>
      ) : null}

      <form onSubmit={submit} className="flex flex-col gap-2">
        {pending.length > 0 ? (
          <ul
            aria-label="Files to send"
            className="flex flex-wrap gap-2"
          >
            {pending.map((p) => (
              <li
                key={p.id}
                className={`inline-flex items-center gap-2 rounded-card border px-2 py-1 text-xs ${
                  p.status === "error"
                    ? "border-miami-red bg-paper"
                    : "border-medium-tan bg-light-tan"
                }`}
              >
                <span className="max-w-52 truncate font-bold">{p.label}</span>
                <span className="text-dark-tan" role="status">
                  {p.status === "reading"
                    ? "Reading..."
                    : p.status === "error"
                      ? p.error
                      : p.detail ?? "ready"}
                </span>
                <button
                  type="button"
                  onClick={() =>
                    setPending((list) => list.filter((x) => x.id !== p.id))
                  }
                  aria-label={`Remove ${p.label}`}
                  className="font-bold text-dark-tan hover:text-miami-red"
                >
                  &times;
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        <label htmlFor="chat-input" className="text-sm font-bold">
          Your message
        </label>
        <textarea
          id="chat-input"
          name="message"
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) submit(e);
          }}
          placeholder={placeholder}
          className="w-full rounded-card border border-medium-tan bg-paper p-3"
          aria-describedby="chat-input-help"
        />
        <p id="chat-input-help" className="text-sm text-dark-tan">
          Press Enter to send. Shift and Enter starts a new line.
        </p>
        <div className="flex flex-wrap gap-2">
          <button
            type="submit"
            disabled={
              busy ||
              anyReading ||
              (input.trim().length === 0 &&
                !pending.some((p) => p.status === "ready"))
            }
            className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
          >
            Send message
          </button>
          {attachments ? (
            <>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept={attachments.accept}
                onChange={(e) => onFilesChosen(e.target.files)}
                className="sr-only"
                id="chat-attach-input"
                aria-label="Choose files to attach"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={busy}
                className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:text-medium-gray"
              >
                Attach file
              </button>
            </>
          ) : null}
          {busy ? (
            <button
              type="button"
              onClick={stop}
              className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
            >
              Stop generating
            </button>
          ) : null}
        </div>
      </form>
    </div>
  );
}
