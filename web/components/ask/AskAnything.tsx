"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { UIMessage } from "ai";
import {
  Chat,
  type ChatToolCall,
  type PreparedAttachment,
} from "@/components/chat/Chat";
import { ToolCard } from "@/components/ask/ToolCard";
import type { ModelOption } from "@/lib/config/models";
import { CHAT_MODULES } from "@/lib/chat/config";
import {
  deleteChat,
  deriveTitle,
  listChats,
  saveChat,
  type StoredChat,
} from "@/lib/ask/chat-store";
import {
  enrichPythonError,
  truncateToolOutput,
  type AskToolName,
} from "@/lib/ask/tools";
import {
  PDF_PAGE_SOFT_CAP,
  attachmentPart,
  classifyFile,
  datasetAnnouncement,
  estimatePdfPages,
  rejectionReason,
  truncateAttachmentText,
} from "@/lib/files/attachments";
import { officeTextFromFile } from "@/lib/files/office-text";
import { notebookToText } from "@/lib/files/notebook-text";
import { prepareImage } from "@/lib/files/image";
import {
  deleteFilesForChat,
  fileRef,
  getFile,
  idFromRef,
  isFileRef,
  putFile,
} from "@/lib/ask/file-store";
import {
  detectDelimiter,
  nameFromFile,
  uniqueName,
} from "@/lib/sandbox/upload";
import {
  createSession,
  isRunSupported,
  type RunSession,
} from "@/lib/run/manager";
import { RUNNABLE_LANGUAGES } from "@/lib/run/languages";
import { buildPyodideIndex, type PyodideIndex } from "@/lib/sandbox/packages";

/** Tool name to runtime language id. */
const TOOL_LANG: Record<AskToolName, "python" | "r" | "sql"> = {
  run_python: "python",
  run_r: "r",
  run_sql: "sql",
};

/** Cap on browser tool executions per turn, so a looping model cannot run away. */
const MAX_TOOL_STEPS_PER_TURN = 10;

/** What the composer's file picker offers (matches classifyFile). */
const ATTACH_ACCEPT =
  ".png,.jpg,.jpeg,.webp,.gif,.pdf,.csv,.tsv,.xlsx,.docx,.pptx,.txt,.md,.json,.py,.r,.rmd,.qmd,.ipynb,.html";

/** Notebook plot outputs attached as native images per notebook (v6.1.1). */
const NOTEBOOK_PLOT_CAP = 4;

/** Browser-only: notebook plot base64 to a File for the image pipeline. */
function fileFromBase64(base64: string, name: string, mediaType: string): File {
  const bin = atob(base64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new File([bytes], name, { type: mediaType });
}

/** A small CSV-ish sample of a SQL result table for the model and the card. */
function tableSample(table: { columns: string[]; rows: Record<string, unknown>[] }): string {
  const head = table.columns.join(", ");
  const rows = table.rows
    .slice(0, 20)
    .map((r) => table.columns.map((c) => String(r[c] ?? "NULL")).join(", "));
  const more =
    table.rows.length > 20 ? `\n[${table.rows.length - 20} more rows]` : "";
  return `${head}\n${rows.join("\n")}${more}`;
}

/** A short relative label ("just now", "3h ago", "Jul 20") for the sidebar. */
function relativeLabel(ts: number): string {
  const mins = Math.round((Date.now() - ts) / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`"${file.name}" could not be read.`));
    reader.readAsDataURL(file);
  });
}

type MessagePart = { type: string; [k: string]: unknown };
type LooseMessage = { id?: string; parts?: MessagePart[]; [k: string]: unknown };

/**
 * The Ask Anything shell: a sidebar of device-stored chats around the shared
 * Chat component. A brand-new chat is held in memory and only written to
 * localStorage once it has a message, so abandoned "New chat" clicks never
 * litter the list. The Chat component is keyed by chat id, so switching chats
 * remounts it with that chat's transcript and model.
 *
 * Slice C adds attachments. Raw payloads (images, PDFs) do not fit the
 * localStorage quota, so persisted messages carry aa-file references while the
 * bytes live in IndexedDB; opening a chat rehydrates them and deleting a chat
 * deletes them. Datasets are imported into the chat's Python session and only
 * their announcement rides in the message.
 */
export function AskAnything({
  models,
  defaultModelId,
}: {
  models: ModelOption[];
  defaultModelId: string;
}) {
  const mod = CHAT_MODULES.ask_anything;
  // localStorage is read once at mount (client component, so it exists).
  const [chats, setChats] = useState<StoredChat[]>(() => {
    try {
      return listChats(window.localStorage);
    } catch {
      return [];
    }
  });
  const [activeId, setActiveId] = useState<string>(() => crypto.randomUUID());
  const [modelForActive, setModelForActive] = useState<string | null>(null);
  const [trimNote, setTrimNote] = useState(false);
  // The chat list collapses behind a toggle on small screens.
  const [listOpen, setListOpen] = useState(false);

  const active = useMemo(
    () => chats.find((c) => c.id === activeId) ?? null,
    [chats, activeId],
  );

  // The list is also readable from callbacks without making their identity
  // depend on it (a chats dependency once fed a persist -> setChats -> new
  // persist -> effect loop that hit React's maximum update depth).
  const chatsRef = useRef(chats);
  useEffect(() => {
    chatsRef.current = chats;
  }, [chats]);

  // --- The agentic tool loop's browser side (design 2026-07-24) --------------
  // Per-chat runtime sessions, created lazily on the first tool call and kept
  // for the chat's lifetime so variables persist across turns. Plots stay
  // client-side (keyed by tool call), the model only learns one was produced.
  const sessionsRef = useRef(
    new Map<string, Partial<Record<"python" | "r" | "sql", RunSession>>>(),
  );
  const plotsRef = useRef(new Map<string, string>());
  const stepsRef = useRef(0);
  const runSupported = useRef(isRunSupported());
  // The Pyodide package index enriches import failures with the checker's
  // verdict; fetched once, best-effort (a miss just means plainer errors).
  const pyIndexRef = useRef<PyodideIndex | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetch("/runtimes/pyodide/pyodide-lock.json")
      .then((r) => (r.ok ? r.json() : null))
      .then((lock) => {
        if (!cancelled && lock) pyIndexRef.current = buildPyodideIndex(lock);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);
  // Every session is released when the page unmounts.
  useEffect(() => {
    const sessions = sessionsRef.current;
    return () => {
      for (const byLang of sessions.values())
        for (const s of Object.values(byLang)) s?.dispose();
      sessions.clear();
    };
  }, []);

  const disposeSessionsFor = useCallback((chatId: string) => {
    const byLang = sessionsRef.current.get(chatId);
    if (!byLang) return;
    for (const s of Object.values(byLang)) s?.dispose();
    sessionsRef.current.delete(chatId);
  }, []);

  /** The active chat's session for a language, created on first use. */
  const sessionFor = useCallback(
    (lang: "python" | "r" | "sql"): RunSession => {
      const language = RUNNABLE_LANGUAGES.find((l) => l.id === lang)!;
      const byLang = sessionsRef.current.get(activeId) ?? {};
      const session = (byLang[lang] ??= createSession(language));
      sessionsRef.current.set(activeId, byLang);
      return session;
    },
    [activeId],
  );

  // One execution per tool call id, ever. React dev StrictMode double-invokes
  // the stream handlers, and a session-stateful runtime must not run the same
  // code twice (x = x + 1 would silently double), so repeat calls share the
  // first execution's promise.
  const inflightRef = useRef(new Map<string, Promise<unknown>>());

  const executeToolOnce = useCallback(
    async ({ toolName, toolCallId, input }: ChatToolCall): Promise<unknown> => {
      const lang = TOOL_LANG[toolName as AskToolName];
      if (!lang) return { status: "error", error: `Unknown tool ${toolName}.` };
      if (!runSupported.current) {
        return {
          status: "error",
          error:
            "Code execution is unavailable in this browser. Answer without running code.",
        };
      }
      if (stepsRef.current >= MAX_TOOL_STEPS_PER_TURN) {
        return {
          status: "error",
          error:
            "Tool step limit reached for this turn. Summarize progress and ask the student before continuing.",
        };
      }
      stepsRef.current++;
      const code = (input as { code?: string } | undefined)?.code ?? "";
      const session = sessionFor(lang);

      const started = Date.now();
      const outcome = await session.run(code);
      const ms = Date.now() - started;
      if (!outcome.ok) {
        let message = outcome.error ?? "The run failed.";
        if (lang === "python")
          message = enrichPythonError(message, pyIndexRef.current);
        return { status: "error", error: truncateToolOutput(message).text, ms };
      }
      let text = outcome.result?.text ?? "";
      if (outcome.result?.table) {
        text = text
          ? `${text}\n${tableSample(outcome.result.table)}`
          : tableSample(outcome.result.table);
      }
      let plots = 0;
      if (outcome.result?.imageDataUrl) {
        plotsRef.current.set(toolCallId, outcome.result.imageDataUrl);
        plots = 1;
      }
      const { text: output, truncated } = truncateToolOutput(text);
      return { status: "ok", output, truncated, plots, ms };
    },
    [sessionFor],
  );

  const executeTool = useCallback(
    async (call: ChatToolCall): Promise<unknown> => {
      const existing = inflightRef.current.get(call.toolCallId);
      if (existing) return existing;
      const run = executeToolOnce(call);
      inflightRef.current.set(call.toolCallId, run);
      return run;
    },
    [executeToolOnce],
  );

  const renderToolPart = useCallback(
    (part: Record<string, unknown>, key: string) => (
      <ToolCard
        key={key}
        part={part}
        plotFor={(id) => plotsRef.current.get(id)}
      />
    ),
    [],
  );
  // ---------------------------------------------------------------------------

  // --- Attachments (slice C) -------------------------------------------------
  // Dataset variable names already used in each chat, so a second sales.csv
  // becomes sales_2 rather than clobbering the first.
  const datasetNamesRef = useRef(new Map<string, Set<string>>());

  /** Turns one chosen file into the message parts the composer sends. Throws
   * with student-readable messages; the chip shows them. */
  const prepareAttachment = useCallback(
    async (file: File): Promise<PreparedAttachment> => {
      const refused = rejectionReason(file.name, file.size, file.type);
      if (refused) throw new Error(refused);
      const cls = classifyFile(file.name, file.type);

      if (cls.kind === "image") {
        const { dataUrl, mediaType } = await prepareImage(file);
        return {
          parts: [
            { type: "file", mediaType, url: dataUrl, filename: file.name },
          ],
          detail: "image",
        };
      }

      if (cls.kind === "pdf") {
        const bytes = new Uint8Array(await file.arrayBuffer());
        const pages = estimatePdfPages(bytes);
        if (pages !== null && pages > PDF_PAGE_SOFT_CAP) {
          throw new Error(
            `"${file.name}" looks like about ${pages} pages; models read at most ${PDF_PAGE_SOFT_CAP} here. Attach the chapter or section you need.`,
          );
        }
        return {
          parts: [
            {
              type: "file",
              mediaType: "application/pdf",
              url: await fileToDataUrl(file),
              filename: file.name,
            },
          ],
          detail: pages !== null ? `PDF, about ${pages} pages` : "PDF",
        };
      }

      if (cls.kind === "dataset") {
        if (!runSupported.current) {
          throw new Error(
            "This browser cannot run the data runtime, so datasets cannot be loaded. Attach the file's contents as text instead.",
          );
        }
        const bytes = new Uint8Array(await file.arrayBuffer());
        const taken =
          datasetNamesRef.current.get(activeId) ?? new Set<string>();
        const varName = uniqueName(nameFromFile(file.name), taken);
        const options =
          cls.format === "csv"
            ? {
                skipRows: 0,
                header: true,
                delimiter: detectDelimiter(
                  new TextDecoder().decode(bytes.slice(0, 4096)),
                ),
              }
            : { skipRows: 0, header: true };
        const session = sessionFor("python");
        const preview = await session.previewFile({
          name: varName,
          format: cls.format!,
          bytes,
          options,
        });
        if (preview?.parseError) {
          throw new Error(
            `"${file.name}" could not be parsed: ${preview.parseError}`,
          );
        }
        const outcome = await session.importFile({
          name: varName,
          format: cls.format!,
          bytes,
          options,
        });
        if (!outcome.ok) {
          throw new Error(
            outcome.error ?? `"${file.name}" could not be loaded.`,
          );
        }
        taken.add(varName);
        datasetNamesRef.current.set(activeId, taken);
        return {
          parts: [
            attachmentPart({
              kind: "dataset",
              name: file.name,
              detail: `loaded as ${varName}`,
              text: datasetAnnouncement({
                fileName: file.name,
                varName,
                columns: preview?.columns ?? [],
                rowCount: preview?.totalRows,
              }),
            }),
          ],
          detail: `loaded as ${varName}`,
        };
      }

      if (cls.kind === "office") {
        const raw = await officeTextFromFile(file, cls.office!);
        if (raw.trim().length === 0) {
          throw new Error(`"${file.name}" contained no readable text.`);
        }
        const { text, truncated } = truncateAttachmentText(raw);
        const detail = cls.office === "pptx" ? "slides text" : "document text";
        return {
          parts: [
            attachmentPart({
              kind: "office",
              name: file.name,
              detail,
              text,
              truncated,
            }),
          ],
          detail,
        };
      }

      if (cls.kind === "notebook") {
        const parsed = notebookToText(await file.text(), {
          maxImages: NOTEBOOK_PLOT_CAP,
        });
        if (parsed && parsed.text.trim().length > 0) {
          const { text, truncated } = truncateAttachmentText(parsed.text);
          const cells = `${parsed.cellCount} cell${parsed.cellCount === 1 ? "" : "s"}`;
          const detail =
            parsed.images.length > 0
              ? `${cells}, ${parsed.language}, ${parsed.images.length} plot${parsed.images.length === 1 ? "" : "s"}`
              : `${cells}, ${parsed.language}`;
          const parts: PreparedAttachment["parts"] = [
            attachmentPart({
              kind: "notebook",
              name: file.name,
              detail,
              text,
              truncated,
            }),
          ];
          const stem = file.name.replace(/\.ipynb$/i, "");
          for (const [i, img] of parsed.images.entries()) {
            const ext = img.mediaType === "image/png" ? "png" : "jpg";
            try {
              const plotFile = fileFromBase64(
                img.base64,
                `${stem}-plot-${i + 1}.${ext}`,
                img.mediaType,
              );
              const { dataUrl, mediaType } = await prepareImage(plotFile);
              parts.push({
                type: "file",
                mediaType,
                url: dataUrl,
                filename: plotFile.name,
              });
            } catch {
              // A malformed embedded plot never blocks the notebook itself.
            }
          }
          return { parts, detail };
        }
        // Unparseable notebook: fall through and ride as plain text.
      }

      // Plain text formats.
      const rawText = await file.text();
      const { text, truncated } = truncateAttachmentText(rawText);
      return {
        parts: [
          attachmentPart({
            kind: "text",
            name: file.name,
            detail: "text",
            text,
            truncated,
          }),
        ],
        detail: "text",
      };
    },
    [activeId, sessionFor],
  );

  // Payloads already written to IndexedDB this session (ids are namespaced by
  // chat and message, so one Set serves all chats).
  const storedIdsRef = useRef(new Set<string>());

  /** Storage copy of the transcript: file-part data URLs become aa-file
   * references, and any new payload is written to IndexedDB on the way. */
  const dehydrateForStorage = useCallback(
    (messages: UIMessage[], chatId: string): UIMessage[] => {
      return (messages as unknown as LooseMessage[]).map((message, mi) => {
        const parts = message.parts;
        if (!parts?.some((p) => p.type === "file" && typeof p.url === "string" && (p.url as string).startsWith("data:"))) {
          return message as unknown as UIMessage;
        }
        return {
          ...message,
          parts: parts.map((part, pi) => {
            if (
              part.type !== "file" ||
              typeof part.url !== "string" ||
              !(part.url as string).startsWith("data:")
            ) {
              return part;
            }
            const id = `${chatId}:${message.id ?? mi}:${pi}`;
            if (!storedIdsRef.current.has(id)) {
              storedIdsRef.current.add(id);
              void putFile({
                id,
                chatId,
                name: String(part.filename ?? "attachment"),
                mediaType: String(part.mediaType ?? ""),
                dataUrl: part.url as string,
              });
            }
            return { ...part, url: fileRef(id) };
          }),
        } as unknown as UIMessage;
      });
    },
    [],
  );

  /** Loading copy of the transcript: aa-file references back to data URLs.
   * A missing payload leaves the reference; the bubble shows a chip. */
  const rehydrateMessages = useCallback(
    async (messages: unknown[]): Promise<UIMessage[]> => {
      const out: LooseMessage[] = [];
      for (const raw of messages as LooseMessage[]) {
        const parts = raw.parts;
        if (!parts?.some((p) => isFileRef(p.url))) {
          out.push(raw);
          continue;
        }
        const newParts: MessagePart[] = [];
        for (const part of parts) {
          if (part.type === "file" && isFileRef(part.url)) {
            const stored = await getFile(idFromRef(part.url as string));
            newParts.push(
              stored ? { ...part, url: stored.dataUrl } : part,
            );
          } else {
            newParts.push(part);
          }
        }
        out.push({ ...raw, parts: newParts });
      }
      return out as unknown as UIMessage[];
    },
    [],
  );

  // The transcript handed to Chat: rehydrated per chat switch. Keyed by chat
  // id so a stale hydration never renders under a different chat.
  const [hydrated, setHydrated] = useState<{
    id: string;
    messages: UIMessage[];
  } | null>(null);
  useEffect(() => {
    let cancelled = false;
    const source =
      chatsRef.current.find((c) => c.id === activeId)?.messages ?? [];
    (async () => {
      const messages = await rehydrateMessages(source);
      if (!cancelled) setHydrated({ id: activeId, messages });
    })();
    return () => {
      cancelled = true;
    };
    // Rehydrate on chat SWITCH only (reading messages via chatsRef): persist
    // updates the chat list continuously, and re-running this on every
    // streamed token would fight the live Chat state.
  }, [activeId, rehydrateMessages]);
  // ---------------------------------------------------------------------------

  const persist = useCallback(
    (messages: UIMessage[]) => {
      // A fresh user turn resets the per-turn tool-step budget.
      if (messages[messages.length - 1]?.role === "user") stepsRef.current = 0;
      if (messages.length === 0) return; // never store an empty chat
      const existing = chatsRef.current.find((c) => c.id === activeId);
      const record: StoredChat = {
        id: activeId,
        title: deriveTitle(messages),
        modelId: modelForActive ?? existing?.modelId ?? defaultModelId,
        createdAt: existing?.createdAt ?? Date.now(),
        updatedAt: Date.now(),
        messages: dehydrateForStorage(messages, activeId),
      };
      try {
        const { trimmed } = saveChat(window.localStorage, record);
        if (trimmed) setTrimNote(true);
        // The sidebar re-renders only when it would LOOK different (new chat,
        // new title, or a trim). Persisting runs on every streamed update, and
        // a setState per token both wastes work and, under a provider's burst
        // streaming, nests updates past React's depth limit. The ref still
        // tracks the latest record so switching chats mid-stream stays exact.
        if (!existing || existing.title !== record.title || trimmed) {
          setChats(listChats(window.localStorage));
        } else {
          chatsRef.current = chatsRef.current.map((c) =>
            c.id === record.id ? record : c,
          );
        }
      } catch {
        // Private mode: the conversation still works, it just will not survive.
      }
    },
    [activeId, defaultModelId, modelForActive, dehydrateForStorage],
  );

  const startNew = useCallback(() => {
    setActiveId(crypto.randomUUID());
    setModelForActive(null);
    setListOpen(false);
  }, []);

  const open = useCallback((id: string) => {
    setActiveId(id);
    setModelForActive(null);
    setListOpen(false);
  }, []);

  const remove = useCallback(
    (id: string) => {
      try {
        deleteChat(window.localStorage, id);
        setChats(listChats(window.localStorage));
      } catch {
        // best-effort
      }
      disposeSessionsFor(id); // its runtime state goes with it
      datasetNamesRef.current.delete(id);
      void deleteFilesForChat(id); // its attachment payloads go with it
      if (id === activeId) startNew();
    },
    [activeId, disposeSessionsFor, startNew],
  );

  const sidebar = (
    <nav aria-label="Your chats" className="flex flex-col gap-2">
      <button
        type="button"
        onClick={startNew}
        className="rounded-card border-2 border-miami-red bg-paper px-3 py-2 text-left font-bold text-miami-red hover:bg-light-tan"
      >
        New chat
      </button>
      {chats.length === 0 ? (
        <p className="px-1 text-sm text-dark-tan">
          Chats you start are saved on this device and listed here.
        </p>
      ) : (
        <ul className="flex flex-col gap-1">
          {chats.map((c) => (
            <li key={c.id} className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => open(c.id)}
                aria-current={c.id === activeId ? "page" : undefined}
                className={`min-w-0 flex-1 rounded-card border px-3 py-2 text-left text-sm ${
                  c.id === activeId
                    ? "border-miami-red bg-light-tan font-bold"
                    : "border-medium-tan bg-paper hover:bg-light-tan"
                }`}
              >
                <span className="block truncate">{c.title}</span>
                <span className="block text-xs text-dark-tan">
                  {relativeLabel(c.updatedAt)}
                </span>
              </button>
              <button
                type="button"
                onClick={() => remove(c.id)}
                aria-label={`Delete chat: ${c.title}`}
                title="Delete this chat from this device"
                className="rounded-card border border-medium-tan bg-paper px-2 py-2 text-sm text-dark-tan hover:border-miami-red hover:text-miami-red"
              >
                &times;
              </button>
            </li>
          ))}
        </ul>
      )}
      {trimNote ? (
        <p role="status" className="px-1 text-xs text-dark-tan">
          Device storage was full, so your oldest chat was removed to make room.
        </p>
      ) : null}
    </nav>
  );

  return (
    <div className="flex flex-col gap-4 md:flex-row md:items-start">
      {/* Mobile: the chat list sits behind a disclosure so the conversation leads. */}
      <div className="md:hidden">
        <button
          type="button"
          onClick={() => setListOpen((v) => !v)}
          aria-expanded={listOpen}
          className="rounded-card border border-medium-tan bg-paper px-3 py-2 font-bold"
        >
          {listOpen ? "Hide chats" : `Chats (${chats.length})`}
        </button>
        {listOpen ? <div className="mt-3">{sidebar}</div> : null}
      </div>
      <aside className="hidden w-64 shrink-0 md:block">{sidebar}</aside>

      <div className="min-w-0 flex-1">
        {hydrated?.id === activeId ? (
          <Chat
            key={activeId}
            moduleKey={mod.key}
            moduleName={mod.name}
            placeholder={mod.placeholder}
            models={models}
            defaultModelId={active?.modelId ?? defaultModelId}
            initialMessages={hydrated.messages}
            api="/api/ask-anything"
            onToolCall={executeTool}
            toolRenderer={renderToolPart}
            attachments={{ accept: ATTACH_ACCEPT, prepare: prepareAttachment }}
            onMessagesChange={persist}
            onModelChange={setModelForActive}
            emptyState={{
              heading: "Ask anything",
              body: "Questions, drafts, analysis, research, planning. The assistant can run Python, R, and SQL in your browser, search the academic literature, and read your files. Your conversation stays on this device.",
            }}
          />
        ) : (
          <p className="text-sm text-dark-tan">Opening chat...</p>
        )}
      </div>
    </div>
  );
}
