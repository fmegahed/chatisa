# Ask Anything, Slice A: Chat Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the General Chat placeholder with the Ask Anything chat shell: a two-pane Claude.ai-style page with device-side (localStorage) multi-chat history, a curated 8-model picker defaulting to Claude Sonnet 5, and streaming replies through the existing `/api/chat` route. No tools yet (slices B-E).

**Architecture:** Slice A rides the existing module-driven `/api/chat` route by registering an `ask_anything` module; the tools-bearing sibling route arrives in slice B. The client is a thin shell around the existing `Chat` component: a sidebar managing localStorage chat records, with `Chat` gaining three optional, non-breaking props (`onMessagesChange`, `onModelChange`, `emptyState`) so the shell can persist transcripts and per-chat model choice. Full design: `2026-07-24-ask-anything-design.md`.

**Tech Stack:** Next.js (this repo's vendored build; read `node_modules/next/dist/docs/` before touching framework APIs), React 19, `@ai-sdk/react` `useChat`, Tailwind v4 Miami tokens, Vitest, Playwright + axe.

## Global Constraints

- **No git commits.** The tree stays uncommitted; every task ends by running its checks.
- **Privacy (ADR-022):** chat content lives ONLY in the browser (`localStorage`); the server records only content-free `usage_events`. No new tables, no conversation ids sent to the server.
- **No em dashes** in any user-facing copy. **Miami brand tokens only** (existing classes: `miami-red`, `light-tan`, `medium-tan`, `dark-tan`, `paper`, `rounded-card`, `ribbon`).
- **WCAG 2.1 AA:** native controls, labelled buttons, `aria-current` for the active chat, no color-only state; axe clean at desktop and 320px.
- **Roster:** exactly these 8 model ids (vision + tools + structured output): `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `claude-opus-4-8`, `claude-sonnet-5`, `gemini-3.1-pro-preview-customtools`, `gemini-3.6-flash`, `moonshotai/Kimi-K2.7-Code:together`. Default `claude-sonnet-5`.
- All commands run from `web/`.

## File Structure

- Create: `lib/prompts/ask-anything.ts` (slice-A system prompt), `lib/ask/chat-store.ts` (pure localStorage store), `components/ask/AskAnything.tsx` (shell), `app/(app)/general-chat/page.tsx` (server page), `tests/unit/ask-roster.test.ts`, `tests/unit/ask-chat-store.test.ts`, `tests/e2e/ask-anything.spec.ts`.
- Modify: `lib/chat/config.ts` (register module), `lib/config/models.ts` (`ModuleKey` union + `PAGE_MODELS` + `DEFAULT_MODELS`), `lib/modules.ts` (rename to Ask Anything), `components/chat/Chat.tsx` (three optional props).

---

## Task 1: Module registration and roster

**Files:**
- Create: `lib/prompts/ask-anything.ts`
- Modify: `lib/chat/config.ts`, `lib/config/models.ts`, `lib/modules.ts`
- Test: `tests/unit/ask-roster.test.ts`

**Interfaces:**
- Produces: `CHAT_MODULES.ask_anything` (key `ask_anything`, slug `general-chat`); `getPageModels("ask_anything")` returning the 8 ids; `buildModelOptions("ask_anything", ...)` defaulting to `claude-sonnet-5`; `getModule("general-chat").name === "Ask Anything"`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/ask-roster.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import {
  MODELS,
  getPageModels,
  buildModelOptions,
} from "@/lib/config/models";
import { CHAT_MODULES } from "@/lib/chat/config";
import { getModule } from "@/lib/modules";

const ROSTER = [
  "gpt-5.6-sol",
  "gpt-5.6-terra",
  "gpt-5.6-luna",
  "claude-opus-4-8",
  "claude-sonnet-5",
  "gemini-3.1-pro-preview-customtools",
  "gemini-3.6-flash",
  "moonshotai/Kimi-K2.7-Code:together",
];

describe("Ask Anything roster", () => {
  it("offers exactly the curated 8", () => {
    expect(getPageModels("ask_anything").sort()).toEqual([...ROSTER].sort());
  });

  it("every roster model has vision, tools, and structured output", () => {
    for (const id of ROSTER) {
      const m = MODELS[id];
      expect(m.supportsVision, id).toBe(true);
      expect(m.supportsFunctionCalling, id).toBe(true);
      expect(m.supportsStructuredOutput, id).toBe(true);
    }
  });

  it("defaults to Claude Sonnet 5", () => {
    const { defaultModelId } = buildModelOptions("ask_anything");
    expect(defaultModelId).toBe("claude-sonnet-5");
  });

  it("registers the chat module under the general-chat slug", () => {
    const mod = CHAT_MODULES.ask_anything;
    expect(mod?.key).toBe("ask_anything");
    expect(mod?.slug).toBe("general-chat");
    expect(mod?.name).toBe("Ask Anything");
    expect(mod?.systemPrompt.length).toBeGreaterThan(100);
  });

  it("renames the module tile to Ask Anything", () => {
    expect(getModule("general-chat")?.name).toBe("Ask Anything");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run tests/unit/ask-roster.test.ts`
Expected: FAIL (`getPageModels("ask_anything")` empty, `CHAT_MODULES.ask_anything` undefined).

- [ ] **Step 3: Create the slice-A system prompt**

Create `lib/prompts/ask-anything.ts`:

```typescript
/**
 * Ask Anything (slice A): a general-purpose assistant, provider-agnostic. This
 * prompt deliberately says nothing about tools; slices B-E replace it with the
 * tool contract (browser runtimes, hosted interpreter routing, web hierarchy)
 * from the design doc. No em dashes in any wording a student may see quoted.
 */
export const ASK_ANYTHING_SYSTEM_PROMPT = `You are ChatISA's Ask Anything assistant for Miami University students. You help with any topic: coursework, writing, analysis, planning, and general questions.

Ground rules:
- Be direct and concrete. Lead with the answer, then the reasoning that matters.
- When a question is ambiguous, state the most reasonable interpretation and answer it, noting the assumption in one line.
- Use plain language; define a technical term the first time you use it.
- Show working for quantitative answers, and say so plainly when you are unsure or when a claim needs checking.
- Format with Markdown: short paragraphs, lists where they help, fenced code blocks with a language tag for any code.
- Never invent citations, links, or data. If you do not know, say so.`;
```

- [ ] **Step 4: Register the chat module**

In `lib/chat/config.ts`, add the import at the top with the other prompt imports:

```typescript
import { ASK_ANYTHING_SYSTEM_PROMPT } from "@/lib/prompts/ask-anything";
```

and add to `CHAT_MODULES` after `ai_comparisons`:

```typescript
  ask_anything: {
    key: "ask_anything",
    slug: "general-chat",
    name: "Ask Anything",
    systemPrompt: ASK_ANYTHING_SYSTEM_PROMPT,
    // No fixed opening turn: an open-ended assistant, not a persona session.
    // 0.7 keeps general answers natural (the tutoring modules pin 0 for
    // determinism; this module is closer to AI Comparison's register).
    temperature: 0.7,
    maxOutputTokens: CHAT_OUTPUT_TOKENS,
    placeholder: "Ask anything: a question, a draft to improve, a problem to work through.",
  },
```

- [ ] **Step 5: Register the roster and default**

In `lib/config/models.ts`:

1. Add `"ask_anything"` to the `ModuleKey` union (line ~387, after `"ai_comparisons"`).
2. Add to `DEFAULT_MODELS` (line ~389):

```typescript
  ask_anything: "claude-sonnet-5",
```

3. Add to `PAGE_MODELS` after `ai_comparisons` (curated list per the design; vision + tools + structured output, verified by the unit test):

```typescript
  // Ask Anything: curated vision+tools+structured roster (design 2026-07-24).
  ask_anything: {
    specificModels: [
      "gpt-5.6-sol",
      "gpt-5.6-terra",
      "gpt-5.6-luna",
      "claude-opus-4-8",
      "claude-sonnet-5",
      "gemini-3.1-pro-preview-customtools",
      "gemini-3.6-flash",
      "moonshotai/Kimi-K2.7-Code:together",
    ],
  },
```

- [ ] **Step 6: Rename the module tile**

In `lib/modules.ts`, replace the `general-chat` entry's `name` and `description`:

```typescript
  {
    slug: "general-chat",
    name: "Ask Anything",
    description:
      "Chat with the frontier model of your choice. Your chats stay on your device.",
    group: "general",
  },
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `npx vitest run tests/unit/ask-roster.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 8: Verify the task (no commit)**

Run: `npx tsc --noEmit` and `npx eslint lib/prompts/ask-anything.ts lib/chat/config.ts lib/config/models.ts lib/modules.ts` — both clean.
Also run: `npx vitest run` — the full unit suite stays green (the shell e2e asserting the old "General Chat" name is updated in Task 5).

---

## Task 2: The localStorage chat store

**Files:**
- Create: `lib/ask/chat-store.ts`
- Test: `tests/unit/ask-chat-store.test.ts`

**Interfaces:**
- Produces (consumed by Task 4):

```typescript
export interface StoredChat {
  id: string;
  title: string;
  modelId: string;
  createdAt: number;
  updatedAt: number;
  messages: unknown[]; // UIMessage[]; stored opaquely
}
export const ASK_CHATS_KEY = "aa-chats-v1";
export function listChats(storage: Storage): StoredChat[];            // newest updated first
export function saveChat(storage: Storage, chat: StoredChat): { trimmed: boolean };
export function deleteChat(storage: Storage, id: string): void;
export function deriveTitle(messages: unknown[]): string;             // first user text, <=60 chars, "New chat" fallback
```

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/ask-chat-store.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import {
  ASK_CHATS_KEY,
  deleteChat,
  deriveTitle,
  listChats,
  saveChat,
  type StoredChat,
} from "@/lib/ask/chat-store";

/** Minimal in-memory Storage for tests. */
function memoryStorage(): Storage {
  const map = new Map<string, string>();
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k) => map.get(k) ?? null,
    key: (i) => [...map.keys()][i] ?? null,
    removeItem: (k) => void map.delete(k),
    setItem: (k, v) => void map.set(k, v),
  } as Storage;
}

function chat(id: string, updatedAt: number, text = "Hello"): StoredChat {
  return {
    id,
    title: text,
    modelId: "claude-sonnet-5",
    createdAt: updatedAt,
    updatedAt,
    messages: [{ id: "m1", role: "user", parts: [{ type: "text", text }] }],
  };
}

describe("ask chat store", () => {
  it("saves, lists newest-first, and deletes", () => {
    const s = memoryStorage();
    saveChat(s, chat("a", 1));
    saveChat(s, chat("b", 2));
    expect(listChats(s).map((c) => c.id)).toEqual(["b", "a"]);
    deleteChat(s, "b");
    expect(listChats(s).map((c) => c.id)).toEqual(["a"]);
  });

  it("upserts by id, bumping order by updatedAt", () => {
    const s = memoryStorage();
    saveChat(s, chat("a", 1));
    saveChat(s, chat("b", 2));
    saveChat(s, { ...chat("a", 3), title: "Edited" });
    const chats = listChats(s);
    expect(chats.map((c) => c.id)).toEqual(["a", "b"]);
    expect(chats[0].title).toBe("Edited");
    expect(chats).toHaveLength(2);
  });

  it("derives the title from the first user text part", () => {
    expect(
      deriveTitle([
        { id: "m1", role: "user", parts: [{ type: "text", text: "  Explain p-values please  " }] },
      ]),
    ).toBe("Explain p-values please");
    expect(deriveTitle([])).toBe("New chat");
    // Longer than 60 chars is cut with an ellipsis character not an em dash.
    const long = "x".repeat(80);
    const t = deriveTitle([{ id: "m", role: "user", parts: [{ type: "text", text: long }] }]);
    expect(t.length).toBeLessThanOrEqual(61);
  });

  it("survives corrupted storage by starting fresh", () => {
    const s = memoryStorage();
    s.setItem(ASK_CHATS_KEY, "{not json");
    expect(listChats(s)).toEqual([]);
    saveChat(s, chat("a", 1));
    expect(listChats(s)).toHaveLength(1);
  });

  it("trims the oldest chats when over budget and reports it", () => {
    const s = memoryStorage();
    const big = "y".repeat(2_000_000); // ~2MB of message text
    saveChat(s, { ...chat("old", 1), messages: [{ id: "m", role: "user", parts: [{ type: "text", text: big }] }] });
    saveChat(s, { ...chat("mid", 2), messages: [{ id: "m", role: "user", parts: [{ type: "text", text: big }] }] });
    const res = saveChat(s, { ...chat("new", 3), messages: [{ id: "m", role: "user", parts: [{ type: "text", text: big }] }] });
    expect(res.trimmed).toBe(true);
    const ids = listChats(s).map((c) => c.id);
    expect(ids).toContain("new"); // the just-saved chat is never trimmed away
    expect(ids).not.toContain("old"); // oldest goes first
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run tests/unit/ask-chat-store.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the store**

Create `lib/ask/chat-store.ts`:

```typescript
/**
 * Device-side chat history for Ask Anything. Everything lives in localStorage
 * (ADR-022: conversation content is never persisted server-side), one versioned
 * key holding every chat. Functions take the Storage explicitly so unit tests
 * inject an in-memory one; callers pass window.localStorage.
 */

export interface StoredChat {
  id: string;
  title: string;
  modelId: string;
  createdAt: number;
  updatedAt: number;
  /** UIMessage[] from the AI SDK, stored opaquely (shape owned by the SDK). */
  messages: unknown[];
}

export const ASK_CHATS_KEY = "aa-chats-v1";

/** Keep the store safely under the ~5MB localStorage quota. */
const BUDGET_CHARS = 3_500_000;

function readAll(storage: Storage): StoredChat[] {
  try {
    const raw = storage.getItem(ASK_CHATS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as { chats?: StoredChat[] };
    return Array.isArray(parsed.chats) ? parsed.chats : [];
  } catch {
    return []; // corrupted or unavailable: start fresh rather than crash
  }
}

function writeAll(storage: Storage, chats: StoredChat[]): void {
  storage.setItem(ASK_CHATS_KEY, JSON.stringify({ chats }));
}

/** Every chat, newest updated first. */
export function listChats(storage: Storage): StoredChat[] {
  return readAll(storage).sort((a, b) => b.updatedAt - a.updatedAt);
}

/**
 * Inserts or replaces by id. When the serialized store would exceed the budget,
 * the oldest chats (by updatedAt) are dropped first; the chat being saved is
 * never dropped. Returns whether anything was trimmed so the UI can say so.
 */
export function saveChat(storage: Storage, chat: StoredChat): { trimmed: boolean } {
  let chats = readAll(storage).filter((c) => c.id !== chat.id);
  chats.push(chat);
  let trimmed = false;
  // Serialize-and-measure; drop oldest until the payload fits.
  for (;;) {
    const raw = JSON.stringify({ chats });
    if (raw.length <= BUDGET_CHARS || chats.length <= 1) break;
    const oldest = [...chats]
      .filter((c) => c.id !== chat.id)
      .sort((a, b) => a.updatedAt - b.updatedAt)[0];
    if (!oldest) break;
    chats = chats.filter((c) => c.id !== oldest.id);
    trimmed = true;
  }
  try {
    writeAll(storage, chats);
  } catch {
    // Quota hit despite the budget (other keys share it): drop oldest and retry once.
    const survivor = chats.filter((c) => c.id === chat.id);
    try {
      writeAll(storage, survivor);
      trimmed = true;
    } catch {
      // Storage unusable (private mode); the in-memory session still works.
    }
  }
  return { trimmed };
}

export function deleteChat(storage: Storage, id: string): void {
  writeAll(storage, readAll(storage).filter((c) => c.id !== id));
}

/** The first user text, tidied and capped, as the sidebar title. */
export function deriveTitle(messages: unknown[]): string {
  for (const m of messages as { role?: string; parts?: { type?: string; text?: string }[] }[]) {
    if (m?.role !== "user") continue;
    const text = (m.parts ?? [])
      .filter((p) => p?.type === "text" && typeof p.text === "string")
      .map((p) => p.text as string)
      .join(" ")
      .trim();
    if (text) return text.length > 60 ? `${text.slice(0, 60)}…` : text;
  }
  return "New chat";
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run tests/unit/ask-chat-store.test.ts`
Expected: PASS (5 tests).

- [ ] **Step 5: Verify the task (no commit)**

Run: `npx tsc --noEmit` and `npx eslint lib/ask/chat-store.ts tests/unit/ask-chat-store.test.ts` — clean.

---

## Task 3: Non-breaking Chat component props

**Files:**
- Modify: `components/chat/Chat.tsx`

**Interfaces:**
- Produces (consumed by Task 4): `ChatProps` gains
  `onMessagesChange?: (messages: UIMessage[]) => void;`
  `onModelChange?: (modelId: string) => void;`
  `emptyState?: { heading: string; body: string };`
  Existing pages (Coding Tutor) pass none of these and behave identically.

- [ ] **Step 1: Extend the props interface**

In `components/chat/Chat.tsx`, extend `ChatProps`:

```typescript
interface ChatProps {
  moduleKey: string;
  moduleName: string;
  placeholder: string;
  models: ChatModelOption[];
  defaultModelId: string;
  initialMessages?: UIMessage[];
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
```

- [ ] **Step 2: Wire the callbacks**

In the `Chat` function body (after the `useChat` destructure), add:

```typescript
  const { onMessagesChange } = props;
  useEffect(() => {
    onMessagesChange?.(messages);
  }, [messages, onMessagesChange]);
```

(Adjust the component to destructure `props` accordingly, or add `onMessagesChange` to the existing destructured parameters and list it in the effect deps.)

Where the model is changed, call through:

```typescript
  <ModelChooser
    options={models}
    value={modelId}
    onChange={(id) => {
      setModelId(id);
      onModelChange?.(id);
    }}
    help="Switching applies to your next message. Earlier replies stay as they are."
  />
```

And replace the empty-state block:

```tsx
  {messages.length === 0 ? (
    <div className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-xl">{emptyState?.heading ?? "Start the conversation"}</h2>
      <p className="mt-2">
        {emptyState?.body ??
          "Ask a question about your code or an analytics concept. Answers include examples in both R and Python."}
      </p>
    </div>
  ) : null}
```

- [ ] **Step 3: Verify nothing regressed**

Run: `npx tsc --noEmit` and `npx eslint components/chat/Chat.tsx` — clean.
Run: `npx playwright test tests/e2e/chat.spec.ts --project=desktop` — the Coding Tutor suite stays green (props unused there).

---

## Task 4: The Ask Anything page and shell

**Files:**
- Create: `app/(app)/general-chat/page.tsx`, `components/ask/AskAnything.tsx`

**Interfaces:**
- Consumes: `CHAT_MODULES.ask_anything`, `buildModelOptions("ask_anything", available)` (Task 1); `listChats`/`saveChat`/`deleteChat`/`deriveTitle`/`StoredChat` (Task 2); `Chat` with the three new props (Task 3).
- Produces: the `/general-chat` route (a real segment overrides the `[module]` placeholder).

- [ ] **Step 1: Server page**

Create `app/(app)/general-chat/page.tsx` (mirrors `coding-companion/page.tsx`):

```tsx
import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { AskAnything } from "@/components/ask/AskAnything";
import { CHAT_MODULES } from "@/lib/chat/config";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";

export const metadata: Metadata = { title: "Ask Anything" };

export default async function AskAnythingPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const mod = CHAT_MODULES.ask_anything;
  const available = filterAvailableModels(getPageModels(mod.key));
  const { options, defaultModelId } = buildModelOptions(mod.key, available);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">{mod.name}</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        A general assistant with the frontier model of your choice. Your chats
        are saved on this device only, never on a server.
      </p>

      {options.length === 0 ? (
        <div role="status" className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5">
          <h2 className="font-bold text-miami-red">No models are available</h2>
          <p className="mt-1">
            This server has no AI provider configured yet. Contact the ChatISA
            maintainers.
          </p>
        </div>
      ) : (
        <div className="mt-8">
          <AskAnything models={options} defaultModelId={defaultModelId} />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: The shell component**

Create `components/ask/AskAnything.tsx`:

```tsx
"use client";

import { useCallback, useMemo, useState } from "react";
import type { UIMessage } from "ai";
import { Chat } from "@/components/chat/Chat";
import type { ModelOption } from "@/lib/config/models";
import { CHAT_MODULES } from "@/lib/chat/config";
import {
  deleteChat,
  deriveTitle,
  listChats,
  saveChat,
  type StoredChat,
} from "@/lib/ask/chat-store";

/** A short relative label ("just now", "3h ago", "Jul 20") for the sidebar. */
function relativeLabel(ts: number): string {
  const mins = Math.round((Date.now() - ts) / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return new Date(ts).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/**
 * The Ask Anything shell: a sidebar of device-stored chats around the shared
 * Chat component. A brand-new chat is held in memory and only written to
 * localStorage once it has a message, so abandoned "New chat" clicks never
 * litter the list. The Chat component is keyed by chat id, so switching chats
 * remounts it with that chat's transcript and model.
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

  const persist = useCallback(
    (messages: UIMessage[]) => {
      if (messages.length === 0) return; // never store an empty chat
      const existing = chats.find((c) => c.id === activeId);
      const record: StoredChat = {
        id: activeId,
        title: deriveTitle(messages),
        modelId: modelForActive ?? existing?.modelId ?? defaultModelId,
        createdAt: existing?.createdAt ?? Date.now(),
        updatedAt: Date.now(),
        messages,
      };
      try {
        const { trimmed } = saveChat(window.localStorage, record);
        if (trimmed) setTrimNote(true);
        setChats(listChats(window.localStorage));
      } catch {
        // Private mode: the conversation still works, it just will not survive.
      }
    },
    [activeId, chats, defaultModelId, modelForActive],
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
      if (id === activeId) startNew();
    },
    [activeId, startNew],
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
        <Chat
          key={activeId}
          moduleKey={mod.key}
          moduleName={mod.name}
          placeholder={mod.placeholder}
          models={models}
          defaultModelId={active?.modelId ?? defaultModelId}
          initialMessages={(active?.messages as UIMessage[]) ?? []}
          onMessagesChange={persist}
          onModelChange={setModelForActive}
          emptyState={{
            heading: "Ask anything",
            body: "Questions, drafts, analysis, planning. Pick a model above; your conversation stays on this device.",
          }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify it compiles and renders**

Run: `npx tsc --noEmit` and `npx eslint app/\(app\)/general-chat/page.tsx components/ask/AskAnything.tsx` — clean.
Run: `npx playwright test tests/e2e/shell.spec.ts --project=desktop` — expect ONE failure: the shell spec still asserts the old "General Chat" name. Task 5 updates it; do not "fix" the module name back.

---

## Task 5: End-to-end tests and gates

**Files:**
- Create: `tests/e2e/ask-anything.spec.ts`
- Modify: `tests/e2e/shell.spec.ts` (the module-name assertions: "General Chat" becomes "Ask Anything"; find them with `grep -n "General Chat" tests/e2e/shell.spec.ts` and update each to the new name)

**Interfaces:**
- Consumes: the mock LLM (`CHATISA_MOCK_LLM=1`, always on under Playwright). The mock's reply to a message containing "SQL" includes a fenced SQL block whose text contains `SELECT 1 AS n` (the chat.spec.ts Customize test relies on the same fixture).

- [ ] **Step 1: Write the e2e spec**

Create `tests/e2e/ask-anything.spec.ts`:

```typescript
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.describe("Ask Anything", () => {
  test("streams a reply and saves the chat on the device", async ({ page }) => {
    await page.goto("/general-chat");
    await expect(
      page.getByRole("heading", { level: 1, name: "Ask Anything" }),
    ).toBeVisible();

    // The curated roster: 8 options, Sonnet 5 preselected.
    const chooser = page.getByLabel(/model/i).first();
    await expect(chooser).toBeVisible();

    await page.getByLabel("Your message").fill("Show me some SQL");
    await page.getByRole("button", { name: "Send message" }).click();
    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("SELECT 1 AS n", { timeout: 15_000 });

    // The sidebar lists the chat, titled from the first message.
    await expect(
      page.getByRole("navigation", { name: "Your chats" }).getByText("Show me some SQL"),
    ).toBeVisible();

    // Stored on the device only.
    const stored = await page.evaluate(() => window.localStorage.getItem("aa-chats-v1"));
    expect(stored).toContain("Show me some SQL");
  });

  test("multiple chats: new, switch, persist across reload, delete", async ({ page }) => {
    await page.goto("/general-chat");
    await page.getByLabel("Your message").fill("First conversation topic");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toBeVisible({
      timeout: 15_000,
    });

    await page.getByRole("button", { name: "New chat" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toHaveCount(0);
    await page.getByLabel("Your message").fill("Second conversation topic");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toBeVisible({
      timeout: 15_000,
    });

    // Both listed; switching restores the first transcript.
    const nav = page.getByRole("navigation", { name: "Your chats" });
    await nav.getByText("First conversation topic").click();
    await expect(
      page.getByRole("article", { name: "You" }).getByText("First conversation topic"),
    ).toBeVisible();

    // Reload: both chats survive (device storage), first is still openable.
    await page.reload();
    await expect(nav.getByText("First conversation topic")).toBeVisible();
    await expect(nav.getByText("Second conversation topic")).toBeVisible();

    // Delete the second; it leaves the list.
    await page
      .getByRole("button", { name: "Delete chat: Second conversation topic" })
      .click();
    await expect(nav.getByText("Second conversation topic")).toHaveCount(0);
  });

  test("is axe-clean, including the sidebar", async ({ page }) => {
    await page.goto("/general-chat");
    await page.getByLabel("Your message").fill("Accessibility pass");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toBeVisible({
      timeout: 15_000,
    });
    const axe = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });
});
```

- [ ] **Step 2: Update the shell spec's module name**

Run `grep -n "General Chat" tests/e2e/shell.spec.ts` and change each match to `Ask Anything` (nav label, home card, and any placeholder-page assertion; the slug stays `general-chat`).

- [ ] **Step 3: Run the new spec**

Run: `npx playwright test tests/e2e/ask-anything.spec.ts --project=desktop`
Expected: 3 passed. Then `--project=mobile-320` (the sidebar collapses behind the "Chats" toggle; the flows still pass by opening it first — if the nav is hidden on mobile, open it with `page.getByRole("button", { name: /Chats/ }).click()` before sidebar assertions; fold that into the tests with a small `openSidebarIfMobile` helper).

- [ ] **Step 4: Full gates (no commit)**

Run, all green:
- `npx tsc --noEmit`
- `npm run lint`
- `npx vitest run` (543 + the ~10 new)
- `npx playwright test` (full suite incl. shell + ask-anything, desktop + mobile-320)

- [ ] **Step 5: Record**

Append the migration-log entry (docs/development/migration-log.md) describing slice A: registration, store, shell, tests, gate numbers. Leave the tree uncommitted.

---

## Self-Review

- **Spec coverage (slice A only):** two-pane shell (Task 4), localStorage multi-chat with delete + trim note (Tasks 2, 4), curated 8 + Sonnet 5 default (Task 1), streaming via existing route (Task 4 uses `Chat` with `module: ask_anything`), placeholder replaced (Task 4 page + Task 5 shell-spec rename). Tools, files, search, hosted interpreter: slices B-E by design.
- **Placeholders:** none; every step has full code or an exact command.
- **Type consistency:** `StoredChat`/`listChats`/`saveChat`/`deleteChat`/`deriveTitle` (Task 2) match Task 4's imports; `onMessagesChange`/`onModelChange`/`emptyState` (Task 3) match Task 4's usage; roster ids in Task 1 match the test and `PAGE_MODELS`.
- **Known seam:** `Chat` posts to `/api/chat`; slice B introduces `/api/ask-anything` (tools) and switches the transport via a new optional `api` prop.
