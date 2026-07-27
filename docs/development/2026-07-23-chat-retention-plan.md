# Chat Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop persisting chat conversation content server-side. Remove the `conversations` and `messages` tables, their data layer, and the unused list endpoint, so the only record of a chat is the content-free `usage_events` row.

**Architecture:** The two chat modules (`coding_companion`, `sandbox_chat`) route through `/api/chat`, which currently writes every turn to `messages` and a title to `conversations`. This plan strips those writes, deletes the callers first (client, route, data layer, endpoint), then drops the tables via a migration. The client keeps its in-session `conversationId` for its own UI threading but no longer sends it. Nothing reads stored history to build model context, so streaming and answer quality are unchanged.

**Tech Stack:** Next.js 16 (App Router), TypeScript, Drizzle ORM + better-sqlite3, AI SDK v7, Vitest, Playwright.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate, not committing. (There is no git repo above the project, so this is automatic.)
- **No secrets in the client;** env var names only, never values.
- **No em dashes in any user-facing text.**
- **This is a customized Next.js.** Follow existing patterns; read `node_modules/next/dist/docs/` before writing framework code.
- **Sequencing:** run this plan only after the Project Assistant Plan 1 execution has finished, because both edit shared gates (`npm run typecheck`, the full test suites) and this plan edits `lib/db/schema.ts`. Do not run the two implementations concurrently.
- **Emails and the analytics table are unchanged.** `usage_events` (module, event type, model, token counts, cost, latency, prompt/response character lengths, outcome) is retained exactly as is. It is the usage-statistics surface and contains no content.

---

## File Structure

**Modified:**
- `lib/chat/config.ts` — drop `conversationId` from `chatRequestSchema`.
- `components/chat/Chat.tsx` — stop creating/sending `conversationId`.
- `components/sandbox/SandboxChat.tsx` — stop creating/sending `conversationId`.
- `app/api/chat/route.ts` — remove all persistence (imports, conversation resolution, both `appendMessage` writes, the `x-conversation-id` header). Keep `recordUsageEvent` and `lastUserText` (used for the content-free char count).
- `lib/db/index.ts` — remove the conversation/message data-layer functions.
- `lib/db/schema.ts` — remove the `conversations` and `messages` table definitions.
- `tests/e2e/chat.spec.ts` — remove the two persistence tests.
- `webapp/docs/development/decision-log.md` and `migration-log.md` — record the reversal.

**Deleted:**
- `app/api/conversations/route.ts` (unused list endpoint).

**Created:**
- `drizzle/00NN_*.sql` — generated drop migration (next number after the current head).
- `tests/unit/chat-retention.test.ts` — guard test.

**Verified call sites (grep, 2026-07-23):** the removed data-layer functions are imported only by `app/api/chat/route.ts` and `app/api/conversations/route.ts`. Nothing reads the `x-conversation-id` response header (the clients generate their own id). `tests/e2e/chat.spec.ts` has exactly two tests that assert persistence.

---

### Task 1: Guard test (acceptance for the whole slice)

This test asserts the end state: no tables, no functions. It fails now and turns green at Task 5. Intermediate tasks are gated on `typecheck`/`lint`.

**Files:**
- Create: `tests/unit/chat-retention.test.ts`

- [ ] **Step 1: Write the test**

```ts
// tests/unit/chat-retention.test.ts
import { describe, expect, it } from "vitest";
import * as schema from "@/lib/db/schema";
import * as db from "@/lib/db";

describe("chat content is not persisted", () => {
  it("has no conversations or messages tables in the schema", () => {
    expect("conversations" in schema).toBe(false);
    expect("messages" in schema).toBe(false);
  });

  it("exposes no conversation data-layer functions", () => {
    const removed = [
      "createConversation",
      "claimConversation",
      "getOwnedConversation",
      "listConversations",
      "listMessages",
      "appendMessage",
      "deleteConversation",
      "deriveTitle",
    ];
    for (const name of removed) {
      expect(name in db).toBe(false);
    }
  });
});
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `npx vitest run tests/unit/chat-retention.test.ts`
Expected: FAIL (both tables and all functions still exist). This is the target state to reach.

---

### Task 2: Remove `conversationId` from the request contract and clients

**Files:**
- Modify: `lib/chat/config.ts`
- Modify: `components/chat/Chat.tsx`
- Modify: `components/sandbox/SandboxChat.tsx`

- [ ] **Step 1: Drop it from the schema**

In `lib/chat/config.ts`, remove the `conversationId` field from `chatRequestSchema` (the `conversationId: z.uuid().optional(),` line). Leave the surrounding `context`, `module`, `modelId`, and `messages` fields intact. `ChatRequest` (the inferred type) updates automatically.

- [ ] **Step 2: Stop sending it from `Chat.tsx`**

In `components/chat/Chat.tsx`:
- Remove the `conversationId` state declaration (the `const [conversationId] = useState(...)` block that generates a `crypto.randomUUID`).
- Change the transport body from `return { body: { module: moduleKey, modelId, conversationId } };` to `return { body: { module: moduleKey, modelId } };`.
- Remove the now-unused `randomUUID`/`crypto` helper if it was only used for that id (check: if the helper function is referenced nowhere else, delete it; otherwise leave it).

- [ ] **Step 3: Stop sending it from `SandboxChat.tsx`**

In `components/sandbox/SandboxChat.tsx`:
- Remove `const [conversationId] = useState(() => crypto.randomUUID());`.
- Remove the `conversationId,` line from the transport `body` object. Leave `module`, `modelId`, and `context` in place.

- [ ] **Step 4: Checkpoint**

Run: `npm run typecheck`
Expected: this now fails inside `app/api/chat/route.ts` (it still destructures and uses `conversationId`). That is expected and is fixed in Task 3. If it fails anywhere else, investigate before continuing.

---

### Task 3: Strip persistence from the chat route

**Files:**
- Modify: `app/api/chat/route.ts`

- [ ] **Step 1: Trim the `@/lib/db` import**

Change the import that currently pulls `appendMessage`, `claimConversation`, `createConversation`, `deriveTitle`, and `recordUsageEvent` so it imports only what remains:

```ts
import { recordUsageEvent } from "@/lib/db";
```

- [ ] **Step 2: Remove `conversationId` from the request destructuring**

In the block that destructures the validated request body, remove the `conversationId,` line. Keep `module` (as `moduleKey`), `modelId`, `context`, and `messages`.

- [ ] **Step 3: Remove conversation resolution and the user-turn write**

Keep the `lastUserText` computation (it feeds the content-free `promptChars`). Delete everything from `const activeConversationId = conversationId` through the user-turn `appendMessage({ ... })` call, including the `if (!activeConversationId) { ... 404 ... }` guard. The region should read:

```ts
  // The last user turn's text is used only for the content-free usage event
  // (its length), never stored.
  const lastUserText = textFromParts(
    [...messages].reverse().find((m) => m.role === "user")?.parts,
  );

  const modelConfig = MODELS[modelId];
  const startedAt = Date.now();
```

- [ ] **Step 4: Remove the assistant-turn write in `onFinish`**

Keep the empty-response handling; delete only the `else { appendMessage({ ... }) }` branch. The block should read:

```ts
        const visible = text.trim();
        if (visible === "") {
          emptyExplanation = describeEmptyResponse(
            finishReason === "length"
              ? "truncated_before_text"
              : "no_text_returned",
          );
          logger.warn(
            {
              requestId,
              module: moduleKey,
              modelId,
              finishReason,
              outputTokens,
            },
            "model returned no visible text",
          );
        }
        recordUsageEvent({
```

The truncation notice that the deleted branch appended to stored content is still shown to the student: it is delivered through `messageMetadata` (the `TRUNCATION_NOTICE` branch in `toUIMessageStreamResponse`), so nothing user-visible is lost.

- [ ] **Step 5: Remove the `x-conversation-id` response header**

In the `return result.toUIMessageStreamResponse({ ... })` call, delete the `headers: { "x-conversation-id": activeConversationId },` property. Leave `messageMetadata` and `onError` intact. (`activeConversationId` no longer exists, so this must go; no client reads the header.)

- [ ] **Step 6: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: `route.ts` is clean. `lib/db/index.ts` and `app/api/conversations/route.ts` still compile (their functions exist until Task 4). If `context` or `textFromParts` is now flagged unused, that is a real signal you removed too much; restore only what the remaining logic needs.

---

### Task 4: Remove the data-layer functions and the list endpoint

**Files:**
- Modify: `lib/db/index.ts`
- Delete: `app/api/conversations/route.ts`

- [ ] **Step 1: Delete the list endpoint**

Remove `app/api/conversations/route.ts`. If `app/api/conversations/` is then empty, remove the directory too.

- [ ] **Step 2: Remove the conversation data-layer functions**

In `lib/db/index.ts`, delete the "conversations" section: `deriveTitle`, `createConversation`, `claimConversation`, `getOwnedConversation`, `listConversations`, `listMessages`, `appendMessage`, and `deleteConversation`, along with the `// ---- conversations` section comment. Do not touch the exam, interview, jobapp, analytics, or `users` functions, and leave the drizzle imports (`and`, `desc`, `eq`, `isNull`, `lt`, `sql`) and `randomUUID` in place: they are still used by the remaining functions.

- [ ] **Step 3: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean. The tables still exist in `schema.ts` (now unused), which is fine; typecheck passes. If any file still imports a removed function, remove that usage (grep confirmed there are none outside the two files handled here).

---

### Task 5: Remove the tables and generate the drop migration

**Files:**
- Modify: `lib/db/schema.ts`
- Create: `drizzle/00NN_*.sql` (generated)

- [ ] **Step 1: Delete the table definitions**

In `lib/db/schema.ts`, remove the `conversations` and `messages` `sqliteTable(...)` definitions and their doc comments. Leave `users` and every other table.

- [ ] **Step 2: Generate the migration**

Run: `npx drizzle-kit generate`
Expected: a new `drizzle/00NN_*.sql` (the next number after the current head) containing `DROP TABLE` for `messages` and `conversations`. Read it and confirm it drops exactly those two tables (messages first, for the foreign key) and alters nothing else.

- [ ] **Step 3: Confirm it applies on a fresh database**

Run (bash): `CHATISA_DATA_DIR=$(mktemp -d) npx tsx -e "import('@/lib/db').then(m => { m.dbReady(); console.log('migrated ok'); m.closeDb(); })"`
(PowerShell: set `$env:CHATISA_DATA_DIR` to a fresh temp dir first, then run the `npx tsx -e` command.)
Expected: prints `migrated ok` with no error.

- [ ] **Step 4: Run the guard test**

Run: `npx vitest run tests/unit/chat-retention.test.ts`
Expected: PASS (both tests). The end state is reached.

- [ ] **Step 5: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean.

---

### Task 6: Update the chat e2e suite

**Files:**
- Modify: `tests/e2e/chat.spec.ts`

- [ ] **Step 1: Remove the two persistence tests**

Delete the test titled `"conversation persists across a page reload"` (it fetches `/api/conversations`, which no longer exists) and the test titled `"rejects another user's conversation id as not found"` (it sends a `conversationId`, no longer part of the contract). Leave every other test, including the streaming test and the axe scan (the word "conversation" in the axe test title and in visible copy like "Start the conversation" is incidental UI text, not persistence; keep it).

- [ ] **Step 2: Run the chat e2e**

Run: `npm run test:e2e -- chat`
Expected: PASS. The chat still streams and renders; only the two persistence assertions are gone.

- [ ] **Step 3: Checkpoint** — working tree stays uncommitted.

---

### Task 7: Docs and full gate

**Files:**
- Modify: `webapp/docs/development/decision-log.md`
- Modify: `webapp/docs/development/migration-log.md`

- [ ] **Step 1: Record the decision**

In `decision-log.md`, add an entry (matching the file's existing ADR style) that supersedes ADR-002: chat conversation content is no longer persisted; rationale (a student's non-coursework or sensitive content should not accumulate; scheduled backups make delete-later insufficient, so the content is never written); consequence (no chat-resume feature; usage statistics retained in content-free form). Reference the design spec `2026-07-23-chat-retention-design.md`.

- [ ] **Step 2: Record the change**

In `migration-log.md`, add a dated `### YYYY-MM-DD —` entry summarizing: `conversations` and `messages` removed, the drop migration number, the route/client/data-layer edits, the deleted list endpoint, and that `usage_events` is retained unchanged.

- [ ] **Step 3: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: all green. Record real counts and any skips honestly. Confirm the exam, interview, and jobapp specs still pass (their persistence was untouched).

- [ ] **Step 4: Backups note (no action, record only)**

Confirm in the migration-log entry that this change does not scrub backups already taken; historical chat content ages out on the existing backup rotation, which is a separate operations concern outside this app and its access.

---

## Self-Review

**1. Spec coverage (against `2026-07-23-chat-retention-design.md`):**
- Principle (write nothing): Tasks 3 to 5 remove every write path and the tables. Covered.
- Scope (chat modules only): only `/api/chat`, its clients, and its tables are touched; exam/interview/jobapp/Project Assistant untouched. Covered.
- Remove tables (Section 4): Task 5. Covered.
- Server/schema/data-layer/endpoint/client changes (Section 5): Tasks 2 to 5, each a named file. Covered.
- Keep `usage_events` (Section 6): explicitly retained; `lastUserText` and `recordUsageEvent` preserved in Task 3. Covered.
- Backups caveat (Section 7): recorded in Task 7 Step 4, no code action. Covered.
- Testing (Section 9): guard test (Task 1/5), regression e2e (Task 6), fresh-DB migration check (Task 5). Covered.
- Docs/ADR (Section 10): Task 7. Covered.

**2. Placeholder scan:** No "TBD"/"handle edge cases". `00NN` is a real instruction ("next number after the current head"), resolved by `drizzle-kit generate`, not a logic placeholder. The one conditional ("delete the randomUUID helper if unused") states the exact test and both branches.

**3. Type/name consistency:** `conversationId` is removed in the same form everywhere it appears (schema field, route destructure, both client bodies). `lastUserText`, `recordUsageEvent`, `textFromParts`, `describeEmptyResponse`, `TRUNCATION_NOTICE` are all kept and referenced consistently with the current route. The removed-function list in the guard test (Task 1) matches exactly the functions deleted in Task 4.

---

## Execution Handoff

**Plan saved to `webapp/docs/development/2026-07-23-chat-retention-plan.md`. Run it only after the Project Assistant Plan 1 execution has settled (shared gates and `lib/db/schema.ts`). Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute the tasks here with checkpoints.

Which approach would you like, and should I proceed once Plan 1 execution reports back?
