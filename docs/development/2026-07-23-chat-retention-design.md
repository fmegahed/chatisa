# Chat retention — design spec

Date: 2026-07-23. Status: approved in brainstorming, pending user review of this
document before an implementation plan is written.

Standing constraints apply: no git commits, no production access, no secrets in
the client, WCAG 2.1 AA, Miami brand tokens, and no em dashes in user-facing
text.

## 1. Problem

Across all modules, the only place raw student free-text accumulates on the
server with no retention limit is the chat conversation content: `messages.content`
(every user and assistant turn, verbatim) and `conversations.title` (derived from
the first line of the student's first message), for the two modules that go
through `/api/chat`. Unlike Exam Ally, Interview Mentor, and JobApp, this content
has no purge and no time limit. If a student uses a chat for something outside
coursework (the motivating example is a medical question), its full text sits in
the database indefinitely, and the professor is then in the position of holding
that content.

Two facts sharpen the decision:
- The server backs up the database on a schedule. A delete-it-later approach (a
  TTL that purges old rows) is therefore insufficient: any row that lives even
  briefly is captured in a backup and lingers there. The only mechanism that
  actually keeps the content off the server is to never write it.
- The persistence is not wired to any user feature today. The `conversations` and
  `messages` tables are written by `/api/chat`, but nothing in the UI reads them
  back. The `/api/conversations` list endpoint exists but no component calls it,
  and the client already generates its own in-session `conversationId` (a
  `crypto.randomUUID` held in React state) to thread the current chat. So removing
  persistence removes zero currently wired functionality; it only forgoes a future
  "reopen a past chat" feature that would have to be built regardless.

## 2. Decision and principle

**Conversational free-text is not persisted server-side.** For chat-style modules
the server keeps only content-free usage events. This is a standing principle, not
a one-off removal: future chat modules (General Chat, AI Comparisons) inherit it.

This supersedes ADR-002, which introduced conversation persistence as groundwork
for a resume feature that was never surfaced.

## 3. Scope

In scope, the two modules routed through `/api/chat`:
- Coding Tutor (`coding_companion`)
- Sandbox "Ask AI" helper (`sandbox_chat`)

Out of scope and unchanged, because they persist coursework or team artifacts by
deliberate design and already carry their own purge-after-use protections:
- Exam Ally (`exam_documents`/`exam_document_pages`/`exam_questions`/`exam_answers`;
  the uploaded PDF is never written to disk and page text is purged after an exam
  is built, per ADR-015).
- Interview Mentor (`interviews`/`interview_turns`; the resume is not stored).
- JobApp (`job_applications`/`tailored_documents`; the resume text is purged after
  review, per ADR-015).
- Project Assistant deliverables and their coach transcripts (a shared team
  artifact the team owns and returns to).

## 4. What is removed

The two tables and everything that exists only to serve them:
- `conversations` (id, userEmail, module, title, modelId, timestamps).
- `messages` (id, conversationId, role, content, modelId, token counts, cost,
  timestamp).

## 5. Changes

**Server route (`app/api/chat/route.ts`).** Remove the persistence calls:
`claimConversation`/`createConversation` (resolving the conversation id) and both
`appendMessage` writes (the user turn before streaming, and the assistant turn in
`onFinish`). Keep the `recordUsageEvent` call. The route still streams identically:
it never read stored history to build model context (the client sends the full
message array each turn via the AI SDK), so answer quality is unchanged. The route
no longer needs a persisted conversation id.

**Schema (`lib/db/schema.ts`).** Delete the `conversations` and `messages` table
definitions.

**Migration.** `npx drizzle-kit generate` produces a migration that drops both
tables (`messages` first, then `conversations`, to satisfy the foreign key).
Dropping the tables also deletes any existing rows. The migration is destructive
by intent and additive to the migration history (a new numbered file), consistent
with how every prior schema change was applied.

**Data layer (`lib/db/index.ts`).** Remove the functions that exist only for these
tables: `createConversation`, `claimConversation`, `getOwnedConversation`,
`listConversations`, `listMessages`, `appendMessage`, `deleteConversation`, and
`deriveTitle`. `npm run typecheck` catches any stray caller. (Verified during
brainstorming: only `app/api/chat/route.ts` and `app/api/conversations/route.ts`
use them; the exam, interview, and jobapp modules use their own tables.)

**List endpoint (`app/api/conversations/route.ts`).** Remove it; it is unused by
the UI and would otherwise return data that no longer exists.

**Client (`components/chat/Chat.tsx`, `components/sandbox/SandboxChat.tsx`, and the
request contract in `lib/chat/config.ts`).** Stop sending `conversationId` on the
wire: drop it from `chatRequestSchema` and from the transport body. The client may
keep an in-memory id for its own UI threading, but it is no longer transmitted or
interpreted by the server. The ephemeral per-request `context` field (the Sandbox's
current script and results) is untouched: it was already never persisted.

## 6. What is kept

`usage_events` is unchanged (ADR-006). It records, per event: module, event type,
model, provider, input/output token counts, cost, latency, prompt and response
character *lengths* (never the text), and outcome. This is the usage-statistics
surface the professor wants, and it already contains no content: an off-topic or
sensitive question appears only as, for example, one `chat_completion` in
`coding_companion` with a prompt character count and a cost, with nothing readable.

Within a live session, both chats behave exactly as before. The only capability
given up is reopening a chat after a refresh or on another device, which no UI
offers today.

## 7. Backups: honest caveat

Dropping the tables clears the live database and stops all future writes, so from
the deploy of this change no new chat content enters the database or any future
backup. But backups already taken still contain past chat rows. This change does
not scrub them; they age out only on the existing backup rotation. Removing that
historical content sooner is an operations action on the backup store, outside
this application and outside the scope and access of this work. This spec records
the caveat so the decision is explicit; if the professor wants the historical
content gone sooner, that is handled separately by shortening backup retention or
purging old backups on the server.

## 8. Error handling and behavior

- No behavior change to streaming, cancellation, rate limiting, empty-response
  handling, or cost reporting. Those paths do not depend on persistence.
- Removing the writes cannot fail in a way that affects the student: there is
  simply no database write on the request path anymore beyond the analytics event,
  which is already wrapped so it can never break a conversation.

## 9. Testing

- **Route/unit:** a chat completion records exactly one content-free `usage_events`
  row and performs no conversation or message write. Assert that the schema no
  longer exports `conversations` or `messages`, and that the removed data-layer
  functions are gone.
- **Regression:** existing chat end-to-end specs still pass (the chat streams and
  renders); they do not assert persistence. Exam, interview, and jobapp specs are
  unaffected because their data layer is untouched.
- **Migration:** on a fresh database the drop migration applies cleanly and the two
  tables are absent afterward.

## 10. Documentation

A short ADR recording the reversal: this supersedes ADR-002. It states the
rationale (privacy; a student's non-coursework or sensitive content should not
accumulate; scheduled backups make delete-later insufficient) and the consequence
(no chat-resume feature; usage statistics retained in content-free form). Recorded
in `webapp/docs/development/migration-log.md` per project practice.

## 11. Non-goals

- No replacement chat-history feature.
- No changes to Exam Ally, Interview Mentor, JobApp, or Project Assistant
  persistence.
- No scrubbing of existing backups (an operations task outside this app).

## 12. Open questions

None. Scope (chat modules only) and mechanism (drop the tables, do not merely
purge) were settled in brainstorming.
