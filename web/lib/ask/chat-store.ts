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
export function saveChat(
  storage: Storage,
  chat: StoredChat,
): { trimmed: boolean } {
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
    // Quota hit despite the budget (other keys share it): keep only the chat
    // being saved and retry once.
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
  for (const m of messages as {
    role?: string;
    parts?: { type?: string; text?: string }[];
  }[]) {
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
