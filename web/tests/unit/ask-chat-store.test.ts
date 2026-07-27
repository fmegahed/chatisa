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
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => [...map.keys()][i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
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
        {
          id: "m1",
          role: "user",
          parts: [{ type: "text", text: "  Explain p-values please  " }],
        },
      ]),
    ).toBe("Explain p-values please");
    expect(deriveTitle([])).toBe("New chat");
    // Longer than 60 chars is cut with an ellipsis character not an em dash.
    const long = "x".repeat(80);
    const t = deriveTitle([
      { id: "m", role: "user", parts: [{ type: "text", text: long }] },
    ]);
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
    saveChat(s, {
      ...chat("old", 1),
      messages: [{ id: "m", role: "user", parts: [{ type: "text", text: big }] }],
    });
    saveChat(s, {
      ...chat("mid", 2),
      messages: [{ id: "m", role: "user", parts: [{ type: "text", text: big }] }],
    });
    const res = saveChat(s, {
      ...chat("new", 3),
      messages: [{ id: "m", role: "user", parts: [{ type: "text", text: big }] }],
    });
    expect(res.trimmed).toBe(true);
    const ids = listChats(s).map((c) => c.id);
    expect(ids).toContain("new"); // the just-saved chat is never trimmed away
    expect(ids).not.toContain("old"); // oldest goes first
  });
});
