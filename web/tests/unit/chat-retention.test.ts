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
