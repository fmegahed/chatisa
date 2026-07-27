import { describe, expect, it } from "vitest";
import { modLabel, shortcutGroups } from "@/lib/sandbox/shortcuts";

describe("platform modifier label", () => {
  it("is Cmd on macOS and Ctrl elsewhere", () => {
    expect(modLabel(true)).toBe("Cmd");
    expect(modLabel(false)).toBe("Ctrl");
  });
});

describe("shortcut list", () => {
  it("renders the modifier per platform", () => {
    const mac = shortcutGroups(true).flatMap((g) => g.items);
    const win = shortcutGroups(false).flatMap((g) => g.items);
    expect(mac.find((s) => s.action.startsWith("Run statement"))?.keys).toBe(
      "Cmd+Enter",
    );
    expect(win.find((s) => s.action.startsWith("Run statement"))?.keys).toBe(
      "Ctrl+Enter",
    );
  });

  it("lists exactly the real bindings, and marks the pipe R only", () => {
    const items = shortcutGroups(false).flatMap((g) => g.items);
    const byAction = Object.fromEntries(items.map((s) => [s.action, s]));
    expect(byAction["Run statement or selection"].keys).toBe("Ctrl+Enter");
    expect(byAction["Run whole script"].keys).toBe("Ctrl+Shift+Enter");
    expect(byAction["Source silently"].keys).toBe("Ctrl+Shift+S");
    expect(byAction["Insert pipe"].keys).toBe("Ctrl+Shift+M");
    expect(byAction["Insert pipe"].scope).toBe("R only");
    expect(byAction["Toggle comment"].keys).toBe("Ctrl+/");
    expect(byAction["Documentation for symbol"].keys).toBe(
      "Ctrl+Click or F1",
    );
    // Autocomplete is Ctrl on every platform, never Cmd.
    expect(byAction["Autocomplete"].keys).toBe("Ctrl+Space");
    expect(shortcutGroups(true).flatMap((g) => g.items).find(
      (s) => s.action === "Autocomplete",
    )?.keys).toBe("Ctrl+Space");
  });

  it("contains no em dashes in any copy", () => {
    const text = shortcutGroups(true)
      .flatMap((g) => [g.title, ...g.items.map((s) => `${s.action} ${s.keys} ${s.scope ?? ""}`)])
      .join(" ");
    expect(text).not.toContain("—");
  });
});
