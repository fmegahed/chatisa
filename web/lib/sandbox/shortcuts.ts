export type Shortcut = {
  /** What the shortcut does, in plain words (no em dashes). */
  action: string;
  /** The rendered keys for the current platform, e.g. "Ctrl+Enter". */
  keys: string;
  /** Present only when the shortcut is language-scoped. */
  scope?: "R only";
};

export type ShortcutGroup = { title: string; items: Shortcut[] };

/** The platform modifier glyph text: Cmd on macOS, Ctrl elsewhere. */
export function modLabel(isMac: boolean): "Cmd" | "Ctrl" {
  return isMac ? "Cmd" : "Ctrl";
}

/**
 * SSR-safe platform probe. Returns false (Ctrl) on the server and on any
 * environment without a navigator, so the first render is deterministic; the
 * dialog refines it in an effect after mount.
 */
export function detectIsMac(): boolean {
  if (typeof navigator === "undefined") return false;
  const s = `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`;
  return /Mac|iPhone|iPad|iPod/i.test(s);
}

/**
 * The full grouped shortcut list, keys already rendered for the platform. Only
 * bindings that actually exist in the editor are listed (see the plan's verified
 * table). Autocomplete is Ctrl+Space on every platform, so it is not templated.
 */
export function shortcutGroups(isMac: boolean): ShortcutGroup[] {
  const mod = modLabel(isMac);
  return [
    {
      title: "Run code",
      items: [
        { action: "Run statement or selection", keys: `${mod}+Enter` },
        { action: "Run whole script", keys: `${mod}+Shift+Enter` },
        { action: "Source silently", keys: `${mod}+Shift+S` },
      ],
    },
    {
      title: "Edit",
      items: [
        { action: "Insert pipe", keys: `${mod}+Shift+M`, scope: "R only" },
        { action: "Toggle comment", keys: `${mod}+/` },
      ],
    },
    {
      title: "Assist",
      items: [
        { action: "Documentation for symbol", keys: `${mod}+Click or F1` },
        { action: "Autocomplete", keys: "Ctrl+Space" },
      ],
    },
  ];
}
