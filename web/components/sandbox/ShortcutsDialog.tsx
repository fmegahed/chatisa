"use client";

import { useEffect, useRef, useSyncExternalStore } from "react";
import { detectIsMac, shortcutGroups } from "@/lib/sandbox/shortcuts";

/** A store that never changes, so useSyncExternalStore reads its snapshot once. */
function subscribeNever(): () => void {
  return () => {};
}

/**
 * A modal listing every Coding Studio editor shortcut, so students can discover
 * them. Platform-aware (Cmd on macOS, Ctrl elsewhere), SSR-safe (defaults to
 * Ctrl until an effect confirms the platform), and accessible: labelled dialog,
 * focus trapped inside, Esc and backdrop close, focus returned to the trigger.
 */
export function ShortcutsDialog(props: { dark: boolean; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  // The element focused before the dialog opened, restored on close.
  const returnFocusRef = useRef<HTMLElement | null>(null);
  // The platform is a client-only fact. useSyncExternalStore gives the server
  // (and the first render) a definite `false` (Ctrl) and the client the real
  // answer, so the first paint is deterministic and hydration matches without
  // setting state from an effect.
  const isMac = useSyncExternalStore(subscribeNever, detectIsMac, () => false);

  useEffect(() => {
    returnFocusRef.current = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    return () => {
      returnFocusRef.current?.focus?.();
    };
  }, []);

  // Trap Tab within the panel and close on Esc.
  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.stopPropagation();
      props.onClose();
      return;
    }
    if (e.key !== "Tab") return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusable = panel.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && active === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  }

  const groups = shortcutGroups(isMac);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="sb-shortcuts-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onKeyDown={onKeyDown}
      onClick={(e) => e.target === e.currentTarget && props.onClose()}
    >
      <div
        ref={panelRef}
        tabIndex={-1}
        data-sb-theme={props.dark ? "dark" : "light"}
        className="sb-root flex max-h-[85vh] w-full max-w-md flex-col rounded-card border border-[var(--sb-border)] bg-[var(--sb-bg)] text-[var(--sb-text)] shadow-xl outline-none"
      >
        <div className="flex items-center justify-between border-b border-[var(--sb-border)] px-4 py-3">
          <h2 id="sb-shortcuts-title" className="text-lg font-bold">
            Keyboard shortcuts
          </h2>
          <button
            type="button"
            onClick={props.onClose}
            className="rounded-card border border-[var(--sb-border)] px-2 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
          >
            Close
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          {groups.map((group) => (
            <section key={group.title}>
              <h3 className="mb-1 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
                {group.title}
              </h3>
              <dl className="divide-y divide-[var(--sb-border)]">
                {group.items.map((s) => (
                  <div
                    key={s.action}
                    className="flex items-center justify-between gap-4 py-1.5"
                  >
                    <dt className="text-sm">
                      {s.action}
                      {s.scope ? (
                        <span className="ml-2 rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
                          {s.scope}
                        </span>
                      ) : null}
                    </dt>
                    <dd>
                      <kbd className="rounded border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-0.5 font-mono text-xs">
                        {s.keys}
                      </kbd>
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
          <p className="text-xs text-[var(--sb-muted)]">
            Shortcuts work while the editor has focus. On macOS, Cmd replaces Ctrl,
            except Autocomplete which stays Ctrl+Space.
          </p>
        </div>
      </div>
    </div>
  );
}
