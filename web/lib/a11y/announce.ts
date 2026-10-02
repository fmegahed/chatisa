/**
 * Screen reader announcements (WCAG 4.1.3 Status Messages).
 *
 * `announce()` sends a message to the single <Announcer /> live region mounted
 * in the root layout, so any component can report an outcome ("Job hidden",
 * "12 jobs match") without rendering its own region. Messages are delivered
 * through a window event so this module stays free of React state.
 */
export const ANNOUNCE_EVENT = "chatisa:announce";

export type Politeness = "polite" | "assertive";

export interface AnnounceDetail {
  message: string;
  politeness: Politeness;
}

export function announce(message: string, politeness: Politeness = "polite") {
  if (typeof window === "undefined" || !message.trim()) return;
  window.dispatchEvent(
    new CustomEvent<AnnounceDetail>(ANNOUNCE_EVENT, {
      detail: { message, politeness },
    }),
  );
}

/**
 * Move focus to an element that is not normally focusable (a heading, a
 * panel) after content changes, so screen reader users land on the new
 * content (WCAG 2.4.3). Adds tabindex="-1" when needed; never adds the
 * element to the tab order.
 */
export function focusElement(el: HTMLElement | null | undefined) {
  if (!el) return;
  if (!el.hasAttribute("tabindex") && el.tabIndex < 0) {
    el.setAttribute("tabindex", "-1");
  }
  el.focus({ preventScroll: true });
  el.scrollIntoView({ block: "nearest" });
}
