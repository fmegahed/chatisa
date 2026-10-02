"use client";

import { useEffect, useState, type KeyboardEvent, type RefObject } from "react";

/**
 * True while the element's content overflows its box. A scroll container joins
 * the tab order only then (WCAG 2.1.1): keyboard users need a stop to scroll a
 * long console or table, but a stop on content that does not scroll is an empty
 * one (#19). Same rule as components/a11y/ScrollRegion, for containers that
 * need their own role (log, tabpanel) or a ref.
 */
export function useOverflows(ref: RefObject<HTMLElement | null>): boolean {
  const [overflows, setOverflows] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const check = () =>
      setOverflows(
        el.scrollWidth > el.clientWidth + 1 ||
          el.scrollHeight > el.clientHeight + 1,
      );
    check();
    const resize = new ResizeObserver(check);
    resize.observe(el);
    const mutate = new MutationObserver(check);
    mutate.observe(el, { childList: true, subtree: true, characterData: true });
    return () => {
      resize.disconnect();
      mutate.disconnect();
    };
  }, [ref]);
  return overflows;
}

/**
 * Arrow-key navigation for a tablist or radiogroup (APG): Left/Up go back,
 * Right/Down go forward (wrapping), Home/End jump to the ends. Moves focus to
 * the new item and selects it. Returns true when it handled the key.
 */
export function rovingKeyDown(
  e: KeyboardEvent,
  ids: string[],
  current: number,
  select: (index: number) => void,
): boolean {
  const last = ids.length - 1;
  let next: number;
  switch (e.key) {
    case "ArrowRight":
    case "ArrowDown":
      next = current >= last ? 0 : current + 1;
      break;
    case "ArrowLeft":
    case "ArrowUp":
      next = current <= 0 ? last : current - 1;
      break;
    case "Home":
      next = 0;
      break;
    case "End":
      next = last;
      break;
    default:
      return false;
  }
  e.preventDefault();
  select(next);
  document.getElementById(ids[next])?.focus();
  return true;
}
