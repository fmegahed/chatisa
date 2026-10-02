"use client";

import { useEffect } from "react";

/**
 * Sets the document title from a page that cannot export metadata
 * (not-found.tsx), so the tab and screen readers name the page (WCAG 2.4.2).
 * Next writes the root layout's default title after this effect runs (its
 * metadata streams in), so the title is re-applied whenever <head> changes
 * while this page is mounted.
 */
export function DocumentTitle({ title }: { title: string }) {
  useEffect(() => {
    const apply = () => {
      if (document.title !== title) document.title = title;
    };
    apply();
    const observer = new MutationObserver(apply);
    observer.observe(document.head, {
      childList: true,
      subtree: true,
      characterData: true,
    });
    return () => observer.disconnect();
  }, [title]);
  return null;
}
