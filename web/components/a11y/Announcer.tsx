"use client";

import { useEffect, useState } from "react";
import { ANNOUNCE_EVENT, type AnnounceDetail } from "@/lib/a11y/announce";

/**
 * The app's shared live regions. Mounted once in the root layout; components
 * call `announce()` from lib/a11y/announce instead of rendering their own.
 * The text is cleared and re-set on each message so repeating the same
 * message ("Saved") is announced again.
 */
export function Announcer() {
  const [polite, setPolite] = useState("");
  const [assertive, setAssertive] = useState("");

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    function onAnnounce(e: Event) {
      const { message, politeness } = (e as CustomEvent<AnnounceDetail>).detail;
      const set = politeness === "assertive" ? setAssertive : setPolite;
      set("");
      clearTimeout(timer);
      timer = setTimeout(() => set(message), 60);
    }
    window.addEventListener(ANNOUNCE_EVENT, onAnnounce);
    return () => {
      clearTimeout(timer);
      window.removeEventListener(ANNOUNCE_EVENT, onAnnounce);
    };
  }, []);

  return (
    <>
      <div
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        data-testid="announcer-polite"
      >
        {polite}
      </div>
      <div
        className="sr-only"
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        data-testid="announcer-assertive"
      >
        {assertive}
      </div>
    </>
  );
}
