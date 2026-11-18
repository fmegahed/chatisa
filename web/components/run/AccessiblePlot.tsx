"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  MAIDR_SCRIPT_URL,
  makeMaidrId,
  sampledNote,
  toMaidrSchema,
  type PlotData,
} from "@/lib/run/maidr-schema";

/**
 * A plot a student can explore from the keyboard and hear (#40).
 *
 * Renders the plot's <img> with its description (#23) as before. When the
 * runtime also sent the plot's data, it becomes a MAIDR chart: Tab to it, then
 * the arrow keys move from point to point and each one is announced; S plays
 * the values as tones, T changes how much text is read, B shows braille, and
 * Ctrl+/ lists every key. When there is no usable data (an unsupported chart,
 * several panels, a failed extraction) it is exactly the old plain image.
 *
 * Three facts about maidr.js 4.16.0, measured in the spike (2026-11-17), shape
 * this component:
 *  - It finds charts by the `maidr` attribute, on load and through a
 *    MutationObserver. That observer watches `maidr` only, not `maidr-data`.
 *  - It MOVES the element it wraps into its own React root. So our React owns
 *    only an empty host <div>; the <img> is created and removed by hand in an
 *    effect, and React never reconciles inside the host.
 *  - It does not notice when a chart leaves the page. Cleanup sends
 *    `maidr:unbindchart` first, then empties the host.
 *
 * maidr.js is GPL-3.0-or-later and is never imported: it is a separate,
 * unmodified file under /runtimes/maidr/ (scripts/setup-runtimes.mjs), loaded
 * here with a script tag the first time a navigable plot appears.
 */

let maidrLoad: Promise<void> | null = null;

/**
 * MAIDR picks its interface language from the browser, so a Spanish-language
 * browser got Spanish announcements inside an English page, read with English
 * pronunciation rules. Its settings live in localStorage under "maidr-settings"
 * and are merged over its defaults, so English is stored as the starting
 * language, once and only when the student has not chosen one. A student can
 * still switch languages in MAIDR's own settings (Ctrl+,).
 */
function seedEnglish() {
  try {
    const KEY = "maidr-settings";
    const current = JSON.parse(localStorage.getItem(KEY) ?? "null") as {
      general?: { language?: string };
    } | null;
    if (current?.general?.language) return;
    localStorage.setItem(
      KEY,
      JSON.stringify({
        ...(current ?? {}),
        general: { ...(current?.general ?? {}), language: "en" },
      }),
    );
  } catch {
    // Storage blocked: MAIDR then follows the browser language.
  }
}

/** Loads maidr.js once per page. A plain script tag rather than next/script
 * because the English setting has to be stored before the script runs. */
function loadMaidr(): Promise<void> {
  if (maidrLoad) return maidrLoad;
  maidrLoad = new Promise<void>((resolve, reject) => {
    seedEnglish();
    const script = document.createElement("script");
    script.src = MAIDR_SCRIPT_URL;
    script.async = true;
    script.dataset.chatisaMaidr = "";
    script.onload = () => resolve();
    script.onerror = () => {
      // Let a later plot try again (a transient network failure).
      maidrLoad = null;
      reject(new Error("maidr.js failed to load"));
    };
    document.head.append(script);
  });
  return maidrLoad;
}

export function AccessiblePlot({
  src,
  alt,
  data,
  title,
  imgClassName,
  hostClassName,
}: {
  src: string;
  /** The description (#23); stays the image's alt text. */
  alt: string;
  /** The plot's data from the runtime, already parsed. */
  data?: PlotData;
  /** The chart's title, if known; MAIDR names the chart with it. */
  title?: string;
  imgClassName?: string;
  hostClassName?: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const [ready, setReady] = useState(false);
  // One id per mounted plot: MAIDR uses it as a DOM id prefix.
  const id = useMemo(() => makeMaidrId(), []);
  const schema = useMemo(
    // Without a title MAIDR would name the chart only by its type ("Bar
    // Chart"), so the description stands in.
    () => toMaidrSchema(data, { id, title: title ?? data?.title ?? alt }),
    [data, id, title, alt],
  );
  const json = schema ? JSON.stringify(schema) : null;

  useEffect(() => {
    const el = host.current;
    if (!el || !json) return;
    const img = document.createElement("img");
    img.src = src;
    img.alt = alt;
    if (imgClassName) img.className = imgClassName;
    el.append(img);
    // MAIDR's observer reacts to the `maidr` attribute, and handles it after
    // the fact. An image removed in between (React mounts effects twice in
    // development; a student paging quickly through plots) made it log "Plot
    // element has no parent node". So the attribute goes on one frame later,
    // and only if the image is still on the page.
    let bound = false;
    const frame = requestAnimationFrame(() => {
      if (!img.isConnected) return;
      img.setAttribute("maidr", json);
      bound = true;
    });

    // MAIDR's focusable chart appears inside the host once it has wrapped the
    // image; that is when the keyboard hint becomes true.
    const hasChart = () => !!el.querySelector('[tabindex="0"]');
    const observer = new MutationObserver(() => setReady(hasChart()));
    observer.observe(el, { childList: true, subtree: true });
    loadMaidr().catch(() => {
      // The image is already on the page with its alt text: the plain plot.
    });

    // Keyboard only. The chart is focusable, so a mouse click or a tap would
    // focus it and wake MAIDR: an outline, its text panel, and arrow keys that
    // play tones (seen Nov 17). A student who is not exploring by keyboard
    // should get the plain picture, as before #40. So for the length of a
    // real pointer press the chart is simply not focusable: its tabindex is
    // removed on pointerdown and put back once the press is over. Clicking it
    // then focuses nothing, exactly as clicking a plain image does, and the
    // image can still be dragged out (stopping or blurring the focus instead
    // cancelled the drag in Chrome, measured Nov 17). The tabindex returns
    // after the click, not on pointerup, because a tap's focus comes after
    // pointerup; any key brings it back at once, so Tab always reaches the
    // chart. A screen reader's own activation sends a click with no
    // pointerdown, and its users Tab to the chart, so they are unaffected.
    const chartSelector = 'figure[id^="maidr-figure-"] > [tabindex="0"]';
    let pressed: HTMLElement | null = null;
    let restoreTimer = 0;
    const restore = () => {
      window.clearTimeout(restoreTimer);
      pressed?.setAttribute("tabindex", "0");
      pressed = null;
    };
    const restoreSoon = () => {
      if (!pressed) return;
      window.clearTimeout(restoreTimer);
      restoreTimer = window.setTimeout(restore, 0);
    };
    const onPointerDown = (e: PointerEvent) => {
      if (!e.isTrusted || !e.pointerType) return;
      const chart = el.querySelector<HTMLElement>(chartSelector);
      if (!chart) return;
      chart.removeAttribute("tabindex");
      pressed = chart;
      // A press that ends without a click (moved off, long-pressed).
      window.clearTimeout(restoreTimer);
      restoreTimer = window.setTimeout(restore, 2000);
    };
    el.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("click", restoreSoon, true);
    document.addEventListener("pointercancel", restoreSoon, true);
    document.addEventListener("dragend", restore, true);
    document.addEventListener("keydown", restore, true);

    return () => {
      cancelAnimationFrame(frame);
      restore();
      el.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("click", restoreSoon, true);
      document.removeEventListener("pointercancel", restoreSoon, true);
      document.removeEventListener("dragend", restore, true);
      document.removeEventListener("keydown", restore, true);
      observer.disconnect();
      if (bound) {
        document.dispatchEvent(new CustomEvent("maidr:unbindchart", { detail: img }));
      }
      el.replaceChildren();
      setReady(false);
    };
  }, [json, src, alt, imgClassName]);

  if (!json) {
    return (
      // A runtime-generated data URL; next/image cannot optimise it.
      // eslint-disable-next-line @next/next/no-img-element
      <img src={src} alt={alt} className={imgClassName} />
    );
  }

  const note = sampledNote(data);
  return (
    <>
      {/* React owns this element only; MAIDR builds everything inside it. */}
      <div ref={host} data-maidr-plot="" className={hostClassName} />
      {ready ? (
        // Read by screen readers in sequence; on screen only while the chart
        // has keyboard focus (globals.css), so a student who never Tabs to a
        // plot sees exactly the picture they saw before #40.
        <p data-maidr-hint="" className="text-xs text-[var(--plot-hint,inherit)]">
          Explore this chart from the keyboard: Tab to it, then use the arrow
          keys. S turns sound on or off, T changes the text, B shows braille;
          Ctrl+/ lists every key.{note ? ` ${note}` : ""}
        </p>
      ) : null}
    </>
  );
}
