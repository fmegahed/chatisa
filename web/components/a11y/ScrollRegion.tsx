"use client";

import { useEffect, useRef, useState } from "react";

/**
 * A named region that joins the tab order only while its content overflows
 * (WCAG 2.1.1). Keyboard users need a stop to scroll wide code or tables, but
 * a stop on content that does not scroll is an empty one (#19).
 */
export function ScrollRegion({
  as: Tag = "div",
  label,
  className,
  children,
}: {
  as?: "div" | "pre";
  label: string;
  className?: string;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLElement>(null);
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
  }, []);

  return (
    <Tag
      ref={ref as never}
      role="region"
      aria-label={label}
      tabIndex={overflows ? 0 : undefined}
      className={className}
    >
      {children}
    </Tag>
  );
}
