"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MODULES } from "@/lib/modules";

/**
 * Primary navigation. Current page is marked with aria-current and a
 * red underline (not color alone). Horizontally scrollable on narrow
 * viewports so nothing is hidden behind a disclosure.
 */
export function ModuleNav() {
  const pathname = usePathname();
  const items = [
    { slug: "", name: "Home" },
    ...MODULES.map(({ slug, name }) => ({ slug, name })),
  ];

  return (
    <nav aria-label="ChatISA modules" className="overflow-x-auto">
      <ul className="flex gap-1 whitespace-nowrap px-2 sm:px-4">
        {items.map(({ slug, name }) => {
          const href = `/${slug}`;
          const current =
            pathname === href || (slug === "" && pathname === "/");
          return (
            <li key={href}>
              {slug === "coding-studio" ? (
                // Full page load so the cross-origin isolation headers apply
                // (SPA navigation would leave the page un-isolated and disable
                // R networking).
                <a
                  href={href}
                  aria-current={current ? "page" : undefined}
                  className={`inline-block border-b-4 px-3 py-3 text-sm font-bold ${
                    current
                      ? "border-miami-red text-miami-red"
                      : "border-transparent text-ink hover:border-medium-tan hover:text-accent-red"
                  }`}
                >
                  {name}
                </a>
              ) : (
                <Link
                  href={href}
                  aria-current={current ? "page" : undefined}
                  className={`inline-block border-b-4 px-3 py-3 text-sm font-bold ${
                    current
                      ? "border-miami-red text-miami-red"
                      : "border-transparent text-ink hover:border-medium-tan hover:text-accent-red"
                  }`}
                >
                  {name}
                </Link>
              )}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
