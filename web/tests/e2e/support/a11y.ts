import type { Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

export const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"];

/** axe at WCAG 2.1 AA, returning a compact summary for readable diffs. */
export async function axeViolations(page: Page) {
  const r = await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze();
  return r.violations.map((v) => ({
    id: v.id,
    targets: v.nodes.map((n) => n.target.join(" ")),
  }));
}

/**
 * Checks axe does not make, each one a finding class from the 2026-09
 * accessibility audit (issues #4-#39):
 *  - nonInteractiveTabStops: tabindex>=0 on content with no interactive role
 *    and nothing to scroll (#16, #19, #20)
 *  - hiddenTabStops: focusable controls that are visually hidden (#6)
 *  - roleNameOnGeneric: aria-label on a div/span with no role (#11, #12)
 *  - placeholderContrast: placeholder text below 4.5:1 (#7, #8)
 *  - unmarkedRequired: required fields whose label shows no "*" (#25, #31)
 */
export async function auditDom(page: Page) {
  return page.evaluate(() => {
    const INTERACTIVE_ROLES = new Set([
      "button", "link", "tab", "radio", "checkbox", "slider", "separator",
      "textbox", "menuitem", "menuitemradio", "menuitemcheckbox", "option",
      "switch", "combobox", "listbox", "gridcell", "treeitem", "scrollbar",
      "spinbutton", "searchbox", "tree", "grid", "tablist", "radiogroup",
    ]);
    const NATIVE = "a[href],button,input,select,textarea,summary,iframe,audio[controls],video[controls]";
    const desc = (el: Element) => {
      const id = el.id ? `#${el.id}` : "";
      const label = el.getAttribute("aria-label");
      const text = (el.textContent ?? "").trim().slice(0, 40);
      return `${el.tagName.toLowerCase()}${id}${label ? `[aria-label="${label}"]` : ""} "${text}"`;
    };
    // checkVisibility covers display:none on any ancestor (md:hidden wrappers).
    const visible = (el: Element) =>
      el.checkVisibility({ visibilityProperty: true }) && !el.closest("[inert]");
    const scrollable = (el: HTMLElement) =>
      el.scrollHeight > el.clientHeight + 1 || el.scrollWidth > el.clientWidth + 1;

    const nonInteractiveTabStops: string[] = [];
    const hiddenTabStops: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("*"))) {
      if (!visible(el) || el.tabIndex < 0) continue;
      if ((el as HTMLButtonElement).disabled) continue;
      const native = el.matches(NATIVE);
      const role = el.getAttribute("role");
      const editable = el.isContentEditable && el.getAttribute("contenteditable") !== null;
      // Only elements that are actually in the tab order.
      if (!native && !el.hasAttribute("tabindex") && !editable) continue;
      if (!native && !editable && !(role && INTERACTIVE_ROLES.has(role)) && !scrollable(el)) {
        nonInteractiveTabStops.push(desc(el));
      }
      const r = el.getBoundingClientRect();
      const s = getComputedStyle(el);
      const clipped = s.clip === "rect(0px, 0px, 0px, 0px)" || s.clipPath.includes("inset(50%)");
      if ((r.width <= 1 && r.height <= 1) || clipped || Number(s.opacity) === 0) {
        // A visually hidden native input is fine when its visible <label>
        // shows the focus ring (one stop, visible focus); otherwise the user
        // tabs onto something they cannot see.
        const label = Array.from((el as HTMLInputElement).labels ?? []).find(
          (l) => l.getBoundingClientRect().width > 1,
        );
        if (label) {
          const prev = document.activeElement as HTMLElement | null;
          // focusVisible: script focus after mouse clicks would otherwise not
          // match :focus-visible, hiding a ring a keyboard user does see.
          // Blur first: refocusing the already-focused element is a no-op.
          el.blur();
          el.focus({ focusVisible: true } as FocusOptions);
          const ring = getComputedStyle(label).outlineStyle !== "none";
          prev?.focus?.();
          if (ring) continue;
        }
        hiddenTabStops.push(desc(el));
      }
    }

    const roleNameOnGeneric = Array.from(
      document.querySelectorAll("div[aria-label]:not([role]),span[aria-label]:not([role])"),
    ).filter(visible).map(desc);

    function lum(rgb: number[]) {
      const c = rgb.map((v) => {
        const x = v / 255;
        return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    const parse = (c: string) => (c.match(/[\d.]+/g) ?? []).map(Number);
    function bgOf(el: Element | null): number[] {
      for (let e = el; e; e = e.parentElement) {
        const p = parse(getComputedStyle(e).backgroundColor);
        if (p.length >= 3 && (p[3] ?? 1) > 0.5) return p.slice(0, 3);
      }
      return [255, 255, 255];
    }
    const placeholderContrast: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLInputElement>("input[placeholder],textarea[placeholder]"))) {
      if (!visible(el) || !el.placeholder) continue;
      const fg = parse(getComputedStyle(el, "::placeholder").color);
      const alpha = fg[3] ?? 1;
      const bg = bgOf(el);
      const mixed = fg.slice(0, 3).map((v, i) => v * alpha + bg[i] * (1 - alpha));
      const [a, b] = [lum(mixed), lum(bg)];
      const ratio = (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
      if (ratio < 4.5) placeholderContrast.push(`${desc(el)} ${ratio.toFixed(2)}:1`);
    }

    const unmarkedRequired: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLInputElement>("input[required],textarea[required],select[required],[aria-required=true]"))) {
      if (!visible(el) || el.type === "hidden" || el.type === "radio") continue;
      const labels = Array.from(el.labels ?? []);
      const byId = el.getAttribute("aria-labelledby")?.split(/\s+/).map((id) => document.getElementById(id)).filter(Boolean) ?? [];
      const text = [...labels, ...byId].map((l) => l!.textContent ?? "").join(" ");
      if (!text.includes("*")) unmarkedRequired.push(desc(el));
    }

    return { nonInteractiveTabStops, hiddenTabStops, roleNameOnGeneric, placeholderContrast, unmarkedRequired };
  });
}

/** Tab through the page and record each stop's role and accessible name. */
export async function tabStops(page: Page, max = 80) {
  await page.locator("body").focus();
  const stops: string[] = [];
  for (let i = 0; i < max; i++) {
    await page.keyboard.press("Tab");
    const s = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      if (!el || el === document.body) return null;
      const name = el.getAttribute("aria-label") ?? (el.textContent ?? "").trim().slice(0, 30);
      return `${el.tagName.toLowerCase()}${el.getAttribute("role") ? `[${el.getAttribute("role")}]` : ""} ${name}`;
    });
    if (!s || stops.includes(s)) break;
    stops.push(s);
  }
  return stops;
}
