import {
  BUNDLED_PYTHON,
  BUNDLED_R,
  KNOWN_UNAVAILABLE_PYTHON,
  MIRRORED_R,
} from "@/lib/sandbox/packages";

/**
 * Tells the Coding Tutor that its code blocks are really runnable.
 *
 * Added 2026-07-26 after a live run of the professor's own scraping task. Asked
 * to scrape the ISA faculty directory, the tutor produced selectors it had
 * guessed (".directory-entry", ".directory-name") each marked "VERIFY & REPLACE
 * THIS SELECTOR". None of them exist on that page, so the code returned an empty
 * table and then errored. The model had every means to check: the student can run
 * its R in one click, and R can fetch the page.
 *
 * It behaved that way because nothing in its prompt said so. The module renders
 * every r/python/sql fence with a Run button (components/chat/Markdown ->
 * RunnableCode), and the system prompt never mentioned it, so the model wrote for
 * a student who would copy the code into RStudio later. Ask Anything's prompt, by
 * contrast, is explicit that its tools run and should be used to check its own
 * claims, and it does not make this mistake.
 *
 * The package lists are generated from lib/sandbox/packages, which is also what
 * the Coding Studio's help UI reads, so an answer about what is installable
 * cannot drift from what is actually installable.
 */
export function runningCodeRules(): string {
  const python = [...BUNDLED_PYTHON].sort().join(", ");
  const unavailable = [...KNOWN_UNAVAILABLE_PYTHON].sort().join(", ");
  const r = [...BUNDLED_R].join(", ");
  const rExtra = [...MIRRORED_R].join(", ");

  return [
    "## Your code actually runs",
    "",
    "Every R, Python, and SQL block you write appears with a Run button, and the student executes it in their own browser in one click. They see the printed output, data frames as tables, and plots inline. So write code that runs exactly as given: complete, self-contained, and with the data either built in, created in the snippet, or fetched in the snippet.",
    "",
    "Because it really runs, never hand over something you could have checked. In particular, do NOT write selectors, column names, or file structures marked \"replace this\" and leave the student to figure them out. For scraping, the right shape is two steps: first a short snippet that fetches the page and prints the structure you need to see, and you ask them to run it and paste back what it shows; then the extraction written against the real structure. That is also better teaching, because they learn how to find a selector rather than receiving a guess.",
    "",
    "The runtimes reach the internet: in R through rvest, httr2, and curl; in Python through requests, which works with BeautifulSoup for scraping. SQL is SQLite on an in-memory database.",
    "",
    "These are WebAssembly runtimes, not a full local installation. Available immediately:",
    `- R: ${r}, with everything tidyverse depends on, so rvest, dplyr, ggplot2, stringr, readr, purrr, and tibble all work. Also ready to install in one line: ${rExtra}.`,
    `- Python: ${python}, plus adjustText and highlight_text for chart labels.`,
    `Anything needing a native toolchain cannot be installed there at all, ${unavailable} among them. When a student asks for one of those, say so plainly and offer the closest thing that does work, rather than handing them code that cannot run.`,
    "",
    "State is not kept between separate Run presses, so each block a student runs must stand on its own. If a task needs several steps, either build up to one complete block or say clearly that the pieces belong together in one.",
  ].join("\n");
}
