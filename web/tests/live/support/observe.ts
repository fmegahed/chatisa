import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import type { Page, TestInfo } from "@playwright/test";

/**
 * The recorder for a live run.
 *
 * A live run's value is its evidence, not its pass/fail bit. A model can produce
 * a plausible-looking answer while the page logs a hydration error, a tool call
 * 500s and retries, or a request to a provider file endpoint fails; none of that
 * shows up in an assertion on visible text. So everything observable is captured
 * and written next to the transcript, and the operator reads the report.
 *
 * Deliberately quiet about what it considers a problem. Classifying is the
 * reader's job; a recorder that filtered aggressively would hide the surprise
 * that makes the run worth doing.
 */

export interface ConsoleEntry {
  type: string;
  text: string;
  at: number;
}

export interface FailedRequest {
  url: string;
  method: string;
  failure: string | null;
  at: number;
}

export interface HttpError {
  url: string;
  method: string;
  status: number;
  body: string;
  at: number;
}

export class Observer {
  readonly console: ConsoleEntry[] = [];
  readonly pageErrors: string[] = [];
  readonly failedRequests: FailedRequest[] = [];
  readonly httpErrors: HttpError[] = [];
  readonly notes: string[] = [];
  private readonly startedAt = Date.now();

  constructor(
    private readonly page: Page,
    private readonly info: TestInfo,
  ) {
    const at = () => Math.round((Date.now() - this.startedAt) / 100) / 10;

    page.on("console", (message) => {
      // Truncated: a fetch failure inside a WASM runtime logs the ENTIRE
      // response body, and a Next.js 404 body is a full HTML document. One such
      // message made an observations.json unreadable, which defeats the point.
      this.console.push({
        type: message.type(),
        text: message.text().slice(0, 600),
        at: at(),
      });
    });
    // An uncaught exception in the page. Never benign.
    page.on("pageerror", (err) => {
      this.pageErrors.push(String(err));
    });
    page.on("requestfailed", (request) => {
      this.failedRequests.push({
        url: request.url(),
        method: request.method(),
        failure: request.failure()?.errorText ?? null,
        at: at(),
      });
    });
    page.on("response", (response) => {
      if (response.status() < 400) return;
      // Body is read lazily and best-effort: a streaming response that is
      // already consumed, or a navigation that raced, must not break the run.
      void response
        .text()
        .then((text) => {
          this.httpErrors.push({
            url: response.url(),
            method: response.request().method(),
            status: response.status(),
            body: text.slice(0, 400),
            at: at(),
          });
        })
        .catch(() => {
          this.httpErrors.push({
            url: response.url(),
            method: response.request().method(),
            status: response.status(),
            body: "<body unavailable>",
            at: at(),
          });
        });
    });
  }

  /** A human-readable marker in the report, for staging a long run. */
  note(text: string): void {
    const seconds = Math.round((Date.now() - this.startedAt) / 100) / 10;
    this.notes.push(`${seconds}s ${text}`);
    // Also to stdout, so a run that is still going shows where it is.
    console.log(`      ${text} @ ${seconds}s`);
  }

  /** Console messages of type "error", which is the usual first real signal. */
  consoleErrors(): ConsoleEntry[] {
    return this.console.filter((c) => c.type === "error");
  }

  /**
   * Writes one artifact into this test's own folder and attaches it, so the
   * HTML/JSON report links it rather than the operator hunting for a path.
   */
  async save(name: string, body: string | Buffer): Promise<string> {
    const dir = join(
      "tests",
      "live",
      ".artifacts",
      this.info.titlePath.join(" - ").replace(/[^a-z0-9 .-]/gi, "_"),
    );
    const path = join(dir, name);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, body);
    await this.info.attach(name, { path });
    return path;
  }

  /** The full record. Called at the end of every live test, pass or fail. */
  async writeReport(extra: Record<string, unknown> = {}): Promise<void> {
    const report = {
      test: this.info.titlePath.join(" > "),
      status: this.info.status,
      durationSeconds: Math.round((Date.now() - this.startedAt) / 1000),
      url: this.page.url(),
      notes: this.notes,
      pageErrors: this.pageErrors,
      consoleErrors: this.consoleErrors(),
      failedRequests: this.failedRequests,
      httpErrors: this.httpErrors,
      ...extra,
    };
    await this.save("observations.json", JSON.stringify(report, null, 2));

    // A short summary to stdout: the operator should not have to open a file to
    // learn that six requests failed.
    const counts = [
      this.pageErrors.length ? `${this.pageErrors.length} page errors` : null,
      this.consoleErrors().length
        ? `${this.consoleErrors().length} console errors`
        : null,
      this.failedRequests.length
        ? `${this.failedRequests.length} failed requests`
        : null,
      this.httpErrors.length ? `${this.httpErrors.length} HTTP >=400` : null,
    ].filter(Boolean);
    console.log(
      counts.length ? `      OBSERVED: ${counts.join(", ")}` : "      observed: clean",
    );
  }
}
