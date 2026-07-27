"use client";

import { useEffect, useState } from "react";
import { proxyCapText } from "@/lib/net/proxy-limits";
import {
  buildPyodideIndex,
  classifyPythonPackage,
  type PyodideIndex,
} from "@/lib/sandbox/packages";

/**
 * A search box that answers "can I use this Python package?" authoritatively, by
 * reading Pyodide's own package lock: ready now, installable with micropip, or not
 * built for the browser. Loaded lazily when the help opens.
 */
function PythonPackageChecker() {
  const [index, setIndex] = useState<PyodideIndex | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let cancelled = false;
    fetch("/runtimes/pyodide/pyodide-lock.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no lock"))))
      .then((lock) => {
        if (!cancelled) setIndex(buildPyodideIndex(lock));
      })
      .catch(() => {
        if (!cancelled) setLoadError(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const result = index ? classifyPythonPackage(query, index) : null;
  const label =
    result?.status === "ready"
      ? "Ready now"
      : result?.status === "installable"
        ? "Installable"
        : result?.status === "unavailable"
          ? "Not available"
          : "";
  const tone =
    result?.status === "unavailable"
      ? "text-[var(--sb-accent)]"
      : "text-[var(--sb-text)]";

  return (
    <div className="rounded border border-[var(--sb-border)] p-2">
      <label className="block text-sm">
        <span className="mb-1 block font-bold">Check a package</span>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="for example seaborn, requests, sympy"
          className="w-64 max-w-full rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-1"
        />
      </label>
      {loadError ? (
        <p className="mt-1 text-xs text-[var(--sb-muted)]">
          The package list could not load. Try the install line above; it will
          tell you if the package is not available.
        </p>
      ) : query && !index ? (
        <p className="mt-1 text-xs text-[var(--sb-muted)]">Checking...</p>
      ) : result ? (
        <p role="status" className={`mt-1 text-xs ${tone}`}>
          <span className="font-bold">{label}.</span> {result.message}
        </p>
      ) : null}
    </div>
  );
}

/**
 * A slim notice between the toolbar and the panels that explains, in plain
 * language, that everything runs in the browser (so your work stays private)
 * and how networking differs by language: R reaches the internet through a
 * built-in proxy, while Python web requests are still limited by the browser's
 * CORS rule. A "What can I install?" toggle opens per-language help on
 * packages: what is already here, how to add more, and why an install
 * sometimes fails.
 */

/** Per-language package help, shown when the notice is expanded. */
function PackageHelp({ languageId }: { languageId: string }) {
  if (languageId === "python") {
    return (
      <>
        <p>
          <strong>Ready now (preloaded):</strong> pandas, numpy, matplotlib,
          scikit-learn, statsmodels, pyarrow, polars, seaborn, openpyxl. Just
          import them. <strong>Available on import:</strong> hundreds more that
          Pyodide builds (requests, beautifulsoup4, networkx, sympy and so on)
          download the first time you import them, or add one up front with
          micropip:
        </p>
        <pre className="mt-1 overflow-x-auto rounded bg-[var(--sb-bg)] p-2 text-xs">
          import micropip{"\n"}await micropip.install(&quot;package-name&quot;)
        </pre>
        <p>
          <strong>Not available:</strong> a package that has to be compiled (for
          example statsforecast) or is not built for the browser. Check any
          package below:
        </p>
        <PythonPackageChecker />
        <p>
          Python web requests go through the browser, so they follow its CORS
          rule, EXCEPT requests: its GET and POST are routed through a
          built-in guarded proxy, so requests.get() works on ordinary websites
          and APIs, and beautifulsoup4 parses what comes back. Responses over{" "}
          {proxyCapText()} and private or internal addresses are refused with a message
          starting &quot;ChatISA proxy:&quot;. Other HTTP clients (urllib,
          httpx) still follow{" "}
          <a
            href="https://pyodide.org/en/stable/usage/api/python-api/http.html#network-limitations-in-pyodide"
            className="underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            the browser&apos;s networking limits
          </a>
          .
        </p>
      </>
    );
  }
  if (languageId === "r") {
    return (
      <>
        <p>
          <strong>Ready now:</strong> tidyverse, readxl, janitor and httr2. They
          load instantly. <strong>Installable:</strong> most of CRAN, built for
          WebAssembly, downloaded from the webR repo the usual way:
        </p>
        <pre className="mt-1 overflow-x-auto rounded bg-[var(--sb-bg)] p-2 text-xs">
          install.packages(&quot;tidymodels&quot;)
        </pre>
        <p>
          <strong>Not available:</strong> a package with no WebAssembly build or
          that needs system libraries. There is no name checker for R here, so if
          you are unsure, run <code>install.packages(&quot;name&quot;)</code> and it
          will tell you. A large package like tidymodels is a one-time download
          and needs a connection.
        </p>
        <p>
          R can fetch web pages and APIs (for example{" "}
          <code>rvest::read_html(url)</code>), routed through a built-in proxy,
          so even sites without CORS headers work.
        </p>
      </>
    );
  }
  return (
    <p>
      SQL runs on SQLite in your browser, so there are no packages to install.
      Charts use ggsql, an experimental tool that turns a query with a VISUALISE
      clause into a plot.
    </p>
  );
}

/** The WebAssembly explainer: what these runtimes are, the exact bundled
 * versions, and where each project lives. */
function RuntimeInfo() {
  const link = (href: string, text: string) => (
    <a
      href={href}
      className="underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      {text}
    </a>
  );
  return (
    <>
      <p>
        The Coding Studio runs real language runtimes compiled to{" "}
        {link("https://webassembly.org/", "WebAssembly")}, so your code executes
        inside this browser tab: nothing you write or upload is sent to a
        server, and there is nothing to install on your computer.
      </p>
      <ul className="list-disc space-y-1 pl-5">
        <li>
          <strong>Python 3.14.0</strong> via{" "}
          {link("https://pyodide.org/en/stable/", "Pyodide")}, the scientific
          Python distribution for the browser.
        </li>
        <li>
          <strong>R 4.6.0</strong> via{" "}
          {link("https://docs.r-wasm.org/webr/latest/", "webR 0.6.0")}, the R
          project&apos;s WebAssembly build.
        </li>
        <li>
          <strong>SQLite 3.53.0</strong> via{" "}
          {link("https://sqlite.org/wasm/doc/trunk/index.md", "SQLite Wasm")},
          the database&apos;s official browser build.
        </li>
      </ul>
      <p className="text-[var(--sb-muted)]">
        The browser build is the real interpreter, with two practical limits:
        packages must have a WebAssembly build (see the package help), and
        network access follows browser rules rather than desktop rules.
      </p>
    </>
  );
}

export function LimitationsNotice({ languageId }: { languageId: string }) {
  const [open, setOpen] = useState<null | "packages" | "runtimes">(null);
  return (
    <div className="border-b border-[var(--sb-border)] bg-[var(--sb-panel)] px-3 py-1.5 text-xs text-[var(--sb-muted)]">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span aria-hidden="true">🔒</span>
        <span>
          This runs entirely in your browser, so your work stays private. Both
          languages can reach the internet: R&apos;s rvest, httr2 and curl are
          tunneled through a built-in proxy, and Python&apos;s requests (GET and
          POST) goes through a guarded proxy as well, so ordinary websites work
          from both.
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => (v === "packages" ? null : "packages"))}
          aria-expanded={open === "packages"}
          // The main text colour (not the accent) so the link keeps AA contrast
          // in both themes; the underline still marks it as a control.
          className="font-bold text-[var(--sb-text)] underline underline-offset-2 hover:text-[var(--sb-accent)]"
        >
          {open === "packages" ? "Hide package help" : "What can I install?"}
        </button>
        <span aria-hidden="true">·</span>
        <button
          type="button"
          onClick={() => setOpen((v) => (v === "runtimes" ? null : "runtimes"))}
          aria-expanded={open === "runtimes"}
          className="font-bold text-[var(--sb-text)] underline underline-offset-2 hover:text-[var(--sb-accent)]"
        >
          {open === "runtimes" ? "Hide runtime info" : "About these runtimes"}
        </button>
      </div>
      {open === "packages" ? (
        <div className="mt-2 max-w-3xl space-y-2 text-[var(--sb-text)]">
          <PackageHelp languageId={languageId} />
          <p className="text-[var(--sb-muted)]">
            Installing a package reaches the package servers over the network.
            That always works. What varies by language is whether your own code
            can reach other websites.
          </p>
        </div>
      ) : null}
      {open === "runtimes" ? (
        <div className="mt-2 max-w-3xl space-y-2 text-[var(--sb-text)]">
          <RuntimeInfo />
        </div>
      ) : null}
    </div>
  );
}
