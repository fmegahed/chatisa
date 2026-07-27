import {
  BUNDLED_R,
  BUNDLED_R_CLOSURE,
  KNOWN_UNAVAILABLE_PYTHON,
  MIRRORED_R,
  classifyPythonPackage,
  normalizePkg,
  type PyodideIndex,
} from "@/lib/sandbox/packages";
import { requirementsFor, type RunLanguageId } from "@/lib/sandbox/requirements";

/**
 * Can this snippet actually run here, and should a Run button be offered?
 *
 * Added 2026-07-26 at the professor's instruction: install what can be installed
 * so the student's code runs, and where a package cannot work at all, do not
 * offer a button whose only possible outcome is an error. Their framing was the
 * right one, that this is a trust problem: a Run button is a promise, and a
 * promise the app knows it cannot keep is worse than no button.
 *
 * FOUR tiers, and the fourth is the important one:
 *
 *   ready       every package is already loaded. Run is instant.
 *   installable everything is obtainable, some on first use. Run, and say so.
 *   unknown     we cannot tell. Run, with no promises.
 *   blocked     a package provably cannot work here. No Run button.
 *
 * "unknown" exists because the cost of the two mistakes is not symmetric. Hiding
 * Run on a snippet that would have worked is a broken feature and a confused
 * student; showing Run on a snippet that fails is the status quo, plus a clear
 * error. So nothing is blocked without positive evidence, and absence of
 * evidence is never treated as evidence.
 */

export type Runnability = "ready" | "installable" | "unknown" | "blocked";

export interface RunnabilityVerdict {
  status: Runnability;
  /** Packages that will need fetching before the code can run. */
  willInstall: string[];
  /** Packages that cannot work here at all. Non-empty only when blocked. */
  impossible: string[];
  /** One sentence for the student. Null when there is nothing worth saying. */
  message: string | null;
}

/**
 * R packages that cannot run in WebAssembly, whatever the repository says.
 *
 * Kept SHORT and only for cases that are certain, because every name here can
 * hide a Run button. Each needs a system dependency the browser has no way to
 * provide: a JVM, an ODBC driver manager, or a native database client.
 */
export const KNOWN_UNAVAILABLE_R = new Set([
  "rJava",
  "RODBC",
  "RMySQL",
  "RPostgreSQL",
  "ROracle",
  "rgdal",
  "Rmpi",
]);

/** What the client knows about R package availability. */
export interface RIndex {
  /** Installed at session start, from our own mirror. */
  mirrored: Set<string>;
  /**
   * Every package WebR's repository serves. Absent when the manifest has not
   * been generated (public/runtimes is created by npm run setup:runtimes), and
   * absence must degrade to "unknown", never to "blocked".
   */
  repo: Set<string> | null;
}

/** The R index that holds when only the shipped constants are known. */
export function baseRIndex(): RIndex {
  return {
    mirrored: new Set<string>([
      ...BUNDLED_R,
      ...BUNDLED_R_CLOSURE,
      ...MIRRORED_R,
    ]),
    repo: null,
  };
}

/**
 * Builds the R index from the generated manifest, keeping the shipped constants
 * as a floor so a truncated manifest cannot un-know a bundled package.
 */
export function buildRIndex(manifest: {
  mirrored?: string[];
  repo?: string[];
}): RIndex {
  return {
    mirrored: new Set<string>([
      ...BUNDLED_R,
      ...BUNDLED_R_CLOSURE,
      ...MIRRORED_R,
      ...(manifest.mirrored ?? []),
    ]),
    repo: Array.isArray(manifest.repo) && manifest.repo.length
      ? new Set(manifest.repo)
      : null,
  };
}

interface PackageVerdict {
  status: "ready" | "installable" | "unknown" | "blocked";
  /** Why, when it matters. */
  note?: string;
}

function classifyR(name: string, index: RIndex): PackageVerdict {
  if (index.mirrored.has(name)) return { status: "ready" };
  if (KNOWN_UNAVAILABLE_R.has(name)) {
    return {
      status: "blocked",
      note: `${name} needs a system component the browser cannot provide`,
    };
  }
  if (index.repo) {
    return index.repo.has(name)
      ? { status: "installable" }
      : {
          status: "blocked",
          note: `${name} is not built for the browser version of R`,
        };
  }
  // No manifest: it may well install from WebR's repository on first use, and we
  // have no basis to say otherwise.
  return { status: "unknown" };
}

function classifyPython(name: string, index: PyodideIndex | null): PackageVerdict {
  const normalized = normalizePkg(name);
  if (KNOWN_UNAVAILABLE_PYTHON.has(normalized)) {
    return {
      status: "blocked",
      note: `${name} needs compiling, which the browser runtime cannot do`,
    };
  }
  if (!index) return { status: "unknown" };

  const verdict = classifyPythonPackage(name, index);
  if (!verdict) return { status: "unknown" };
  if (verdict.status === "ready") return { status: "ready" };
  if (verdict.status === "installable") return { status: "installable" };
  // classifyPythonPackage says "unavailable" for anything outside Pyodide's
  // lock, but its own message admits micropip may still manage a pure-Python
  // package. That hedge is exactly the "unknown" tier: not good enough to
  // promise, not bad enough to block.
  return { status: "unknown" };
}

export function assessRunnability(
  language: RunLanguageId,
  code: string,
  indexes: { python: PyodideIndex | null; r: RIndex },
): RunnabilityVerdict {
  const required = requirementsFor(language, code);
  if (language === "sql" || required.length === 0) {
    return { status: "ready", willInstall: [], impossible: [], message: null };
  }

  const willInstall: string[] = [];
  const impossible: string[] = [];
  const notes: string[] = [];
  let anyUnknown = false;

  for (const name of required) {
    const verdict =
      language === "r"
        ? classifyR(name, indexes.r)
        : classifyPython(name, indexes.python);
    if (verdict.status === "blocked") {
      impossible.push(name);
      if (verdict.note) notes.push(verdict.note);
    } else if (verdict.status === "installable") {
      willInstall.push(name);
    } else if (verdict.status === "unknown") {
      anyUnknown = true;
    }
  }

  if (impossible.length) {
    // The whole point: say which package, and why, instead of a Run button that
    // can only fail.
    const reason = notes[0] ?? `${impossible[0]} cannot run in the browser`;
    return {
      status: "blocked",
      willInstall,
      impossible,
      message: `This code cannot run here: ${reason}. Copy it and run it in R or Python on your computer.`,
    };
  }
  if (willInstall.length) {
    return {
      status: "installable",
      willInstall,
      impossible: [],
      message: `The first run installs ${willInstall.join(", ")}, so it takes a little longer.`,
    };
  }
  if (anyUnknown) {
    return { status: "unknown", willInstall: [], impossible: [], message: null };
  }
  return { status: "ready", willInstall: [], impossible: [], message: null };
}
