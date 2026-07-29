/**
 * ChatISA production launcher: ONE process to start, no dependencies beyond
 * Node itself. It does three jobs the old Streamlit command did in one line:
 *
 *   1. Loads chatisa.env (KEY=VALUE lines beside this script) into the
 *      environment, so secrets live in one file outside the app code.
 *   2. Starts the Next.js standalone server as a child on 127.0.0.1:3000
 *      (never exposed), restarting it if it ever crashes.
 *   3. Terminates HTTPS on 443 with the university certificate files and
 *      relays every request to the child, streaming both ways (chat responses
 *      stream; nothing is buffered).
 *
 * Configuration (all read from chatisa.env, falling back to defaults):
 *   CHATISA_SSL_CERT  path to the certificate (default: the path the old
 *                     Streamlit command used)
 *   CHATISA_SSL_KEY   path to the private key
 *   CHATISA_HTTPS_PORT  public port (default 443)
 *   CHATISA_HTTP_ONLY   "1" runs plain HTTP on CHATISA_HTTPS_PORT instead,
 *                       for a local smoke test only; never set on the server.
 *
 * Run:  node chatisa-server.mjs   (the bat file does exactly this)
 */
import { readFileSync, existsSync, createWriteStream } from "node:fs";
import { spawn } from "node:child_process";
import { createServer as createHttpsServer } from "node:https";
import { createServer as createHttpServer, request as httpRequest } from "node:http";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

// Everything shows in the console window (so a double-click is diagnosable
// on sight) AND lands in chatisa.log (so history survives the window).
const logStream = createWriteStream(join(here, "chatisa.log"), { flags: "a" });
const teeLine = (line) => {
  const stamped = `${line}\n`;
  process.stdout.write(stamped);
  logStream.write(`${new Date().toISOString()} ${stamped}`);
};
console.log = (...a) => teeLine(a.join(" "));
console.warn = console.log;
console.error = console.log;

// --- 1. Environment ---------------------------------------------------------
const envFile = join(here, "chatisa.env");
if (existsSync(envFile)) {
  // strip a UTF-8 BOM (Notepad adds one) so the first key parses.
  const text = readFileSync(envFile, "utf8").replace(/^﻿/, "");
  const commented = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 1) continue;
    const key = trimmed.slice(0, eq).trim();
    const rest = trimmed.slice(eq + 1).trim();
    // Match how dotenv reads the SAME line on the dev machine, or a value that
    // works in `next dev` silently breaks here. Dev proved fine and production
    // broke on exactly this (2026-07-27): an inline comment after
    // DEEPGRAM_TOKEN was stripped by dotenv in dev but shipped into the
    // credential by this parser, and Deepgram answered "Invalid credentials".
    // So: surrounding quotes come off, and everything from an unquoted # on
    // is a comment, not value.
    const quoted = rest.match(/^"([^"]*)"/) ?? rest.match(/^'([^']*)'/);
    const value = quoted ? quoted[1] : rest.split("#")[0].trim();
    if (!quoted && rest.includes("#")) commented.push(key);
    if (!(key in process.env)) process.env[key] = value;
  }
  console.log(`[chatisa] loaded environment from ${envFile}`);
  for (const key of commented) {
    // Names only, never values: this line goes to the log.
    console.log(`[chatisa] note: dropped an inline # comment from the value of ${key}`);
  }
} else {
  console.error(`[chatisa] PROBLEM: ${envFile} not found.`);
  if (existsSync(`${envFile}.txt`)) {
    console.error(
      "[chatisa] Found chatisa.env.txt instead: Notepad added a hidden .txt.",
    );
    console.error(
      "[chatisa] Fix: in a Command Prompt here, run:  ren chatisa.env.txt chatisa.env",
    );
  } else {
    console.error(
      "[chatisa] Fix: copy chatisa.env.example, rename the copy to chatisa.env, fill it in.",
    );
  }
}

// Pre-flight: name what is missing HERE, in this window, before the app
// starts and turns the same problem into a bare "Internal Server Error".
const REQUIRED = ["AUTH_SECRET", "AUTH_URL", "AUTH_GOOGLE_ID", "AUTH_GOOGLE_SECRET"];
const missing = REQUIRED.filter((k) => !process.env[k]?.trim());
if (missing.length > 0) {
  console.error("");
  console.error(`[chatisa] NOT STARTING. These required settings are empty or missing:`);
  for (const k of missing) console.error(`[chatisa]   - ${k}`);
  console.error(`[chatisa] Open chatisa.env in Notepad, fill them in, save, and run again.`);
  console.error("");
  // Leave the window up long enough to read when double-clicked.
  setTimeout(() => process.exit(1), 30_000);
  throw new Error("missing required configuration");
}
const RECOMMENDED = [
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "GOOGLE_API_KEY",
  "HF_TOKEN",
  "DEEPGRAM_TOKEN",
];
for (const k of RECOMMENDED.filter((k) => !process.env[k]?.trim())) {
  console.warn(`[chatisa] note: ${k} is not set; the models it serves will be hidden`);
}
// Job Scout's weekly harvest sources (2026-07-28). Absent keys leave the
// feed empty rather than hiding models, so they get their own note.
for (const k of ["RAPIDAPI_KEY", "USAJOBS_API_KEY", "USAJOBS_EMAIL"].filter(
  (k) => !process.env[k]?.trim(),
)) {
  console.warn(`[chatisa] note: ${k} is not set; Job Scout's weekly feed will be missing that source`);
}

const INTERNAL_PORT = 3000;
const PUBLIC_PORT = Number(process.env.CHATISA_HTTPS_PORT ?? 443);
const HTTP_ONLY = process.env.CHATISA_HTTP_ONLY === "1";
const CERT = process.env.CHATISA_SSL_CERT ?? "C:\\Users\\webapp\\.conda\\chat_isa\\ssl\\chatisa.pem";
const KEY = process.env.CHATISA_SSL_KEY ?? "C:\\Users\\webapp\\.conda\\chat_isa\\ssl\\chatisapriv.key";

// --- 2. The Next.js standalone server, kept alive ---------------------------
const serverJs = join(here, "server.js");
if (!existsSync(serverJs)) {
  console.error(`[chatisa] server.js not found beside this script (${serverJs}).`);
  console.error("[chatisa] This script must sit in the deploy bundle's root folder.");
  process.exit(1);
}

// Node-version check. The database driver is a compiled binary targeted at
// ONE Node version (see chatisa-manifest.json, written at build time); a
// mismatch must be a plain sentence, not ERR_DLOPEN_FAILED (2026-07-25).
try {
  const manifest = JSON.parse(
    readFileSync(join(here, "chatisa-manifest.json"), "utf8"),
  );
  if (manifest.nodeAbi && manifest.nodeAbi !== process.versions.modules) {
    console.error("");
    console.error(
      `[chatisa] NOT STARTING. This bundle was built for Node ${manifest.nodeMajor ?? "?"} ` +
        `(ABI ${manifest.nodeAbi}); this machine is running Node ${process.versions.node.split(".")[0]} ` +
        `(ABI ${process.versions.modules}).`,
    );
    console.error(
      `[chatisa] Fix: install Node.js ${manifest.nodeMajor ?? "the matching version"} from https://nodejs.org`,
    );
    console.error(
      "[chatisa] (or rebuild the bundle with --node-abi=" + process.versions.modules + ").",
    );
    console.error("");
    setTimeout(() => process.exit(1), 30_000);
    throw new Error("node version mismatch");
  }
} catch (err) {
  if (err?.message === "node version mismatch") throw err;
  // No manifest (older bundle): fall through and let it try.
}

// Copy-integrity check. A folder dragged over Remote Desktop can silently
// lose files (first observed 2026-07-25: .next/node_modules vanished in
// transit and every request failed with a cryptic module error). Verify the
// pieces that a partial copy tends to drop, and say so in plain language.
const MUST_EXIST = [
  [".next/node_modules", "the server's bundled module aliases"],
  ["node_modules/next/package.json", "the Next.js runtime"],
  ["node_modules/better-sqlite3/package.json", "the database driver"],
  ["public/runtimes/pyodide/pyodide.mjs", "the Python runtime files"],
  [".next/static", "the client assets"],
];
const damaged = MUST_EXIST.filter(([p]) => !existsSync(join(here, p)));
// An extraction can materialize a broken link as an EMPTY directory, which
// fools a bare existence check (it did on 2026-07-25): every alias package
// must actually contain its package.json.
try {
  const aliasDir = join(here, ".next", "node_modules");
  const { readdirSync } = await import("node:fs");
  for (const name of readdirSync(aliasDir)) {
    if (!existsSync(join(aliasDir, name, "package.json"))) {
      damaged.push([`.next/node_modules/${name}`, "an aliased package, empty or broken"]);
    }
  }
} catch {
  /* covered by the MUST_EXIST entry above */
}
if (damaged.length > 0) {
  console.error("");
  console.error("[chatisa] NOT STARTING. This folder is INCOMPLETE; missing:");
  for (const [p, what] of damaged) console.error(`[chatisa]   - ${p} (${what})`);
  console.error(
    "[chatisa] This usually means the copy to this machine lost files in transit.",
  );
  console.error(
    "[chatisa] Fix: delete this folder, copy chatisa-app.zip over instead, and extract it",
  );
  console.error(
    "[chatisa] with 7-Zip (right-click > 7-Zip > Extract Here), NOT Explorer's Extract All:",
  );
  console.error(
    "[chatisa] despite the name the archive is a tar, which Explorer refuses. Then put your",
  );
  console.error(
    "[chatisa] chatisa.env back inside the extracted folder, beside this script.",
  );
  console.error("");
  setTimeout(() => process.exit(1), 30_000);
  throw new Error("incomplete bundle copy");
}

let child = null;
let shuttingDown = false;
function startChild() {
  child = spawn(process.execPath, [serverJs], {
    env: {
      ...process.env,
      NODE_ENV: "production",
      PORT: String(INTERNAL_PORT),
      HOSTNAME: "127.0.0.1",
    },
    stdio: ["ignore", "pipe", "pipe"],
    cwd: here,
  });
  // The app's own output goes to the window and the log, like ours.
  const forward = (chunk) => {
    process.stdout.write(chunk);
    logStream.write(chunk);
  };
  child.stdout.on("data", forward);
  child.stderr.on("data", forward);
  child.on("exit", (code) => {
    if (shuttingDown) return;
    console.error(`[chatisa] app exited (code ${code}); restarting in 3s`);
    setTimeout(startChild, 3000);
  });
}
startChild();

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    shuttingDown = true;
    child?.kill();
    process.exit(0);
  });
}

// Startup feature verification: run the deep health check (real PDF worker
// parse, db write with read-back, brand assets) once the app is up, and say
// the result in plain language. A failing feature is WARNED about loudly but
// does not stop the server: chat must not go down because PDF parsing did.
(async function verifyFeatures() {
  for (let attempt = 0; attempt < 24; attempt++) {
    await new Promise((r) => setTimeout(r, 5000));
    try {
      const res = await fetch(`http://127.0.0.1:${INTERNAL_PORT}/api/health?deep=1`);
      const body = await res.json();
      const deep = body?.checks?.deep;
      if (!deep) continue;
      // Informational states that are not failures (mirrors OPTIONAL_DEEP in
      // make-deploy-bundle.mjs): scout reports "ok, N active postings" when
      // healthy and "no harvest yet" on a fresh database minutes before the
      // boot harvest fills it; speech is optional per deployment. Stale
      // feeds and real failures still warn.
      const informational = {
        scout: /^(ok, \d+ active postings|no harvest yet)$/,
        speech: /^not configured/,
      };
      const bad = Object.entries(deep).filter(
        ([name, v]) => v !== "ok" && !informational[name]?.test(String(v)),
      );
      if (bad.length === 0) {
        console.log("[chatisa] feature check: database, PDF worker, and brand assets all verified");
      } else {
        for (const [name, detail] of bad) {
          console.error(`[chatisa] WARNING: feature "${name}" is BROKEN in this deployment: ${detail}`);
        }
        console.error("[chatisa] The site stays up, but fix the above before students hit it.");
      }
      return;
    } catch {
      /* app still starting; retry */
    }
  }
  console.error("[chatisa] WARNING: the feature check never completed; inspect chatisa.log");
})();

// --- 3. TLS termination and relay -------------------------------------------
function relay(req, res) {
  const upstream = httpRequest(
    {
      host: "127.0.0.1",
      port: INTERNAL_PORT,
      method: req.method,
      path: req.url,
      headers: {
        ...req.headers,
        // The app builds absolute URLs (OAuth callbacks) from these.
        "x-forwarded-proto": HTTP_ONLY ? "http" : "https",
        "x-forwarded-host": req.headers.host ?? "",
      },
    },
    (upstreamRes) => {
      res.writeHead(upstreamRes.statusCode ?? 502, upstreamRes.headers);
      upstreamRes.pipe(res);
    },
  );
  upstream.on("error", () => {
    if (!res.headersSent) {
      res.writeHead(503, { "content-type": "text/plain" });
    }
    res.end("ChatISA is restarting. Refresh in a few seconds.");
  });
  req.pipe(upstream);
}

let front;
if (HTTP_ONLY) {
  front = createHttpServer(relay);
  console.log(`[chatisa] SMOKE TEST MODE: plain http on port ${PUBLIC_PORT}`);
} else {
  let tls;
  try {
    tls = { cert: readFileSync(CERT), key: readFileSync(KEY) };
  } catch (err) {
    console.error(`[chatisa] could not read the SSL files:\n  cert: ${CERT}\n  key:  ${KEY}`);
    console.error(`[chatisa] ${err.message}`);
    console.error("[chatisa] Fix the paths in chatisa.env (CHATISA_SSL_CERT / CHATISA_SSL_KEY).");
    process.exit(1);
  }
  front = createHttpsServer(tls, relay);
}

front.on("upgrade", (req, socket) => socket.destroy());
front.listen(PUBLIC_PORT, () => {
  console.log(
    `[chatisa] listening on ${HTTP_ONLY ? "http" : "https"}://0.0.0.0:${PUBLIC_PORT} -> app on 127.0.0.1:${INTERNAL_PORT}`,
  );
});
front.on("error", (err) => {
  console.error(`[chatisa] cannot listen on port ${PUBLIC_PORT}: ${err.message}`);
  console.error("[chatisa] Port 443 needs the old app stopped, and the task to run as an account allowed to bind it.");
  process.exit(1);
});
