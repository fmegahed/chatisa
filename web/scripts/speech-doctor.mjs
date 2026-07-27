/**
 * ChatISA speech doctor: pinpoints WHY the server cannot reach Deepgram.
 *
 * Built for the 2026-07-27 outage: /api/health?deep=1 said "Deepgram could
 * not be reached", which names the symptom but not the cause. This script
 * runs ON THE SERVER, needs nothing but Node (no npm install), and tells the
 * network story step by step: DNS, TCP, TLS, HTTPS, then a real token mint.
 * The same steps run against api.openai.com as a control, so "everything
 * fails" (a general egress problem) is distinguishable from "only Deepgram
 * fails" (a host-specific firewall rule — the fix is an IT ticket asking for
 * outbound TCP 443 to api.deepgram.com).
 *
 * Run it from the chatisa-app folder (or anywhere near chatisa.env):
 *   node speech-doctor.mjs
 *
 * It never prints the DEEPGRAM_TOKEN value, only whether one was found.
 * The output is safe to paste into an IT ticket.
 */
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolve4, resolve6 } from "node:dns/promises";
import { connect as tcpConnect } from "node:net";
import { connect as tlsConnect } from "node:tls";

const here = dirname(fileURLToPath(import.meta.url));
const TARGET = "api.deepgram.com";
const CONTROL = "api.openai.com";
const STEP_TIMEOUT_MS = 10_000;

// --- chatisa.env, found wherever this script happens to sit ----------------
// Same parser as the FIXED chatisa-server.mjs: BOM stripped, quotes unwrapped,
// unquoted inline # comments dropped. Launchers older than 2026-07-27 did NOT
// drop inline comments, so a line like  DEEPGRAM_TOKEN=abc123 # my note
// shipped "abc123 # my note" to Deepgram as the credential. That case is
// called out loudly below because it looks exactly like a bad key.
let tokenLineHadComment = false;
for (const candidate of [
  join(here, "chatisa.env"),
  join(here, "chatisa-app", "chatisa.env"),
  join(here, "..", "chatisa.env"),
]) {
  if (!existsSync(candidate)) continue;
  const text = readFileSync(candidate, "utf8").replace(/^﻿/, "");
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq < 1) continue;
    const key = trimmed.slice(0, eq).trim();
    const rest = trimmed.slice(eq + 1).trim();
    const quoted = rest.match(/^"([^"]*)"/) ?? rest.match(/^'([^']*)'/);
    const value = quoted ? quoted[1] : rest.split("#")[0].trim();
    if (key === "DEEPGRAM_TOKEN" && !quoted && rest.includes("#")) {
      tokenLineHadComment = true;
    }
    if (!(key in process.env)) process.env[key] = value;
  }
  console.log(`Using settings from ${candidate}`);
  break;
}
if (tokenLineHadComment) {
  console.log("");
  console.log("!!! The DEEPGRAM_TOKEN line in chatisa.env has an inline # comment.");
  console.log("!!! chatisa-server.mjs launchers from before 2026-07-27 pass that");
  console.log("!!! comment INTO the credential, and Deepgram then rejects it as");
  console.log("!!! invalid. Fix: edit chatisa.env so the line is only");
  console.log("!!!     DEEPGRAM_TOKEN=<the key, nothing after it>");
  console.log("!!! then restart the ChatISA task. This script already tests with");
  console.log("!!! the comment stripped, so a PASS below does NOT clear an old");
  console.log("!!! launcher.");
  console.log("");
}

const token = process.env.DEEPGRAM_TOKEN?.trim() ?? "";
console.log(
  token
    ? `DEEPGRAM_TOKEN: found (${token.length} characters; value never printed)`
    : "DEEPGRAM_TOKEN: NOT FOUND. Reachability is still tested; the mint step is skipped.",
);
console.log(`Node ${process.version} on ${process.platform}\n`);

// Every cause in the chain, because fetch hides ETIMEDOUT/ENOTFOUND in .cause.
function chain(err) {
  const parts = [];
  const seen = new Set();
  let cur = err;
  while (cur !== undefined && cur !== null && !seen.has(cur)) {
    seen.add(cur);
    if (cur instanceof Error) {
      parts.push(cur.code ? `${cur.message} (${cur.code})` : cur.message);
      cur = cur.cause;
    } else {
      parts.push(String(cur));
      break;
    }
  }
  return parts.join(" <- ") || "unknown error";
}

function withTimeout(promise, what) {
  return Promise.race([
    promise,
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`no answer after ${STEP_TIMEOUT_MS / 1000}s (${what})`)),
        STEP_TIMEOUT_MS,
      ).unref(),
    ),
  ]);
}

function tryTcp(host, address) {
  return withTimeout(
    new Promise((resolvePromise, reject) => {
      const socket = tcpConnect({ host: address, port: 443 });
      socket.on("connect", () => {
        socket.destroy();
        resolvePromise();
      });
      socket.on("error", (err) => {
        socket.destroy();
        reject(err);
      });
    }),
    `TCP connect to ${address}:443`,
  );
}

function tryTls(host) {
  return withTimeout(
    new Promise((resolvePromise, reject) => {
      const socket = tlsConnect({ host, port: 443, servername: host });
      socket.on("secureConnect", () => {
        const cert = socket.getPeerCertificate();
        const summary = {
          authorized: socket.authorized,
          subject: cert?.subject?.CN ?? "?",
          issuer: cert?.issuer?.O ?? cert?.issuer?.CN ?? "?",
          validTo: cert?.valid_to ?? "?",
        };
        socket.destroy();
        resolvePromise(summary);
      });
      socket.on("error", (err) => {
        socket.destroy();
        reject(err);
      });
    }),
    `TLS handshake with ${host}`,
  );
}

/** One host's full story. Returns per-step "ok"/failure text for the verdict. */
async function examine(host) {
  console.log(`--- ${host} ---`);
  const result = { dns: null, tcp: null, tls: null, https: null };

  let v4 = [];
  let v6 = [];
  try {
    v4 = await withTimeout(resolve4(host), "DNS A lookup");
  } catch (err) {
    console.log(`  DNS (IPv4): FAILED: ${chain(err)}`);
  }
  try {
    v6 = await withTimeout(resolve6(host), "DNS AAAA lookup");
  } catch {
    /* no AAAA record is normal, not a finding */
  }
  if (v4.length > 0 || v6.length > 0) {
    result.dns = "ok";
    console.log(`  DNS: ok -> ${[...v4, ...v6].join(", ")}`);
  } else {
    result.dns = "failed";
    console.log("  DNS: FAILED for both IPv4 and IPv6.");
  }

  for (const address of [...v4.slice(0, 2), ...v6.slice(0, 2)]) {
    try {
      await tryTcp(host, address);
      result.tcp = result.tcp ?? "ok";
      console.log(`  TCP  ${address}:443: ok`);
    } catch (err) {
      result.tcp = result.tcp === "ok" ? "ok" : "failed";
      console.log(`  TCP  ${address}:443: FAILED: ${chain(err)}`);
    }
  }

  if (result.tcp === "ok") {
    try {
      const tls = await tryTls(host);
      result.tls = tls.authorized ? "ok" : "untrusted";
      console.log(
        `  TLS: ${tls.authorized ? "ok" : "certificate NOT trusted"}; ` +
          `subject=${tls.subject}, issuer=${tls.issuer}, expires=${tls.validTo}`,
      );
      // A university TLS-inspection appliance re-signs the certificate; the
      // issuer then names the appliance or the institution, not a public CA,
      // and Node (which only trusts public CAs) refuses the connection.
    } catch (err) {
      result.tls = "failed";
      console.log(`  TLS: FAILED: ${chain(err)}`);
    }
  }

  try {
    const res = await fetch(`https://${host}/`, {
      signal: AbortSignal.timeout(STEP_TIMEOUT_MS),
    });
    result.https = "ok";
    // Any HTTP status at all proves the full path works; 4xx is expected
    // without credentials.
    console.log(`  HTTPS: ok (server answered with HTTP ${res.status})`);
  } catch (err) {
    result.https = "failed";
    console.log(`  HTTPS: FAILED: ${chain(err)}`);
  }

  console.log("");
  return result;
}

const deepgram = await examine(TARGET);
const control = await examine(CONTROL);

// The exact call the app makes, credential and all.
let mint = "skipped (no DEEPGRAM_TOKEN found)";
if (token && deepgram.https === "ok") {
  try {
    const res = await fetch(`https://${TARGET}/v1/auth/grant`, {
      method: "POST",
      headers: {
        Authorization: `Token ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ttl_seconds: 30 }),
      signal: AbortSignal.timeout(STEP_TIMEOUT_MS),
    });
    if (res.ok) {
      const body = await res.json().catch(() => ({}));
      mint = body.access_token
        ? "ok: Deepgram minted a token. Speech should work."
        : `HTTP ${res.status} but no token in the response`;
    } else {
      mint = `Deepgram answered HTTP ${res.status}: the network is fine; check the key and the account's credit`;
    }
  } catch (err) {
    mint = `FAILED: ${chain(err)}`;
  }
} else if (token) {
  mint = "skipped (Deepgram is unreachable; fix that first)";
}
console.log(`--- token mint (the app's actual call) ---\n  ${mint}\n`);

// --- Verdict ----------------------------------------------------------------
console.log("=== VERDICT ===");
if (mint.startsWith("ok")) {
  console.log("Everything works from this machine. If the site still has no");
  console.log("speech, restart the ChatISA task so the app rereads chatisa.env.");
} else if (deepgram.dns !== "ok" && control.dns === "ok") {
  console.log(`DNS on this server cannot resolve ${TARGET} (the control host`);
  console.log("resolves fine). Ask IT to fix DNS for this name.");
} else if (deepgram.tcp !== "ok" && control.tcp === "ok") {
  console.log(`Outbound TCP 443 to ${TARGET} is BLOCKED, while the control`);
  console.log(`host ${CONTROL} connects fine. This is a host-specific egress`);
  console.log("firewall rule. Ask IT to allow outbound TCP 443 from this");
  console.log(`server to ${TARGET} (and paste this output into the ticket).`);
} else if (deepgram.tls === "untrusted" || deepgram.tls === "failed") {
  console.log("The TCP connection opens but the TLS handshake fails or the");
  console.log("certificate is not trusted. If the issuer printed above names");
  console.log("the university or an appliance rather than a public CA, a");
  console.log("TLS-inspection device is re-signing the traffic; ask IT to");
  console.log(`exempt ${TARGET} from inspection for this server.`);
} else if (control.https !== "ok") {
  console.log("BOTH hosts are unreachable: this server has no general outbound");
  console.log("HTTPS (or needs a proxy Node.js does not use). If chat works on");
  console.log("the site, rerun this script AS THE ACCOUNT the scheduled task");
  console.log("uses; per-account firewall rules can differ.");
} else {
  console.log("The steps above disagree in an unusual way; read them in order —");
  console.log("the first FAILED line is where the connection dies.");
}
