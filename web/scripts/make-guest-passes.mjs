/**
 * Mints guest magic passes for external collaborators.
 *
 *   node scripts/make-guest-passes.mjs [count] [expiry] [baseUrl]
 *   node scripts/make-guest-passes.mjs 10 2026-09-30 https://chatisa.fsb.miamioh.edu
 *
 * Prints the shareable links (KEEP THESE; they cannot be recovered from the
 * server, which stores only hashes) and the two env lines to deploy. Each
 * link's position is its guest number: the third link is guest-3, and its
 * usage shows up under guest-3@guest.chatisa.
 *
 * To revoke ONE guest without disturbing the others, replace their hash in
 * CHATISA_GUEST_PASS_HASHES with 64 zeros (positions must be kept, since
 * position = identity). To end the trial, remove the variable or let the
 * expiry pass.
 */
import { randomBytes, createHash } from "node:crypto";

const count = Number(process.argv[2] ?? 10);
const expiry = process.argv[3] ?? "2026-09-30";
const baseUrl = (process.argv[4] ?? "https://YOUR-DOMAIN").replace(/\/$/, "");

if (!Number.isInteger(count) || count < 1 || count > 50) {
  console.error("count must be between 1 and 50");
  process.exit(1);
}
if (Number.isNaN(Date.parse(expiry))) {
  console.error("expiry must be an ISO date, for example 2026-09-30");
  process.exit(1);
}

const passes = Array.from({ length: count }, () =>
  randomBytes(16).toString("hex"),
);
const hashes = passes.map((p) =>
  createHash("sha256").update(p, "utf8").digest("hex"),
);

console.log(`Guest links (expire ${expiry}); share ONE per collaborator:\n`);
passes.forEach((p, i) => {
  console.log(`  guest-${String(i + 1).padEnd(2)} ${baseUrl}/guest?pass=${p}`);
});
console.log("\nAdd to the server environment (.env.local or the deployment):\n");
console.log(`CHATISA_GUEST_PASS_HASHES=${hashes.join(",")}`);
console.log(`CHATISA_GUEST_EXPIRES=${expiry}`);
