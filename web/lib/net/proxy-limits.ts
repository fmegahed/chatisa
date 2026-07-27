/**
 * The Python web proxy's size limits, and the words used to describe them.
 *
 * Separate from lib/net/py-proxy because that module is "server-only": it
 * resolves hostnames and holds the SSRF discipline. These numbers, though, are
 * quoted in a tool description and in the Ask Anything system prompt, and those
 * live in modules that can end up in a client bundle. Importing the server-only
 * module for one integer would break the build.
 *
 * Client-safe by construction: constants and a formatter, nothing else.
 */

/**
 * Response cap, raised from 4 MB to 25 MB on 2026-07-26 (professor's decision).
 *
 * 4 MB is a low ceiling for analytics coursework, and it blocked the professor's
 * own County Business Patterns exercise: cbp23st.zip is 11,115,845 bytes. Both
 * alternatives were worse. The Census API needs a key (verified: it answers
 * "Missing Key" without one), so there is no smaller keyless slice; and attaching
 * the file instead does not work either, because the archive expands to
 * 92,967,992 bytes of CSV, far past the 25 MB attachment cap.
 *
 * 25 MB matches that attachment cap deliberately, so there is ONE number to
 * explain: 25 MB in, 25 MB out.
 */
export const PROXY_RESPONSE_MAX = 25_000_000;

/** Request bodies. Unchanged: a student POSTing to an API sends a query, not a
 * file. */
export const PROXY_BODY_MAX = 1_000_000;

/**
 * Raised with the cap. 12 s could not finish an 11 MB download on a normal
 * connection, so a larger cap without a larger deadline would have refused the
 * same file for a different reason.
 */
export const PROXY_TIMEOUT_MS = 60_000;

/** The cap in the words used everywhere it is mentioned, so prose cannot drift
 * from the constant. */
export function proxyCapText(): string {
  return `${Math.round(PROXY_RESPONSE_MAX / 1_000_000)} MB`;
}
