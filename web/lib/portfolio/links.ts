/**
 * Link normalization for Portfolio Builder (2026-08-20). Students type
 * "linkedin.com/in/ada", not "https://linkedin.com/in/ada", and a bare domain
 * is not a URL: it used to fail the generation route's z.url() and 400 the
 * whole payload, losing everything the student had entered. So the browser
 * repairs what it can before sending, and the route drops what it cannot
 * parse instead of rejecting the request.
 */

/** host.tld, optionally with a path, and no scheme in front of it. */
const BARE_DOMAIN = /^[\w.-]+\.[a-z]{2,}(\/\S*)?$/i;

/**
 * Returns the URL a link should be sent as, or null when the text is not a
 * URL at all. Empty input is null: an untouched optional field is not an
 * error.
 */
export function normalizeUrl(raw: string): string | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const candidate = BARE_DOMAIN.test(trimmed) ? `https://${trimmed}` : trimmed;
  try {
    return new URL(candidate).toString();
  } catch {
    return null;
  }
}
