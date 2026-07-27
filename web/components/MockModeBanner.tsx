/**
 * Shown whenever the server is running the canned test model.
 *
 * This exists because a leftover test server was once picked up by an ordinary
 * `npm run dev` and served canned answers that looked like real model output.
 * Silence was the bug; the banner makes that state impossible to mistake.
 */
export function MockModeBanner() {
  if (process.env.CHATISA_MOCK_LLM !== "1") return null;

  return (
    <div
      role="alert"
      className="border-b-2 border-miami-red bg-corn-yellow px-4 py-2 text-center text-sm font-bold text-ink"
    >
      Test mode: every AI answer on this server is canned sample text, not a
      real model response.
    </div>
  );
}
