/**
 * The deck template filename, named once (2026-07-25). It used to be spelled
 * out in six places, which is how the system prompt came to describe a 12-slide
 * template after the file had been replaced underneath it.
 *
 * This lives in its own module, deliberately WITHOUT "server-only", because the
 * Ask Anything system prompt interpolates it and that prompt travels through
 * lib/chat/config into a "use client" component. Importing it from lib/ask/hosted
 * (which is server-only, it reads the file and holds provider keys) would break
 * the client build.
 */
export const DECK_TEMPLATE = "miami_template_by_fadel_megahed.pptx";
