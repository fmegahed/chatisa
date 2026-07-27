import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { tool, uploadFile, type ModelMessage, type Tool } from "ai";
import { z } from "zod";
import { PIE_POLICY, paletteRules, portableRules } from "@/lib/ask/chart-style";
import { DECK_TEMPLATE } from "@/lib/ask/deck-template";
import { logger } from "@/lib/log";

/**
 * Hosted code execution for Ask Anything (slice E): the provider-run sandboxes
 * that build real files (PowerPoints via python-pptx, Word, Excel), which the
 * browser's WASM runtimes cannot. Both roster providers execute their own tool
 * natively inside the streaming response, so there is no cross-provider
 * bridge: an OpenAI chat uses OpenAI's code interpreter, an Anthropic chat
 * uses Anthropic's code execution container.
 *
 * The Miami deck template rides along: for OpenAI it is attached to the
 * interpreter container's file list (materialized only when the tool is
 * actually used); for Anthropic it is injected as a container_upload file part
 * when the student's message asks for a deck or document (a container_upload
 * block forces a container, so it must not ride on every request). The
 * template is uploaded to each provider once per server process and the
 * file id cached.
 */

export type HostedProvider = "anthropic" | "openai";

export const PPTX_MIME =
  "application/vnd.openxmlformats-officedocument.presentationml.presentation";

// Re-exported so server callers can keep importing it from here; the constant
// itself lives in a client-safe module (see lib/ask/deck-template).
export { DECK_TEMPLATE };

/** The student is asking for a generated file (deck, document, workbook).
 * Gates the Anthropic template injection and nothing else; the tool itself is
 * always declared and the system prompt carries the real routing rules. */
export function wantsGeneratedFile(lastUserText: string): boolean {
  return /power\s?point|pptx|slide deck|slides|presentation|\bdeck\b|docx|word (document|file|report)|xlsx|excel (file|workbook)|spreadsheet/i.test(
    lastUserText,
  );
}

function anthropicProvider() {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  return apiKey ? createAnthropic({ apiKey }) : null;
}

function openaiProvider() {
  const apiKey = process.env.OPENAI_API_KEY;
  return apiKey ? createOpenAI({ apiKey }) : null;
}

// One upload per provider per server process; a failed upload clears the slot
// so the next request retries instead of caching the failure.
const templateUploads = new Map<HostedProvider, Promise<string | null>>();

async function uploadTemplate(provider: HostedProvider): Promise<string | null> {
  try {
    const api = provider === "anthropic" ? anthropicProvider() : openaiProvider();
    if (!api) return null;
    const bytes = await readFile(
      path.join(process.cwd(), "assets", "brand", DECK_TEMPLATE),
    );
    const result = await uploadFile({
      api,
      data: new Uint8Array(bytes),
      mediaType: PPTX_MIME,
      filename: DECK_TEMPLATE,
    });
    const ref = result.providerReference as Record<string, string>;
    const id = ref[provider] ?? Object.values(ref)[0] ?? null;
    logger.info({ provider, templateFileId: id }, "miami deck template uploaded");
    return id;
  } catch (err) {
    logger.warn(
      { provider, err: String(err) },
      "miami deck template upload failed; decks will be unbranded",
    );
    return null;
  }
}

/** The provider-side file id of the Miami deck template, uploaded on first
 * use. Null (never throwing) when the provider is unconfigured or the upload
 * fails: hosted execution still works, decks just start from scratch. */
export async function templateFileId(
  provider: HostedProvider,
): Promise<string | null> {
  if (process.env.CHATISA_MOCK_LLM === "1") return `mock-template-${provider}`;
  let pending = templateUploads.get(provider);
  if (!pending) {
    pending = uploadTemplate(provider);
    templateUploads.set(provider, pending);
  }
  const id = await pending;
  if (id === null) templateUploads.delete(provider);
  return id;
}

/**
 * The hosted execution tool for the chat's provider. Declared on every
 * request (a declaration alone costs nothing; containers spin up only when
 * the model invokes the tool). In mock mode there is nothing to declare: the
 * mock scripts the provider-executed parts itself.
 */
export function hostedToolsFor(
  provider: HostedProvider,
  openaiTemplateId: string | null,
): Record<string, Tool> {
  if (process.env.CHATISA_MOCK_LLM === "1") {
    // A schema-only stand-in: without a declared tool, streamText drops the
    // streamed tool input on the floor (found 2026-07-24: the card lost its
    // code). No execute and the mock marks its calls providerExecuted, so
    // neither the client executor nor auto-send ever touches it.
    return {
      code_execution: tool({
        inputSchema: z.looseObject({}),
      }) as unknown as Tool,
    };
  }
  if (provider === "openai") {
    const prov = openaiProvider();
    if (!prov) return {};
    return {
      // The generics of provider-executed tools do not sit inside Tool's
      // union; the runtime object is exactly what streamText expects.
      code_interpreter: prov.tools.codeInterpreter(
        openaiTemplateId
          ? { container: { fileIds: [openaiTemplateId] } }
          : {},
      ) as unknown as Tool,
    };
  }
  const prov = anthropicProvider();
  if (!prov) return {};
  return {
    code_execution: prov.tools.codeExecution_20260120() as unknown as Tool,
  };
}

/**
 * A user message that puts the Miami template into Anthropic's execution
 * container (a `container_upload` block via the file part's provider option).
 * Inserted just before the student's latest message, and only when that
 * message asks for a generated file, because the block itself forces a
 * container into existence.
 *
 * The note carries the chart rules too (2026-07-25). This is the moment the
 * model opens the template, so it is the right place for the palette and the
 * title contract; the container has no network, so the note also says plainly
 * which packages must not be imported there.
 */
export function anthropicTemplateMessage(fileId: string): ModelMessage {
  return {
    role: "user",
    content: [
      {
        type: "file",
        mediaType: PPTX_MIME,
        filename: DECK_TEMPLATE,
        data: { type: "reference", reference: { anthropic: fileId } },
        providerOptions: { anthropic: { containerUpload: true } },
      },
      {
        type: "text",
        text: [
          `(System note: the Miami University PowerPoint template ${DECK_TEMPLATE} is uploaded into your code execution container. Build any requested deck FROM this template with python-pptx, keeping its layouts and branding.`,
          "",
          "Any chart you draw for it follows the house style:",
          paletteRules(),
          portableRules(),
          "",
          PIE_POLICY,
          "",
          "This container has no network, so matplotlib only: do not import adjustText, highlight_text, ggrepel, or ggtext here. Data annotations are optional in a deck chart; the palette, the descriptive title, and the insight subtitle are not.)",
        ].join("\n"),
      },
    ],
  };
}
