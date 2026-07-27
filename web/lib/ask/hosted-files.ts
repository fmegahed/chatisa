import "server-only";

/**
 * Retrieval of files created by the hosted code-execution sandboxes (slice E).
 * The provider stores the file; ChatISA STREAMS it through to the student and
 * keeps nothing (ADR-022). Ids are validated against the providers' formats
 * before any request is made, and filenames are sanitized before landing in a
 * Content-Disposition header.
 */

export const ANTHROPIC_FILE_ID = /^file[_-][A-Za-z0-9_-]{4,64}$/;
export const OPENAI_CONTAINER_ID = /^cntr[_-][A-Za-z0-9_-]{4,64}$/;
export const OPENAI_FILE_ID = /^cfile[_-][A-Za-z0-9_-]{4,64}$/;

const EXT_MIME: Record<string, string> = {
  pptx: "application/vnd.openxmlformats-officedocument.presentationml.presentation",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  xlsx: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  csv: "text/csv",
  pdf: "application/pdf",
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  txt: "text/plain",
  md: "text/plain",
  json: "application/json",
  html: "text/html",
};

export function mediaTypeForName(name: string): string {
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  return EXT_MIME[ext] ?? "application/octet-stream";
}

/** Display-only filename: control characters and path separators stripped,
 * length capped; never used as a path. */
export function safeDownloadName(raw: string, fallback: string): string {
  const base = raw.split(/[\\/]/).pop() ?? "";
  const cleaned = base
    .replace(/[\u0000-\u001f\u007f"\\]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120);
  return cleaned.length > 0 ? cleaned : fallback;
}

/** A minimal but structurally valid zip (empty central directory), used as the
 * mock-mode stand-in for generated Office files. */
export function mockZipBytes(): Uint8Array {
  return new Uint8Array([
    0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0,
  ]);
}

export interface HostedFileInfo {
  id: string;
  filename: string;
  sizeBytes: number | null;
}

const ANTHROPIC_HEADERS = () => ({
  "x-api-key": process.env.ANTHROPIC_API_KEY ?? "",
  "anthropic-version": "2023-06-01",
  "anthropic-beta": "files-api-2025-04-14",
});

const OPENAI_HEADERS = () => ({
  authorization: `Bearer ${process.env.OPENAI_API_KEY ?? ""}`,
});

/** Metadata (filename) for an Anthropic-stored file. */
export async function anthropicFileMeta(
  fileId: string,
): Promise<{ filename: string; mimeType: string | null } | null> {
  const res = await fetch(`https://api.anthropic.com/v1/files/${fileId}`, {
    headers: ANTHROPIC_HEADERS(),
    signal: AbortSignal.timeout(15_000),
  });
  if (!res.ok) return null;
  const meta = (await res.json()) as { filename?: string; mime_type?: string };
  return {
    filename: meta.filename ?? fileId,
    mimeType: meta.mime_type ?? null,
  };
}

/** The content stream of an Anthropic-stored file (downloadable outputs of
 * the code execution tool). */
export async function anthropicFileContent(
  fileId: string,
): Promise<Response> {
  return fetch(`https://api.anthropic.com/v1/files/${fileId}/content`, {
    headers: ANTHROPIC_HEADERS(),
    signal: AbortSignal.timeout(60_000),
  });
}

/** Files inside an OpenAI code-interpreter container, uploads excluded (the
 * student should be offered what the run CREATED, not the template back). */
export async function openaiContainerFiles(
  containerId: string,
): Promise<HostedFileInfo[] | null> {
  const res = await fetch(
    `https://api.openai.com/v1/containers/${containerId}/files?limit=100`,
    { headers: OPENAI_HEADERS(), signal: AbortSignal.timeout(15_000) },
  );
  if (!res.ok) return null;
  const body = (await res.json()) as {
    data?: { id: string; path?: string; bytes?: number; source?: string }[];
  };
  return (body.data ?? [])
    .filter((f) => f.source !== "user")
    .map((f) => ({
      id: f.id,
      filename: safeDownloadName(f.path ?? "", f.id),
      sizeBytes: f.bytes ?? null,
    }));
}

export async function openaiContainerFileContent(
  containerId: string,
  fileId: string,
): Promise<Response> {
  return fetch(
    `https://api.openai.com/v1/containers/${containerId}/files/${fileId}/content`,
    { headers: OPENAI_HEADERS(), signal: AbortSignal.timeout(60_000) },
  );
}
