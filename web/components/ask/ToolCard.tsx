"use client";

import { useEffect, useState } from "react";
import {
  TOOL_LABELS,
  createdFileIds,
  openaiContainerId,
  toolSummary,
} from "@/lib/ask/tool-card";

/**
 * One tool call in an Ask Anything reply: a collapsible card showing what ran
 * and what came back. Browser code tools show code, output, and plots; the
 * server research tools (slice C) show queries and linked results; the hosted
 * sandboxes (slice E) say plainly that they ran on the provider's servers.
 * States follow the AI SDK tool part lifecycle. The card is a native <details>,
 * so it is keyboard-operable for free; running state is announced with text,
 * never colour alone.
 *
 * Files a hosted run created are rendered OUTSIDE the <details>, always
 * visible. They used to sit inside it, which meant a student who asked for a
 * PowerPoint got a collapsed grey bar reading "Ran on Anthropic's servers" and
 * no visible sign of their deck (found 2026-07-25). The generated file is the
 * deliverable, not a detail of the run.
 */

function DownloadLink({ href, name }: { href: string; name: string }) {
  return (
    <a
      href={href}
      download={name}
      className="inline-block rounded-card border border-medium-tan bg-paper px-2 py-1 text-xs font-bold hover:border-miami-red hover:text-miami-red"
    >
      Download {name}
    </a>
  );
}

/** The always-visible frame around a run's created files. */
function CreatedFiles({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-card border-2 border-miami-red bg-paper p-3">
      <p className="mb-2 text-sm font-bold">Files this run created</p>
      <div className="flex flex-wrap gap-2">{children}</div>
    </div>
  );
}

/** Anthropic streams file ids but not filenames, so each name is fetched from
 * the pass-through route's metadata view; until it arrives the link still
 * works, just with a generic name. */
function AnthropicDownloads({ fileIds }: { fileIds: string[] }) {
  const idKey = fileIds.join(",");
  const [names, setNames] = useState<Record<string, string>>({});
  useEffect(() => {
    const ids = idKey ? idKey.split(",") : [];
    let cancelled = false;
    Promise.all(
      ids.map(async (id) => {
        try {
          const res = await fetch(
            `/api/ask-anything/files/anthropic/${encodeURIComponent(id)}?meta=1`,
          );
          if (!res.ok) return null;
          const body = (await res.json()) as { filename?: string };
          return body.filename ? ([id, body.filename] as [string, string]) : null;
        } catch {
          return null;
        }
      }),
    ).then((pairs) => {
      if (cancelled) return;
      setNames(
        Object.fromEntries(
          pairs.filter((pair): pair is [string, string] => pair !== null),
        ),
      );
    });
    return () => {
      cancelled = true;
    };
  }, [idKey]);
  return (
    <>
      {fileIds.map((id, i) => (
        <DownloadLink
          key={id}
          href={`/api/ask-anything/files/anthropic/${encodeURIComponent(id)}`}
          name={names[id] ?? (fileIds.length > 1 ? `file ${i + 1}` : "the file")}
        />
      ))}
    </>
  );
}

/** Files an OpenAI interpreter run created: listed lazily through our
 * pass-through route (the stream itself only carries the container id). */
function OpenAIContainerFiles({ containerId }: { containerId: string }) {
  const [files, setFiles] = useState<
    { id: string; filename: string }[] | null
  >(null);
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/ask-anything/files/openai/${encodeURIComponent(containerId)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((body: { files?: { id: string; filename: string }[] } | null) => {
        if (!cancelled) setFiles(body?.files ?? []);
      })
      .catch(() => {
        if (!cancelled) setFiles([]);
      });
    return () => {
      cancelled = true;
    };
  }, [containerId]);
  if (files === null)
    return <p className="text-xs text-dark-tan">Checking for created files...</p>;
  if (files.length === 0) return null;
  return (
    <CreatedFiles>
      {files.map((f) => (
        <DownloadLink
          key={f.id}
          href={`/api/ask-anything/files/openai/${encodeURIComponent(containerId)}/${encodeURIComponent(f.id)}?name=${encodeURIComponent(f.filename)}`}
          name={f.filename}
        />
      ))}
    </CreatedFiles>
  );
}

/** Result body for the hosted sandboxes; null hands off to other renderers.
 * Created files are NOT rendered here: they belong outside the disclosure. */
function hostedToolBody(
  toolName: string,
  input: Record<string, unknown> | undefined,
  output: Record<string, unknown> | undefined,
): React.ReactNode | null {
  if (toolName === "code_execution") {
    const command =
      typeof input?.command === "string" && !input?.code
        ? String(input.command)
        : null;
    const stdout = typeof output?.stdout === "string" ? output.stdout : "";
    const stderr = typeof output?.stderr === "string" ? output.stderr : "";
    return (
      <div className="space-y-2 text-xs">
        <p className="text-dark-tan">
          This ran in Anthropic&apos;s hosted sandbox, not in your browser.
        </p>
        {command ? (
          <pre className="overflow-x-auto rounded-card bg-light-tan p-2">
            <code>{command}</code>
          </pre>
        ) : null}
        {stdout ? (
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-card bg-light-tan p-2">
            {stdout}
          </pre>
        ) : null}
        {stderr ? (
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-card border border-medium-tan p-2">
            {stderr}
          </pre>
        ) : null}
      </div>
    );
  }
  if (toolName === "code_interpreter") {
    const outputs =
      (output?.outputs as { type?: string; logs?: string }[] | undefined) ?? [];
    const logs = outputs
      .filter((o) => o.type === "logs" && o.logs)
      .map((o) => o.logs)
      .join("\n");
    return (
      <div className="space-y-2 text-xs">
        <p className="text-dark-tan">
          This ran in OpenAI&apos;s hosted sandbox, not in your browser.
        </p>
        {logs ? (
          <pre className="overflow-x-auto whitespace-pre-wrap rounded-card bg-light-tan p-2">
            {logs}
          </pre>
        ) : null}
      </div>
    );
  }
  return null;
}

/** The executor's output object for the code tools (lib/ask/tools contract). */
interface ToolOutput {
  status?: string;
  output?: string;
  error?: string;
  truncated?: boolean;
  plots?: number;
  ms?: number;
}

interface PaperOut {
  title?: string;
  authors?: string[];
  year?: number | null;
  venue?: string | null;
  citations?: number | null;
  url?: string | null;
  source?: string;
}

/** Result body for the research and style tools; null hands off to the
 * generic renderers. */
function serverToolBody(
  toolName: string,
  input: Record<string, unknown> | undefined,
  output: Record<string, unknown> | undefined,
): React.ReactNode | null {
  if (toolName === "search_papers") {
    const papers = (output?.papers as PaperOut[] | undefined) ?? [];
    const unavailable = (output?.unavailable as string[] | undefined) ?? [];
    return (
      <div className="space-y-2 text-xs">
        {input?.query ? (
          <p className="text-dark-tan">
            Query: <span className="font-bold">{String(input.query)}</span>
          </p>
        ) : null}
        {papers.length === 0 ? (
          <p>No papers came back for this query.</p>
        ) : (
          <ol className="list-decimal space-y-1 pl-5">
            {papers.map((p, i) => (
              <li key={i}>
                {p.url ? (
                  <a
                    href={String(p.url)}
                    target="_blank"
                    rel="noopener noreferrer nofollow"
                    className="font-bold text-accent-red underline underline-offset-2"
                  >
                    {p.title}
                    <span className="sr-only"> (opens in a new tab)</span>
                  </a>
                ) : (
                  <span className="font-bold">{p.title}</span>
                )}{" "}
                <span className="text-dark-tan">
                  {[p.year, p.venue, p.citations != null ? `${p.citations} citations` : null]
                    .filter(Boolean)
                    .join(", ")}
                </span>
              </li>
            ))}
          </ol>
        )}
        {unavailable.length > 0 ? (
          <p className="text-dark-tan">
            Unavailable this time: {unavailable.join(", ")}.
          </p>
        ) : null}
      </div>
    );
  }
  if (toolName === "get_paper") {
    const title = output?.title ? String(output.title) : null;
    const abstract = output?.abstract ? String(output.abstract) : null;
    return (
      <div className="space-y-2 text-xs">
        {title ? <p className="font-bold">{title}</p> : null}
        {abstract ? (
          <p>{abstract.length > 600 ? `${abstract.slice(0, 600)}...` : abstract}</p>
        ) : null}
        {output?.url ? (
          <a
            href={String(output.url)}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="font-bold text-accent-red underline underline-offset-2"
          >
            Open the paper
            <span className="sr-only"> (opens in a new tab)</span>
          </a>
        ) : null}
      </div>
    );
  }
  if (toolName === "read_url") {
    const text = output?.text ? String(output.text) : null;
    return (
      <div className="space-y-2 text-xs">
        {input?.url ? (
          <p className="break-all text-dark-tan">{String(input.url)}</p>
        ) : null}
        {text ? (
          <p className="whitespace-pre-wrap">
            {text.length > 700 ? `${text.slice(0, 700)}...` : text}
          </p>
        ) : null}
      </div>
    );
  }
  if (toolName === "get_miami_style") {
    const content = output?.content ? String(output.content) : null;
    return (
      <div className="space-y-2 text-xs">
        {input?.kind ? (
          <p className="text-dark-tan">
            Asset: <span className="font-bold">{String(input.kind)}</span>
          </p>
        ) : null}
        {content ? (
          <pre className="overflow-x-auto rounded-card bg-light-tan p-2">
            {content.length > 500 ? `${content.slice(0, 500)}...` : content}
          </pre>
        ) : null}
      </div>
    );
  }
  return null;
}

export function ToolCard({
  part,
  plotFor,
}: {
  part: Record<string, unknown>;
  /** Looks up a captured plot for a tool call; images stay client-side (they
   * are shown to the student, and the model is told a plot was produced). */
  plotFor: (toolCallId: string) => string | undefined;
}) {
  const toolName =
    (part.toolName as string | undefined) ??
    (typeof part.type === "string" && part.type.startsWith("tool-")
      ? (part.type as string).slice(5)
      : "tool");
  const label = TOOL_LABELS[toolName] ?? toolName;
  const state = part.state as string | undefined;
  const toolCallId = (part.toolCallId as string | undefined) ?? "";
  // Provider-executed calls whose input streamed as deltas can surface the
  // final input as its raw JSON string; normalize so the code still shows.
  let input = part.input as Record<string, unknown> | undefined;
  if (typeof part.input === "string") {
    try {
      input = JSON.parse(part.input) as Record<string, unknown>;
    } catch {
      input = { code: part.input };
    }
  }
  const output = (part.output ?? undefined) as
    | (ToolOutput & Record<string, unknown>)
    | undefined;
  const errorText = part.errorText as string | undefined;

  const running = state === "input-streaming" || state === "input-available";
  const failed =
    state === "output-error" ||
    (output != null && (output.status === "error" || output.error != null));

  const summary = toolSummary({ toolName, running, failed, input, output });
  const plot = toolCallId ? plotFor(toolCallId) : undefined;
  const serverBody = failed
    ? null
    : hostedToolBody(toolName, input, output as Record<string, unknown>) ??
      serverToolBody(toolName, input, output as Record<string, unknown>);
  const fileIds = failed ? [] : createdFileIds(toolName, output);
  const containerId = failed ? null : openaiContainerId(toolName, input);

  return (
    <div className="space-y-2">
      <details
        className="rounded-card border border-medium-tan bg-light-tan"
        open={failed || undefined}
      >
        <summary className="cursor-pointer px-3 py-2 text-sm font-bold text-dark-tan">
          {summary}
        </summary>
        <div className="space-y-2 border-t border-medium-tan bg-paper p-3">
          {typeof input?.code === "string" && input.code ? (
            <pre className="overflow-x-auto rounded-card bg-light-tan p-2 text-xs">
              <code>{String(input.code)}</code>
            </pre>
          ) : null}
          {failed ? (
            <pre className="overflow-x-auto whitespace-pre-wrap rounded-card border-2 border-miami-red p-2 text-xs">
              {String(output?.error ?? errorText ?? "The run failed.")}
            </pre>
          ) : serverBody ? (
            serverBody
          ) : output?.output ? (
            <pre className="overflow-x-auto whitespace-pre-wrap rounded-card bg-light-tan p-2 text-xs">
              {output.output}
            </pre>
          ) : null}
          {output?.truncated ? (
            <p className="text-xs text-dark-tan">
              Long output was shortened here and for the model.
            </p>
          ) : null}
          {plot ? (
            // The plot the code produced; alt text names the tool that made it.
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={plot}
              alt={`Plot produced by the ${label} run`}
              className="max-w-full rounded-card border border-medium-tan"
            />
          ) : null}
        </div>
      </details>
      {fileIds.length > 0 ? (
        <CreatedFiles>
          <AnthropicDownloads fileIds={fileIds} />
        </CreatedFiles>
      ) : null}
      {containerId ? <OpenAIContainerFiles containerId={containerId} /> : null}
    </div>
  );
}
