"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { RunnableCode } from "@/components/run/RunnableCode";
import { CopyButton } from "@/components/chat/CopyButton";
import { languageFromClassName, runnerFor } from "@/lib/run/languages";
import { normalizeMathDelimiters } from "@/lib/chat/math";

/**
 * Model output renderer. react-markdown does not render raw HTML, so model
 * text cannot inject markup. Links are treated as untrusted. TeX between
 * $ / $$ (or the \( \) and \[ \] forms, normalized first) renders via KaTeX;
 * malformed TeX shows as-is rather than crashing the reply.
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="chat-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }]]}
        components={{
          a: ({ href, children }) => (
            <a
              href={href}
              rel="noopener noreferrer nofollow"
              target="_blank"
              className="font-bold text-accent-red underline underline-offset-2"
            >
              {children}
              <span className="sr-only"> (opens in a new tab)</span>
            </a>
          ),
          pre: ({ children }) => {
            const code = extractText(children);
            // react-markdown puts the fence language on the inner <code>
            // element's className as "language-xxx"; a runnable one gets a Run
            // button and an output panel.
            const language = languageFromClassName(codeClassName(children));
            const runner = runnerFor(language);
            if (runner) {
              return (
                <RunnableCode language={runner} code={code}>
                  {children}
                </RunnableCode>
              );
            }
            return (
              <figure className="my-3">
                <figcaption className="mb-1 flex justify-end">
                  <CopyButton text={code} />
                </figcaption>
                {/*
                  Focusable so keyboard users can scroll long lines
                  (WCAG 2.1.1). Labelled so the region is announced.
                */}
                <pre
                  tabIndex={0}
                  role="region"
                  aria-label="Code sample"
                  className="overflow-x-auto rounded-card border border-medium-tan bg-light-tan p-3 text-sm"
                >
                  {children}
                </pre>
              </figure>
            );
          },
          table: ({ children }) => (
            <div
              tabIndex={0}
              role="region"
              aria-label="Table"
              className="my-3 overflow-x-auto"
            >
              <table className="w-full border-collapse text-sm">
                {children}
              </table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-medium-tan bg-light-tan p-2 text-left">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-medium-tan p-2">{children}</td>
          ),
        }}
      >
        {normalizeMathDelimiters(children)}
      </ReactMarkdown>
    </div>
  );
}

/** The className of the inner <code> child, where react-markdown records the
 * fence language. The <pre> receives that <code> element as its only child. */
function codeClassName(children: React.ReactNode): string | undefined {
  const child = Array.isArray(children) ? children[0] : children;
  if (child && typeof child === "object" && "props" in child) {
    const props = (child as { props?: { className?: string } }).props;
    return props?.className;
  }
  return undefined;
}

/** Recursively pull plain text out of a rendered node tree. */
function extractText(node: React.ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map(extractText).join("");
  if (typeof node === "object" && "props" in (node as never)) {
    const props = (node as { props?: { children?: React.ReactNode } }).props;
    return extractText(props?.children);
  }
  return "";
}
