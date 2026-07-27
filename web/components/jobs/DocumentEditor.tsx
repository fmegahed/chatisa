"use client";

import { useState } from "react";
import type { CoverLetterContent, ResumeContent } from "@/lib/documents/schema";

interface FlaggedClaim {
  text: string;
  sourceLine: string | null;
  verdict: string;
  note: string | null;
}

/**
 * Reviewing and editing a generated document.
 *
 * The student edits every line before anything leaves the app. That is the
 * pedagogical point, not a formality: these are claims they will have to defend
 * in an interview, so they should be claims they have actually read.
 *
 * Flagged lines are shown with a prominent warning that can be overridden
 * rather than blocked (user decision, 2026-07-21). The check is a heuristic and
 * will sometimes be wrong about something the student can legitimately defend,
 * so the final call is theirs. What we owe them is that the risk is impossible
 * to miss, not that the decision is taken away.
 */

function flagFor(
  flagged: FlaggedClaim[],
  text: string,
): FlaggedClaim | undefined {
  return flagged.find((f) => f.text.trim() === text.trim());
}

function Warning(props: { flag: FlaggedClaim }) {
  return (
    <p
      role="note"
      className="mt-1 border-l-4 border-miami-red bg-light-tan px-3 py-2 text-sm"
    >
      <strong className="text-miami-red">Check this line.</strong>{" "}
      {props.flag.note}
      {props.flag.sourceLine ? (
        <span className="mt-1 block text-dark-tan">
          Closest line in your resume: {props.flag.sourceLine}
        </span>
      ) : null}
    </p>
  );
}

export function ResumeEditor(props: {
  content: ResumeContent;
  flagged: FlaggedClaim[];
  onChange: (content: ResumeContent) => void;
  disabled: boolean;
}) {
  const { content } = props;

  function editBullet(
    sectionIndex: number,
    entryIndex: number,
    bulletIndex: number,
    text: string,
  ) {
    const next: ResumeContent = structuredClone(content);
    next.sections[sectionIndex].entries[entryIndex].bullets[bulletIndex].text =
      text;
    props.onChange(next);
  }

  return (
    <div className="flex flex-col gap-5">
      {content.sections.map((section, si) => (
        <section
          key={`${section.heading}-${si}`}
          className="rounded-card border border-medium-tan bg-paper p-5"
        >
          <h3 className="text-lg font-bold uppercase">{section.heading}</h3>

          {section.entries.map((entry, ei) => (
            <div key={`${entry.organization}-${ei}`} className="mt-4">
              <p className="font-bold">
                {entry.organization}
                {entry.location ? `, ${entry.location}` : ""}
              </p>
              <p className="text-sm italic text-dark-tan">
                {entry.title}
                {entry.dates ? ` · ${entry.dates}` : ""}
              </p>

              <ul className="mt-2 flex flex-col gap-3">
                {entry.bullets.map((bullet, bi) => {
                  const flag = flagFor(props.flagged, bullet.text);
                  const id = `bullet-${si}-${ei}-${bi}`;
                  return (
                    <li key={id}>
                      <label htmlFor={id} className="sr-only">
                        Bullet {bi + 1} of {entry.title}
                      </label>
                      <textarea
                        id={id}
                        value={bullet.text}
                        onChange={(e) => editBullet(si, ei, bi, e.target.value)}
                        disabled={props.disabled}
                        rows={2}
                        className={[
                          "w-full rounded-card border bg-paper p-2 text-sm",
                          flag ? "border-2 border-miami-red" : "border-medium-tan",
                        ].join(" ")}
                        aria-describedby={flag ? `${id}-warning` : undefined}
                      />
                      {flag ? (
                        <div id={`${id}-warning`}>
                          <Warning flag={flag} />
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </section>
      ))}

      {content.skills.length > 0 ? (
        <section className="rounded-card border border-medium-tan bg-paper p-5">
          <h3 className="text-lg font-bold uppercase">Skills / Certifications</h3>
          <label htmlFor="skills" className="mt-2 block text-sm text-dark-tan">
            One per line
          </label>
          <textarea
            id="skills"
            value={content.skills.join("\n")}
            onChange={(e) =>
              props.onChange({
                ...content,
                skills: e.target.value.split("\n").filter((s) => s.trim() !== ""),
              })
            }
            disabled={props.disabled}
            rows={4}
            className="mt-1 w-full rounded-card border border-medium-tan bg-paper p-2 text-sm"
          />
        </section>
      ) : null}
    </div>
  );
}

export function CoverLetterEditor(props: {
  content: CoverLetterContent;
  flagged: FlaggedClaim[];
  onChange: (content: CoverLetterContent) => void;
  disabled: boolean;
}) {
  const { content } = props;

  function editParagraph(index: number, text: string) {
    const next: CoverLetterContent = structuredClone(content);
    next.paragraphs[index].text = text;
    props.onChange(next);
  }

  const words = content.paragraphs.reduce(
    (sum, p) => sum + p.text.trim().split(/\s+/).filter(Boolean).length,
    0,
  );

  return (
    <div className="rounded-card border border-medium-tan bg-paper p-5">
      <p className="font-bold">{content.salutation}</p>

      <div className="mt-3 flex flex-col gap-4">
        {content.paragraphs.map((paragraph, i) => {
          const flag = flagFor(props.flagged, paragraph.text);
          const id = `paragraph-${i}`;
          return (
            <div key={id}>
              <label htmlFor={id} className="block text-sm font-bold">
                Paragraph {i + 1}
                {paragraph.addresses ? (
                  <span className="ml-2 font-normal text-dark-tan">
                    answers: {paragraph.addresses}
                  </span>
                ) : null}
              </label>
              <textarea
                id={id}
                value={paragraph.text}
                onChange={(e) => editParagraph(i, e.target.value)}
                disabled={props.disabled}
                rows={4}
                className={[
                  "mt-1 w-full rounded-card border bg-paper p-2 text-sm",
                  flag ? "border-2 border-miami-red" : "border-medium-tan",
                ].join(" ")}
                aria-describedby={flag ? `${id}-warning` : undefined}
              />
              {flag ? (
                <div id={`${id}-warning`}>
                  <Warning flag={flag} />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <p className="mt-3 font-bold">{content.closing}</p>
      <p className="mt-1">{content.name}</p>

      {/* The school's own finished letter runs about 205 words, so this is a
          concrete target rather than a vague "one page". */}
      <p role="status" className="mt-3 text-sm text-dark-tan">
        {words} words. The Farmer School example runs about 205.
        {words > 320 ? " Yours is long enough that it will spill past one page." : ""}
      </p>
    </div>
  );
}

/** Banner above the editor summarising what needs attention. */
export function GroundingBanner(props: {
  flagged: FlaggedClaim[];
  message: string | null;
}) {
  const [dismissed, setDismissed] = useState(false);
  if (props.flagged.length === 0 || dismissed) return null;

  return (
    <div
      role="alert"
      className="mb-5 rounded-card border-2 border-miami-red bg-paper p-4"
    >
      <h3 className="font-bold text-miami-red">
        Read these lines before you send anything
      </h3>
      <p className="mt-1">{props.message}</p>
      <p className="mt-2 text-sm">
        You can send it anyway. This check is a rough one and it is sometimes
        wrong. But an interviewer will ask you about anything on your resume, so
        make sure you can talk about every line on it.
      </p>
      <button
        type="button"
        onClick={() => setDismissed(true)}
        className="mt-3 rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-light-tan"
      >
        I have read these
      </button>
    </div>
  );
}
