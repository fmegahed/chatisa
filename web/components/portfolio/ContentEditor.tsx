"use client";

import type { ReactNode } from "react";
import type { CareerContent, ShowcaseContent, SiteContent } from "@/lib/portfolio/content";

/**
 * The editor half of the review step (2026-08-20). Every field the model
 * generated is a plain input the student can rewrite, and every list can be
 * added to, reordered, and trimmed. Nothing here calls a model: generation
 * happened in the last input step, and this is where the student takes the
 * draft over.
 */

/**
 * Label containment, not htmlFor: the same label text repeats across the
 * items of a list, so a derived id would repeat too and every control after
 * the first would be labelled by the wrong one. Wrapping associates them
 * without needing ids at all.
 */
function Text(props: { label: string; value: string; onChange: (v: string) => void; rows?: number }) {
  return (
    <label className="mt-3 block font-bold">
      {props.label}
      {props.rows ? (
        <textarea
          rows={props.rows}
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
        />
      ) : (
        <input
          value={props.value}
          onChange={(e) => props.onChange(e.target.value)}
          className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
        />
      )}
    </label>
  );
}

function List<T>(props: {
  title: string;
  items: T[];
  onChange: (items: T[]) => void;
  blank: () => T;
  render: (item: T, set: (next: T) => void) => ReactNode;
  max: number;
}) {
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= props.items.length) return;
    const next = [...props.items];
    [next[i], next[j]] = [next[j], next[i]];
    props.onChange(next);
  };
  return (
    <fieldset className="mt-4 rounded-card border border-medium-tan p-3">
      <legend className="font-bold">{props.title}</legend>
      {props.items.map((item, i) => (
        <div key={i} className="mt-2 border-t border-medium-tan pt-2 first:border-0">
          {props.render(item, (next) => props.onChange(props.items.map((x, j) => (j === i ? next : x))))}
          <div className="mt-1 flex gap-3 text-sm">
            <button type="button" className="underline" onClick={() => move(i, -1)}>Move up</button>
            <button type="button" className="underline" onClick={() => move(i, 1)}>Move down</button>
            <button type="button" className="underline" onClick={() => props.onChange(props.items.filter((_, j) => j !== i))}>Remove</button>
          </div>
        </div>
      ))}
      {props.items.length < props.max ? (
        <button type="button" className="mt-2 underline" onClick={() => props.onChange([...props.items, props.blank()])}>Add</button>
      ) : null}
    </fieldset>
  );
}

const csv = (s: string[]) => s.join(", ");
const uncsv = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);

function CareerEditor(props: { value: CareerContent; onChange: (c: CareerContent) => void }) {
  const c = props.value;
  const set = (p: Partial<CareerContent>) => props.onChange({ ...c, ...p });
  return (
    <div>
      <Text label="Site title" value={c.siteTitle} onChange={(siteTitle) => set({ siteTitle })} />
      <Text label="Headline" value={c.headline} onChange={(headline) => set({ headline })} />
      <Text label="About" value={c.about} onChange={(about) => set({ about })} rows={5} />
      <List
        title="Skill groups" items={c.skillGroups} max={6}
        onChange={(skillGroups) => set({ skillGroups })}
        blank={() => ({ title: "", skills: [] })}
        render={(g, s) => (
          <>
            <Text label="Group" value={g.title} onChange={(title) => s({ ...g, title })} />
            <Text label="Skills (comma separated)" value={csv(g.skills)} onChange={(v) => s({ ...g, skills: uncsv(v) })} />
          </>
        )}
      />
      <List
        title="Projects" items={c.projects} max={5}
        onChange={(projects) => set({ projects })}
        blank={() => ({ slug: "project", title: "", blurb: "", skills: [], externalUrl: null })}
        render={(p, s) => (
          <>
            <Text label="Title" value={p.title} onChange={(title) => s({ ...p, title })} />
            <Text label="Blurb" value={p.blurb} onChange={(blurb) => s({ ...p, blurb })} rows={3} />
            <Text label="Skills (comma separated)" value={csv(p.skills)} onChange={(v) => s({ ...p, skills: uncsv(v) })} />
            <Text label="External link" value={p.externalUrl ?? ""} onChange={(v) => s({ ...p, externalUrl: v || null })} />
          </>
        )}
      />
      <List
        title="Courses" items={c.courses} max={8}
        onChange={(courses) => set({ courses })}
        blank={() => ({ code: "", why: "" })}
        render={(x, s) => (
          <>
            <Text label="Course" value={x.code} onChange={(code) => s({ ...x, code })} />
            <Text label="Why it matters" value={x.why} onChange={(why) => s({ ...x, why })} />
          </>
        )}
      />
      <List
        title="Experience" items={c.experience} max={6}
        onChange={(experience) => set({ experience })}
        blank={() => ({ org: "", role: "", dates: "", bullets: [] })}
        render={(e, s) => (
          <>
            <Text label="Organization" value={e.org} onChange={(org) => s({ ...e, org })} />
            <Text label="Role" value={e.role} onChange={(role) => s({ ...e, role })} />
            <Text label="Dates" value={e.dates} onChange={(dates) => s({ ...e, dates })} />
            <Text
              label="Bullets (one per line)"
              value={e.bullets.join("\n")}
              onChange={(v) => s({ ...e, bullets: v.split("\n").filter(Boolean) })}
              rows={3}
            />
          </>
        )}
      />
      <List
        title="Education" items={c.education} max={3}
        onChange={(education) => set({ education })}
        blank={() => ({ school: "", degree: "", dates: "" })}
        render={(e, s) => (
          <>
            <Text label="School" value={e.school} onChange={(school) => s({ ...e, school })} />
            <Text label="Degree" value={e.degree} onChange={(degree) => s({ ...e, degree })} />
            <Text label="Dates" value={e.dates} onChange={(dates) => s({ ...e, dates })} />
          </>
        )}
      />
    </div>
  );
}

function ShowcaseEditor(props: {
  value: ShowcaseContent;
  onChange: (c: ShowcaseContent) => void;
  figures: string[];
}) {
  const c = props.value;
  const set = (p: Partial<ShowcaseContent>) => props.onChange({ ...c, ...p });
  return (
    <div>
      <Text label="Title" value={c.title} onChange={(title) => set({ title })} />
      <Text label="Tagline" value={c.tagline} onChange={(tagline) => set({ tagline })} />
      <Text label="The problem" value={c.problem} onChange={(problem) => set({ problem })} rows={4} />
      <Text label="The data" value={c.data} onChange={(data) => set({ data })} rows={4} />
      <Text label="Approach" value={c.approach} onChange={(approach) => set({ approach })} rows={5} />
      <List
        title="Findings" items={c.findings} max={6}
        onChange={(findings) => set({ findings })}
        blank={() => ({ heading: "", body: "", figure: null })}
        render={(f, s) => (
          <>
            <Text label="Heading" value={f.heading} onChange={(heading) => s({ ...f, heading })} />
            <Text label="Body" value={f.body} onChange={(body) => s({ ...f, body })} rows={3} />
            <label className="mt-2 block font-bold">
              Figure
              <select
                value={f.figure ?? ""}
                onChange={(e) => s({ ...f, figure: e.target.value || null })}
                className="mt-1 block rounded-card border border-medium-tan p-1 font-normal"
              >
                <option value="">None</option>
                {props.figures.map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </label>
          </>
        )}
      />
      <List
        title="Deliverables" items={c.deliverables} max={12}
        onChange={(deliverables) => set({ deliverables })}
        blank={() => ({ label: "", path: "" })}
        render={(d, s) => (
          <>
            <Text label="Label" value={d.label} onChange={(label) => s({ ...d, label })} />
            <Text label="Path in the repository" value={d.path} onChange={(path) => s({ ...d, path })} />
          </>
        )}
      />
      <Text label="Skills (comma separated)" value={csv(c.skills)} onChange={(v) => set({ skills: uncsv(v) })} />
      <Text label="What I would do next" value={c.nextSteps} onChange={(nextSteps) => set({ nextSteps })} rows={3} />
    </div>
  );
}

export function ContentEditor(props: {
  value: SiteContent;
  onChange: (next: SiteContent) => void;
  figures: string[];
}) {
  return props.value.kind === "career"
    ? <CareerEditor value={props.value.content} onChange={(content) => props.onChange({ kind: "career", content })} />
    : <ShowcaseEditor value={props.value.content} onChange={(content) => props.onChange({ kind: "showcase", content })} figures={props.figures} />;
}
