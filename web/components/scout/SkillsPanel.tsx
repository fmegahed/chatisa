"use client";

import { useMemo, useState } from "react";
import { SKILLS, getSkill } from "@/lib/scout/taxonomy";
import { profileStrengths, strengthWord } from "@/lib/scout/matching";
import { COURSE_SKILLS, type CourseSkillLevel } from "@/lib/scout/course-skills";
import type {
  ProfileExtra,
  ProjectRecord,
  SkillOverride,
} from "@/lib/scout/profile-store";
import { demandRanking, type FeedPosting } from "@/lib/scout/feed-types";

/**
 * The live "skills you are building" panel: the feedback loop the first
 * version was missing (user feedback, 2026-07-29). Recomputed from
 * profileStrengths on every course toggle; levels are words, never meters.
 */

const CATEGORY_LABELS: Record<string, string> = {
  programming: "Programming",
  analytics: "Analytics",
  machine_learning_ai: "Machine Learning and AI",
  data_management: "Data Management",
  visualization_bi: "Visualization and BI",
  information_systems: "Information Systems",
  security_risk: "Security and Risk",
  professional: "Professional",
};

const LEVEL_HELP: Record<CourseSkillLevel, string> = {
  anchor: "I can show real work with this",
  applied: "I have used it repeatedly",
  exposure: "I know the basics",
};

export function SkillsPanel(props: {
  strengths: Map<string, number>;
  draftCourses: Set<string>;
  draftExtras: ProfileExtra[];
  overrides: SkillOverride[];
  isFirstRun: boolean;
  projects: ProjectRecord[];
  postings: FeedPosting[];
  onAddManual: (skillId: string, level: CourseSkillLevel) => void;
  onRemoveExtra: (skillId: string) => void;
  onSetOverride: (skillId: string, level: SkillOverride["level"] | null) => void;
}) {
  const [manualSkill, setManualSkill] = useState("");
  const [manualLevel, setManualLevel] = useState<CourseSkillLevel>("applied");

  // First run edits a draft the parent has not saved yet, so strengths are
  // recomputed here from the draft; saved profiles get the parent's map
  // (which already includes built-project contributions).
  const strengths = useMemo(() => {
    if (!props.isFirstRun) return props.strengths;
    return profileStrengths(
      [...props.draftCourses],
      props.draftExtras,
      props.overrides,
    );
  }, [
    props.isFirstRun,
    props.strengths,
    props.draftCourses,
    props.draftExtras,
    props.overrides,
  ]);

  const sourcesOf = useMemo(() => {
    return (skillId: string): string[] => {
      const out = COURSE_SKILLS.filter(
        (l) => l.skillId === skillId && props.draftCourses.has(l.course),
      ).map((l) => l.course);
      for (const e of props.draftExtras) {
        if (e.skillId !== skillId) continue;
        out.push(
          e.source === "resume"
            ? "your resume"
            : e.source === "freeform"
              ? "your experience"
              : "added by you",
        );
      }
      for (const p of props.projects) {
        if (p.repoUrl && p.skillIds.includes(skillId)) out.push(p.repoName);
      }
      return out;
    };
  }, [props.draftCourses, props.draftExtras, props.projects]);

  const grouped = useMemo(() => {
    const rows = [...strengths.entries()]
      .map(([skillId, strength]) => ({ skillId, strength }))
      .sort((a, b) => b.strength - a.strength);
    const byCategory = new Map<string, typeof rows>();
    for (const row of rows) {
      const cat = getSkill(row.skillId)?.category ?? "professional";
      byCategory.set(cat, [...(byCategory.get(cat) ?? []), row]);
    }
    return [...byCategory.entries()];
  }, [strengths]);

  const demand = useMemo(
    () => demandRanking(props.postings, 10),
    [props.postings],
  );

  const extraIds = new Set(props.draftExtras.map((e) => e.skillId));

  return (
    <section aria-labelledby="skills-panel">
      <h2 id="skills-panel" className="text-2xl">
        Skills you are building
      </h2>
      {strengths.size === 0 ? (
        <p className="mt-2 rounded-card border border-medium-tan bg-light-tan p-4">
          Check a course on the left and your skills appear here.
        </p>
      ) : (
        <div className="mt-2 max-h-[28rem] space-y-4 overflow-y-auto rounded-card border border-medium-tan bg-paper p-4">
          {grouped.map(([category, rows]) => (
            <div key={category}>
              <h3 className="font-bold">
                {CATEGORY_LABELS[category] ?? category}
              </h3>
              <ul className="mt-1 space-y-1">
                {rows.map((row) => {
                  const label = getSkill(row.skillId)?.label ?? row.skillId;
                  const override = props.overrides.find(
                    (o) => o.skillId === row.skillId,
                  );
                  return (
                    <li
                      key={row.skillId}
                      className="flex flex-wrap items-baseline gap-x-2"
                    >
                      <span>{label}</span>
                      {/* The word is editable: the student's own judgment
                          beats the computation (user feedback, 2026-07-29). */}
                      <select
                        value={override?.level ?? "auto"}
                        aria-label={`Your level for ${label}`}
                        onChange={(e) =>
                          props.onSetOverride(
                            row.skillId,
                            e.target.value === "auto"
                              ? null
                              : (e.target.value as SkillOverride["level"]),
                          )
                        }
                        className="rounded-card border border-medium-tan bg-paper px-1 py-0.5 font-bold"
                      >
                        <option value="auto">
                          {override
                            ? "Auto (from your courses)"
                            : `${strengthWord(row.strength)} (from your courses)`}
                        </option>
                        <option value="strong">Strong</option>
                        <option value="working">Working</option>
                        <option value="introduced">Introduced</option>
                      </select>
                      <span className="text-dark-tan">
                        ({sourcesOf(row.skillId).join(", ")}
                        {override ? ", set by you" : ""})
                      </span>
                      {extraIds.has(row.skillId) ? (
                        <button
                          type="button"
                          onClick={() => props.onRemoveExtra(row.skillId)}
                          className="underline"
                          aria-label={`Remove ${label} from your added skills`}
                        >
                          Remove
                        </button>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 rounded-card border border-medium-tan bg-paper p-4">
        <h3 className="font-bold">Add a skill yourself</h3>
        <p className="text-dark-tan">
          For anything the courses and resume missed. Be ready to back it up
          in an interview.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <label>
            <span className="sr-only">Skill to add</span>
            <select
              value={manualSkill}
              onChange={(e) => setManualSkill(e.target.value)}
              className="rounded-card border border-medium-tan bg-paper px-2 py-1"
            >
              <option value="">Pick a skill...</option>
              {SKILLS.filter((s) => !strengths.has(s.id)).map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          <fieldset className="flex flex-wrap gap-3">
            <legend className="sr-only">How deep does it go</legend>
            {(Object.keys(LEVEL_HELP) as CourseSkillLevel[]).map((l) => (
              <label key={l} className="flex items-center gap-1">
                <input
                  type="radio"
                  name="manual-level"
                  checked={manualLevel === l}
                  onChange={() => setManualLevel(l)}
                />
                <span>{LEVEL_HELP[l]}</span>
              </label>
            ))}
          </fieldset>
          <button
            type="button"
            disabled={!manualSkill}
            onClick={() => {
              props.onAddManual(manualSkill, manualLevel);
              setManualSkill("");
            }}
            className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan disabled:border-medium-gray disabled:text-medium-gray"
          >
            Add
          </button>
        </div>
      </div>

      {demand.length > 0 ? (
        <div className="mt-4 rounded-card border border-medium-tan bg-light-tan p-4">
          <h3 className="font-bold">This week&apos;s most-wanted skills</h3>
          <p className="text-dark-tan">
            Across {props.postings.length} current postings, with where you
            stand.
          </p>
          <ul className="mt-1 space-y-1">
            {demand.map((d) => {
              const s = strengths.get(d.skillId) ?? 0;
              return (
                <li key={d.skillId} className="flex flex-wrap items-baseline gap-x-2">
                  <span>{getSkill(d.skillId)?.label ?? d.skillId}</span>
                  <span className="text-dark-tan">({d.count} postings)</span>
                  <span className="font-bold">
                    {s > 0 ? strengthWord(s) : "Not yet"}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
