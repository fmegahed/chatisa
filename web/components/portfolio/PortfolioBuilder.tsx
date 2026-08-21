"use client";

import { useEffect, useReducer, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import {
  CAREER_STEPS, SHOWCASE_STEPS, initialDraft,
  type Action, type Draft, type Step,
} from "@/lib/portfolio/draft";
import {
  careerSite, getDraft, loadSites, migrateJobScoutPortfolio, newSiteId, removeSite,
  type SiteRecord,
} from "@/lib/portfolio/store";
import { ModeStep } from "./ModeStep";
import { ResumeStep } from "./career/ResumeStep";
import { ClassesStep } from "./career/ClassesStep";
import { ProjectsStep } from "./career/ProjectsStep";
import { DetailsStep } from "./career/DetailsStep";
import { CourseStep } from "./showcase/CourseStep";
import { FilesStep } from "./showcase/FilesStep";
import { StoryStep } from "./showcase/StoryStep";
import { ReviewStep } from "./ReviewStep";

/**
 * The wizard shell: one reducer holds the whole draft, the mode chosen on the
 * first step decides which ordered list of steps follows, and each step gets
 * the draft plus a patch function and its Back/Next wiring. Generation itself
 * lives in the last input step of each mode (DetailsStep, StoryStep), whose
 * Next button reads "Generate".
 */

function reducer(state: Draft, action: Action): Draft {
  return action.type === "reset" ? action.draft : { ...state, ...action.patch };
}

/**
 * A student has one career portfolio. Picking "Career portfolio" when one
 * already exists therefore continues that site rather than minting a second
 * record that would publish over the first one's repository.
 */
function pickMode(mode: "career" | "showcase"): Partial<Draft> {
  if (mode !== "career") return {};
  const existing = careerSite();
  return existing ? { siteId: existing.id } : {};
}

export function PortfolioBuilder(props: {
  models: ModelOption[];
  defaultModelId: string;
  githubEnabled: boolean;
  studentName: string;
  initialMode: "career" | "showcase" | null;
}) {
  const [draft, dispatch] = useReducer(reducer, props.studentName, (name) =>
    initialDraft(name, newSiteId()),
  );
  const [sites, setSites] = useState<SiteRecord[]>([]);
  const patch = (p: Partial<Draft>) => dispatch({ type: "patch", patch: p });

  useEffect(() => {
    // Async IIFE with setState only after an await (the house pattern; the
    // lint rule forbids synchronous setState in an effect body).
    void (async () => {
      // The v6.3.0 lift moves the draft too, so await it before reading the
      // site list: otherwise the migrated site shows with nothing to open.
      await migrateJobScoutPortfolio();
      setSites(loadSites());
      if (props.initialMode) {
        patch({
          mode: props.initialMode,
          step: props.initialMode === "career" ? "resume" : "course",
          ...pickMode(props.initialMode),
        });
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const steps =
    draft.mode === "career" ? CAREER_STEPS : draft.mode === "showcase" ? SHOWCASE_STEPS : ["mode" as Step];
  const index = steps.indexOf(draft.step);
  const go = (delta: number) =>
    patch({ step: steps[Math.min(steps.length - 1, Math.max(0, index + delta))] });
  const nav = { index, total: steps.length - 1, onBack: index > 0 ? () => go(-1) : null, onNext: () => go(1) };

  async function openSite(site: SiteRecord) {
    const stored = await getDraft(site.id);
    if (!stored) {
      patch({ siteId: site.id, mode: site.kind, step: site.kind === "career" ? "resume" : "course" });
      return;
    }
    patch({
      siteId: site.id, mode: site.kind, step: "review", content: stored.content, html: stored.html,
      resumeLink: stored.resumeLink,
      readme: stored.readme ?? null, skillIds: stored.skillIds ?? [],
      photo: stored.photoBase64 ? { base64: stored.photoBase64, bytes: 0 } : null,
      name: stored.student?.name ?? props.studentName, links: stored.student?.links ?? [],
      courses: stored.student?.courses ?? [], course: stored.showcaseMeta?.course ?? "",
      semester: stored.showcaseMeta?.semester ?? "", team: stored.showcaseMeta?.team ?? [],
      files: stored.files.filter((f) => f.projectSlug === null),
      projects: Array.from(
        new Set(stored.files.filter((f) => f.projectSlug !== null).map((f) => f.projectSlug as string)),
      ).map((slug) => ({
        slug, title: "", externalUrl: "",
        files: stored.files.filter((f) => f.projectSlug === slug),
      })),
    });
  }

  if (draft.step === "mode") {
    return (
      <ModeStep
        sites={sites}
        onPick={(mode) =>
          patch({ mode, step: mode === "career" ? "resume" : "course", ...pickMode(mode) })
        }
        onOpen={(site) => void openSite(site)}
        onRemove={(site) => setSites(removeSite(site.id))}
      />
    );
  }
  const common = { draft, patch, nav };
  switch (draft.step) {
    case "resume": return <ResumeStep {...common} />;
    case "classes": return <ClassesStep {...common} />;
    case "projects": return <ProjectsStep {...common} />;
    case "details":
      return <DetailsStep {...common} models={props.models} defaultModelId={props.defaultModelId} />;
    case "course": return <CourseStep {...common} />;
    case "files": return <FilesStep {...common} />;
    case "story":
      return <StoryStep {...common} models={props.models} defaultModelId={props.defaultModelId} />;
    case "review":
      return (
        <ReviewStep
          {...common}
          models={props.models}
          defaultModelId={props.defaultModelId}
          githubEnabled={props.githubEnabled}
          onPublished={() => setSites(loadSites())}
          onStartOver={() =>
            dispatch({ type: "reset", draft: initialDraft(props.studentName, newSiteId()) })
          }
        />
      );
  }
}
