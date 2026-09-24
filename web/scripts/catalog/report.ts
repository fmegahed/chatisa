/**
 * The catalog refresh report (v6.7.0): the pull request's description, so
 * a reviewer sees in plain words what changed and what needs a decision.
 * Pure rendering plus a small CLI that reads the collector's and mapper's
 * outputs and writes catalog/REPORT.md.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import type { MergeReport } from "./merge";

export interface CollectReport {
  seeding: boolean;
  report: MergeReport;
  unparsed: { program: string; row: string }[];
  missingFromBulletin: string[];
  counts: { programs: number; courses: number; withSyllabus: number };
}

export interface MapReport {
  mapped: string[];
  agreed: number;
  disputed: { course: string; skillId: string; a: string | null; b: string | null; approved?: string }[];
  keptUnproposed?: { course: string; skillId: string }[];
  autoLowered?: { course: string; skillId: string; a: string; b: string }[];
  singleModelDropped?: { course: string; skillId: string; level: string }[];
  noAnchor?: string[];
  skippedByCap: string[];
  deferred?: string[];
  failed?: { course: string; error: string }[];
  written?: { course: string; skillId: string; level: string; evidence?: string }[];
  costUsd: number;
}

const list = (items: string[]) => items.map((i) => `- ${i}`).join("\n");

/**
 * `openFromEarlier`: courses mapping-state still marks "disputed" from any
 * run, so a decision nobody made yet is listed again rather than dropping
 * out of view once its fingerprint is recorded.
 */
export function renderReport(c: CollectReport, m: MapReport | null, openFromEarlier: string[] = []): string {
  const r = c.report;
  const out: string[] = ["# Catalog refresh", ""];
  out.push(`${c.counts.programs} programs, ${c.counts.courses} courses, ${c.counts.withSyllabus} with syllabus evidence.`, "");

  const changed = r.added.length + r.retired.length + r.retitled.length + r.suspectedRenumbering.length + r.prereqChanged.length;
  if (changed === 0) out.push("No changes to courses or programs.", "");
  if (r.added.length) out.push("## Added", list(r.added), "");
  if (r.retired.length) out.push("## No longer in the bulletin (kept, marked retired)", list(r.retired), "");
  if (r.retitled.length) out.push("## Renamed", list(r.retitled.map((t) => `${t.code}: ${t.from} → ${t.to}`)), "");
  if (r.suspectedRenumbering.length) {
    out.push(
      "## Suspected renumbering (confirm before linking)",
      list(r.suspectedRenumbering.map((s) => `${s.from} → ${s.to} (${s.title})`)),
      "",
    );
  }
  if (r.prereqChanged.length) out.push("## Prerequisites changed", list(r.prereqChanged), "");

  if (m) {
    out.push("## Skill links", `${m.mapped.length} courses mapped, ${m.agreed} links agreed by both models and added.`, "");
    // After a merge these count as approved and no later run changes them,
    // so every new anchor is shown with the evidence a student's profile
    // will quote.
    const anchors = (m.written ?? []).filter((w) => w.level === "anchor");
    const others = (m.written ?? []).filter((w) => w.level !== "anchor");
    if (anchors.length) {
      out.push(
        "### New anchor links (check the evidence: these become approved when merged)",
        list(anchors.map((w) => `${w.course} · ${w.skillId} (anchor)${w.evidence ? `: "${w.evidence}"` : ""}`)),
        "",
      );
    }
    if (others.length) {
      out.push(
        `<details><summary>New applied and exposure links (${others.length})</summary>`,
        "",
        list(others.map((w) => `${w.course} · ${w.skillId} (${w.level})`)),
        "",
        "</details>",
        "",
      );
    }
    if (m.failed?.length) {
      out.push("### Not mapped this run (a model call failed; the next run retries)", list(m.failed.map((f) => `${f.course}: ${f.error}`)), "");
    }
    if (m.noAnchor?.length) {
      out.push("### Courses to map by hand (nothing both models agreed gives them an anchor)", list(m.noAnchor), "");
    }
    if (m.disputed.length) {
      out.push(
        "### Needs your decision",
        list(m.disputed.map((d) => d.approved
          ? `${d.course} · ${d.skillId}: approved as ${d.approved}; the models say ${d.a ?? "none"} and ${d.b ?? "none"}`
          : `${d.course} · ${d.skillId}: ${d.a ?? "none"} or ${d.b ?? "none"}`)),
        "",
      );
    }
    if (m.keptUnproposed?.length) {
      out.push("### Approved links neither model proposed (kept; remove any that no longer fit)", list(m.keptUnproposed.map((k) => `${k.course} · ${k.skillId}`)), "");
    }
    if (m.autoLowered?.length || m.singleModelDropped?.length) {
      out.push(
        "<details><summary>Resolved automatically (conservative): " +
          `${m.autoLowered?.length ?? 0} set to the lower of two levels, ${m.singleModelDropped?.length ?? 0} single-model suggestions not added</summary>`,
        "",
        list((m.autoLowered ?? []).map((x) => `${x.course} · ${x.skillId}: ${x.a} / ${x.b} → lower`)),
        list((m.singleModelDropped ?? []).map((x) => `${x.course} · ${x.skillId} (${x.level}, one model only)`)),
        "",
        "</details>",
        "",
      );
    }
    if (m.skippedByCap.length) out.push("### Not mapped this run (spend cap reached)", list(m.skippedByCap), "");
    if (m.deferred?.length) out.push("### Not mapped yet (course limit for this run; the next run continues)", list(m.deferred), "");
    out.push(`Model cost this run: $${m.costUsd.toFixed(2)}.`, "");
  }
  const mappedNow = new Set(m?.mapped ?? []);
  const leftOver = openFromEarlier.filter((code) => !mappedNow.has(code));
  if (leftOver.length) {
    out.push("### Still waiting for a decision from earlier runs", list(leftOver), "");
  }

  if (c.unparsed.length || c.missingFromBulletin.length) {
    out.push("## Not understood (check the parser)");
    if (c.unparsed.length) out.push(list(c.unparsed.map((u) => `${u.program}: ${u.row}`)));
    if (c.missingFromBulletin.length) out.push(list(c.missingFromBulletin.map((x) => `${x}: listed in a program, not found in the course pages`)));
    out.push("");
  }
  return out.join("\n");
}

if (process.argv[1] && process.argv[1].endsWith("report.ts")) {
  const dir = path.join(process.cwd(), "catalog");
  const read = <T>(f: string): T | null => (existsSync(path.join(dir, f)) ? JSON.parse(readFileSync(path.join(dir, f), "utf8")) : null);
  const collect = read<CollectReport>(".collect-report.json");
  if (!collect) throw new Error("Run catalog:collect first.");
  const state = read<Record<string, { status: string }>>("mapping-state.json") ?? {};
  const open = Object.entries(state).filter(([, s]) => s.status === "disputed").map(([code]) => code).sort();
  writeFileSync(path.join(dir, "REPORT.md"), renderReport(collect, read<MapReport>(".map-report.json"), open));
  console.log("wrote catalog/REPORT.md");
}
