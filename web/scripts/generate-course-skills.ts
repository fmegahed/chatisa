/**
 * Regenerates the course-to-skill mapping when the curriculum changes.
 * Dev-machine only (run: `npm run scout:course-skills`); never shipped in
 * the deploy bundle and never run by the server.
 *
 * Two frontier models map each course description against the taxonomy
 * independently (design 2026-07-28 §2.2). Agreements auto-accept;
 * disagreements are flagged in a review markdown for the instructor. The
 * output is NOT written over lib/scout/course-skills.ts automatically: the
 * instructor reviews and applies, because a wrong anchor becomes a wrong
 * resume line.
 *
 * Needs ANTHROPIC_API_KEY and OPENAI_API_KEY in .env.local. Optional:
 * --courses "ISA 444,ISA 336" to regenerate a subset.
 */

import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { z } from "zod";
import { generateObject } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createOpenAI } from "@ai-sdk/openai";
import { COURSES } from "../lib/scout/courses";
import { SKILL_IDS } from "../lib/scout/taxonomy";

// .env.local, hand-parsed: this script runs outside Next's env loading.
try {
  const envFile = readFileSync(path.join(process.cwd(), ".env.local"), "utf8");
  for (const line of envFile.split("\n")) {
    const m = /^([A-Z_][A-Z0-9_]*)=(.*)$/.exec(line.trim());
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
} catch {
  // Fine: keys may already be in the environment.
}

const MODELS = [
  {
    name: "claude-sonnet-5",
    model: () =>
      createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY })(
        "claude-sonnet-5",
      ),
  },
  {
    name: "gpt-5.6-terra",
    model: () =>
      createOpenAI({ apiKey: process.env.OPENAI_API_KEY })("gpt-5.6-terra"),
  },
];

const linkSchema = z.object({
  links: z
    .array(
      z.object({
        skillId: z.enum(SKILL_IDS as [string, ...string[]]),
        level: z.enum(["anchor", "applied", "exposure"]),
        evidence: z.string().max(160),
      }),
    )
    .min(1)
    .max(10),
});

const INSTRUCTIONS = `You map one university course to the skills it builds, for job matching.

Use ONLY ids from the vocabulary. Levels: "anchor" = the course is substantially about it and graded deliverables demonstrate it (1-3 per course); "applied" = used repeatedly as a working tool; "exposure" = introduced. Write evidence as a short student-voice phrase ("built and evaluated forecasting models on business data") drawn from the description. Do not claim a specific software tool unless the description names it. Fewer, defensible links beat generous ones: an anchor here becomes a resume claim.`;

async function main() {
  const onlyArg = process.argv.indexOf("--courses");
  const only =
    onlyArg >= 0
      ? new Set(process.argv[onlyArg + 1].split(",").map((s) => s.trim()))
      : null;
  const courses = COURSES.filter(
    (c) => !c.special && (!only || only.has(c.code)),
  );
  if (!process.env.ANTHROPIC_API_KEY || !process.env.OPENAI_API_KEY) {
    console.error(
      "Set ANTHROPIC_API_KEY and OPENAI_API_KEY (in the environment or .env.local).",
    );
    process.exit(1);
  }

  const lines: string[] = [
    `# Course-skill regeneration (${new Date().toISOString().slice(0, 10)})`,
    "",
    "AGREE = both models emitted the skill (levels may differ; the stricter",
    "level is suggested). ONLY-<model> rows need instructor judgment.",
    "",
  ];

  for (const course of courses) {
    process.stdout.write(`${course.code}... `);
    const prompt = `Course: ${course.code} ${course.title} (${course.credits} credits)\n\n${course.description}\n\nVocabulary ids:\n${SKILL_IDS.join(", ")}`;
    const results = await Promise.all(
      MODELS.map(async ({ name, model }) => {
        try {
          const { object } = await generateObject({
            model: model(),
            schema: linkSchema,
            instructions: INSTRUCTIONS,
            prompt,
            maxOutputTokens: 1_500,
          });
          return { name, links: object.links };
        } catch (err) {
          return { name, links: [], error: String(err) };
        }
      }),
    );
    const [a, b] = results;
    lines.push(`## ${course.code} — ${course.title}`, "");
    const bIds = new Map(b.links.map((l) => [l.skillId, l]));
    const levels = ["anchor", "applied", "exposure"];
    for (const link of a.links) {
      const other = bIds.get(link.skillId);
      if (other) {
        const stricter =
          levels[Math.max(levels.indexOf(link.level), levels.indexOf(other.level))];
        lines.push(
          `- AGREE ${link.skillId} (${link.level} vs ${other.level} -> ${stricter}): ${link.evidence}`,
        );
        bIds.delete(link.skillId);
      } else {
        lines.push(`- ONLY-${a.name} ${link.skillId} (${link.level}): ${link.evidence}`);
      }
    }
    for (const link of bIds.values()) {
      lines.push(`- ONLY-${b.name} ${link.skillId} (${link.level}): ${link.evidence}`);
    }
    for (const r of results) {
      if ("error" in r && r.error) lines.push(`- MODEL-ERROR ${r.name}: ${r.error}`);
    }
    lines.push("");
    console.log("done");
  }

  const out = path.join(
    process.cwd(),
    "..",
    "docs",
    "development",
    `${new Date().toISOString().slice(0, 10)}-course-skills-regen.md`,
  );
  writeFileSync(out, lines.join("\n"));
  console.log(`\nReview file written: ${out}`);
  console.log(
    "Apply accepted rows by editing lib/scout/course-skills.ts; the integrity tests keep it honest.",
  );
}

void main();
