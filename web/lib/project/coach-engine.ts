// lib/project/coach-engine.ts
import {
  applyGenericOp,
  buildEmptyContent,
  coachContentSchema,
  type CoachSpec,
  type GenericContent,
} from "@/lib/project/coach-framework";
import { getCoachSpec } from "@/lib/project/coach-specs";
import {
  applyScopingOp,
  emptyScopingContent,
  scopingContentSchema,
  type ScopingContent,
  type ScopingOp,
} from "@/lib/project/scoping";
import {
  SCOPING_COACH_PROMPT,
  serializeScopingForPrompt,
} from "@/lib/prompts/project-scoping";

export type GenericOp =
  | { kind: "setField"; path: string; value: string }
  | { kind: "addRow"; table: string }
  | { kind: "setRow"; table: string; index: number; row: Record<string, string> };

export interface CoachEngine {
  emptyContent(): unknown;
  /** Parse a stored contentJson, falling back to an empty deliverable. */
  parseContent(contentJson: string): unknown;
  /** Validate an untrusted value (a direct edit), or null if it is not valid. */
  parseUnknown(value: unknown): unknown | null;
  applyOp(content: unknown, op: GenericOp): unknown;
  serializeForPrompt(content: unknown): string;
  systemPrompt: string;
}

function safeJson(contentJson: string): unknown {
  try {
    return JSON.parse(contentJson || "{}");
  } catch {
    return {};
  }
}

function scopingEngine(): CoachEngine {
  return {
    emptyContent: () => emptyScopingContent(),
    parseContent: (json) => {
      const parsed = scopingContentSchema.safeParse(safeJson(json));
      return parsed.success ? parsed.data : emptyScopingContent();
    },
    parseUnknown: (value) => {
      const parsed = scopingContentSchema.safeParse(value);
      return parsed.success ? parsed.data : null;
    },
    applyOp: (content, op) => applyScopingOp(content as ScopingContent, op as ScopingOp),
    serializeForPrompt: (content) => serializeScopingForPrompt(content as ScopingContent),
    systemPrompt: SCOPING_COACH_PROMPT,
  };
}

function genericEngine(spec: CoachSpec): CoachEngine {
  const schema = coachContentSchema(spec);
  return {
    emptyContent: () => buildEmptyContent(spec),
    parseContent: (json) => {
      const parsed = schema.safeParse(safeJson(json));
      return parsed.success ? parsed.data : buildEmptyContent(spec);
    },
    parseUnknown: (value) => {
      const parsed = schema.safeParse(value);
      return parsed.success ? parsed.data : null;
    },
    applyOp: (content, op) => applyGenericOp(spec, content as GenericContent, op),
    serializeForPrompt: (content) => JSON.stringify(content),
    systemPrompt: spec.systemPrompt,
  };
}

export function getCoachEngine(coachType: string): CoachEngine | null {
  if (coachType === "scoping") return scopingEngine();
  const spec = getCoachSpec(coachType);
  return spec ? genericEngine(spec) : null;
}
