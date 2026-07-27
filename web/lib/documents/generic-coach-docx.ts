// lib/documents/generic-coach-docx.ts
import "server-only";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";
import {
  coverBlocks,
  docFromChildren,
  genericBlocks,
  type ScopingDocHeader,
} from "@/lib/documents/coach-docx";

export async function renderGenericCoachDocx(
  spec: CoachSpec,
  content: GenericContent,
  header: ScopingDocHeader,
): Promise<Buffer> {
  return docFromChildren([
    ...coverBlocks(header, `${spec.title}: ${header.projectName || "Project"}`, 32),
    ...genericBlocks(spec, content),
  ]);
}
