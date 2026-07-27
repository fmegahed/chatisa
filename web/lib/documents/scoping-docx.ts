// lib/documents/scoping-docx.ts
import "server-only";
import type { ScopingContent } from "@/lib/project/scoping";
import {
  coverBlocks,
  docFromChildren,
  scopingBlocks,
  type ScopingDocHeader,
} from "@/lib/documents/coach-docx";

export type { ScopingDocHeader };

export async function renderScopingDocx(
  content: ScopingContent,
  header: ScopingDocHeader,
): Promise<Buffer> {
  return docFromChildren([
    ...coverBlocks(header, header.projectName || "Project scope"),
    ...scopingBlocks(content),
  ]);
}
