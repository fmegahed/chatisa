/**
 * Work in progress for the Portfolio Builder wizard (2026-08-23). The wizard
 * once held every upload in React state only, so a reload (which an error
 * message actually recommended) wiped a student's files and photo. The draft
 * is now written to IndexedDB as the student moves through the steps and
 * offered back on the next visit. One record per browser: the wizard builds
 * one site at a time.
 *
 * The resume is a File, which JSON cannot carry, so it travels as base64 and
 * is rebuilt on load. Nothing here goes to the server.
 */
import { getItem, putItem, removeItem } from "@/lib/scout/device-files";
import type { Draft } from "./draft";
import { base64ToBytes, fileToBase64 } from "./intake";

const WIP_KEY = "pb-wip";

type StoredWip = Omit<Draft, "resume"> & {
  resumeFile: { name: string; type: string; base64: string } | null;
  savedAt: string;
};

export type Wip = Draft & { savedAt: string };

/** Saves the draft; a draft still on the mode step is nothing to keep. */
export async function saveWip(draft: Draft): Promise<boolean> {
  if (draft.step === "mode" || draft.mode === null) return true;
  const { resume, ...rest } = draft;
  const record: StoredWip = {
    ...rest,
    resumeFile: resume ? { name: resume.name, type: resume.type, base64: await fileToBase64(resume) } : null,
    savedAt: new Date().toISOString(),
  };
  return putItem(WIP_KEY, record);
}

export async function loadWip(): Promise<Wip | null> {
  const stored = await getItem<StoredWip>(WIP_KEY);
  if (!stored || stored.step === "mode" || !stored.mode) return null;
  const { resumeFile, ...rest } = stored;
  const resume = resumeFile
    ? new File([base64ToBytes(resumeFile.base64)], resumeFile.name, { type: resumeFile.type })
    : null;
  return { ...rest, resume };
}

export function clearWip(): Promise<void> {
  return removeItem(WIP_KEY);
}
