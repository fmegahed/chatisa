import "server-only";
import JSZip from "jszip";

/**
 * Makes a generated .pptx openable by PowerPoint.
 *
 * The model builds decks with python-pptx inside the hosted sandbox, and the
 * code it writes reliably leaves each slide with TWO slideLayout relationships
 * pointing at the same layout:
 *
 *   <Relationship Id="rId1" Type=".../slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
 *   <Relationship Id="rId2" Type=".../slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
 *
 * A slide part must have exactly one. PowerPoint refuses the whole package:
 * opening it through COM fails with E_FAIL, and interactively it offers to
 * repair the file instead of showing the deck. So the student who asked for a
 * deck receives a file they cannot open, while the deck's actual content is
 * fine.
 *
 * Measured 2026-07-25 on a real generated deck: five slides, five duplicate
 * relationships, unopenable. Removing exactly those five made PowerPoint open
 * it and export all five slides. Nothing else in the package was wrong.
 *
 * This runs on the download path rather than as a prompt instruction because it
 * is deterministic: whatever code the model writes next week, the bytes the
 * student receives are a valid package. The prompt also asks for one layout
 * relationship, which reduces how often this has to do anything.
 */

const SLIDE_LAYOUT_TYPE = "officeDocument/2006/relationships/slideLayout";
const SLIDE_RELS_PATH = /^ppt\/slides\/_rels\/slide\d+\.xml\.rels$/;
const RELATIONSHIP_TAG = /<Relationship\b[^>]*?\/>/g;

export interface PptxRepairResult {
  bytes: Uint8Array;
  /** Duplicate slideLayout relationships removed. Zero means untouched. */
  removed: number;
}

/**
 * Drops every slideLayout relationship after the first in one rels part.
 * Exported for the unit tests; `repairPptx` is the entry point.
 */
export function dedupeLayoutRelationships(xml: string): {
  xml: string;
  removed: number;
} {
  let seenLayout = false;
  let removed = 0;
  const out = xml.replace(RELATIONSHIP_TAG, (tag) => {
    if (!tag.includes(SLIDE_LAYOUT_TYPE)) return tag;
    if (seenLayout) {
      removed += 1;
      return "";
    }
    seenLayout = true;
    return tag;
  });
  return { xml: out, removed };
}

/**
 * Returns repaired bytes, or the input unchanged when there is nothing to fix
 * or the bytes are not a readable zip. Never throws: a download must not fail
 * because a repair could not be attempted.
 */
export async function repairPptx(input: Uint8Array): Promise<PptxRepairResult> {
  let zip: JSZip;
  try {
    zip = await JSZip.loadAsync(input);
  } catch {
    return { bytes: input, removed: 0 };
  }

  const targets = Object.keys(zip.files).filter((name) =>
    SLIDE_RELS_PATH.test(name),
  );
  if (targets.length === 0) return { bytes: input, removed: 0 };

  let removed = 0;
  for (const name of targets) {
    const file = zip.file(name);
    if (!file) continue;
    const original = await file.async("string");
    const fixed = dedupeLayoutRelationships(original);
    if (fixed.removed > 0) {
      zip.file(name, fixed.xml);
      removed += fixed.removed;
    }
  }

  // Re-zipping a package that was already valid would change the bytes for no
  // reason, so a clean deck is passed through exactly as the provider sent it.
  if (removed === 0) return { bytes: input, removed: 0 };

  try {
    const bytes = await zip.generateAsync({
      type: "uint8array",
      compression: "DEFLATE",
    });
    return { bytes, removed };
  } catch {
    return { bytes: input, removed: 0 };
  }
}
