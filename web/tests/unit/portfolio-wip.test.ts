import "fake-indexeddb/auto";
import { describe, expect, it } from "vitest";
import { initialDraft } from "@/lib/portfolio/draft";
import { clearWip, loadWip, saveWip } from "@/lib/portfolio/wip";

/**
 * The wizard used to hold every upload in React state only, so the "reload
 * and try again" advice in an error wiped the student's work. The
 * work-in-progress record survives a reload, resume PDF included.
 */
describe("portfolio work in progress", () => {
  it("round-trips a draft including the resume file", async () => {
    const draft = initialDraft("Ada", "site-1");
    draft.mode = "career";
    draft.step = "projects";
    draft.resume = new File([new Uint8Array([37, 80, 68, 70])], "resume.pdf", { type: "application/pdf" });
    draft.projects = [{ slug: "churn", title: "Churn", externalUrl: "", files: [
      { name: "model.R", role: "code", publish: true, bytes: 7, text: "lm(y~x)", base64: null },
    ] }];
    expect(await saveWip(draft)).toBe(true);
    const back = await loadWip();
    expect(back).not.toBeNull();
    expect(back!.step).toBe("projects");
    expect(back!.projects[0].files[0].text).toBe("lm(y~x)");
    expect(back!.resume?.name).toBe("resume.pdf");
    expect(new Uint8Array(await back!.resume!.arrayBuffer())).toEqual(new Uint8Array([37, 80, 68, 70]));
    expect(back!.savedAt).toMatch(/^\d{4}-/);
  });

  it("does not keep a draft that has not left the mode step, and clears on request", async () => {
    await clearWip();
    expect(await saveWip(initialDraft("Ada", "site-2"))).toBe(true);
    expect(await loadWip()).toBeNull();
    const d = initialDraft("Ada", "site-3");
    d.mode = "showcase";
    d.step = "files";
    await saveWip(d);
    expect((await loadWip())?.siteId).toBe("site-3");
    await clearWip();
    expect(await loadWip()).toBeNull();
  });
});
