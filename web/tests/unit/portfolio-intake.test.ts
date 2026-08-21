import { describe, expect, it } from "vitest";
import { prepareFile, pushable, toRoutePayloadFile } from "@/lib/portfolio/intake";

describe("prepareFile", () => {
  it("reads code as text and publishes it", async () => {
    const f = new File(["x <- 1"], "model.R", { type: "text/plain" });
    const p = await prepareFile(f, "code");
    expect(p).toMatchObject({ name: "model.R", role: "code", publish: true, text: "x <- 1", base64: null });
    expect(toRoutePayloadFile(p)).toEqual({ kind: "text", name: "model.R", content: "x <- 1" });
    expect(pushable(p)).toBe(true);
  });
  it("keeps data files unpublished by default", async () => {
    const p = await prepareFile(new File(["a,b"], "train.csv"), "data");
    expect(p.publish).toBe(false);
  });
  it("stores images as base64 only", async () => {
    const p = await prepareFile(new File([new Uint8Array([137, 80, 78, 71])], "roc.png", { type: "image/png" }), "figure");
    expect(p.text).toBeNull();
    expect(p.base64).toBe("iVBORw==");
    expect(toRoutePayloadFile(p)).toEqual({ kind: "binary", name: "roc.png", sizeBytes: 4 });
  });
  it("strips notebook outputs to cell text", async () => {
    const nb = JSON.stringify({ cells: [{ cell_type: "code", source: ["print(1)"], outputs: [] }], metadata: {}, nbformat: 4, nbformat_minor: 5 });
    const p = await prepareFile(new File([nb], "Final Project.ipynb"), "notebook");
    expect(p.text).toContain("print(1)");
    expect(p.base64).not.toBeNull();
  });
  it("keeps an oversize notebook's text for the model but never pushes it", async () => {
    // A notebook over the 400 KB push cap still yields stripped cell text for
    // the prompt, but its bytes are not held, so it must not be published as
    // stripped text under a .ipynb name.
    const marker = "oversize notebook cell";
    const cell = (source: string) => JSON.stringify({
      cells: [{ cell_type: "markdown", source: [source] }],
      metadata: {}, nbformat: 4, nbformat_minor: 5,
    });
    const raw = cell(marker + "x".repeat(400_001 - cell(marker).length));
    expect(raw.length).toBe(400_001);
    const p = await prepareFile(new File([raw], "Big Notebook.ipynb"), "notebook");
    expect(p.text).toContain(marker);
    expect(p.base64).toBe("");
    expect(p.publish).toBe(false);
    expect(pushable(p)).toBe(false);
  });
  it("marks oversize binaries as not pushable", async () => {
    const big = new File([new Uint8Array(400_001)], "huge.bin");
    const p = await prepareFile(big, "other");
    expect(p.publish).toBe(false);
    expect(pushable(p)).toBe(false);
  });
});
