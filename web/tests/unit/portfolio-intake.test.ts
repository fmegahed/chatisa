import { describe, expect, it } from "vitest";
import { prepareFile, pushable, toRoutePayloadFile } from "@/lib/portfolio/intake";
import { PUSH_LIMITS } from "@/lib/scout/github";

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
  it("keeps a mid-size notebook's text for the model and its bytes for the push", async () => {
    // A 400 KB+ notebook used to lose its bytes (old 400 KB push cap). Now the
    // stripped cells feed the prompt and the original notebook is pushed.
    const marker = "mid-size notebook cell";
    const cell = (source: string) => JSON.stringify({
      cells: [{ cell_type: "markdown", source: [source] }],
      metadata: {}, nbformat: 4, nbformat_minor: 5,
    });
    const raw = cell(marker + "x".repeat(400_001 - cell(marker).length));
    const p = await prepareFile(new File([raw], "Big Notebook.ipynb"), "notebook");
    expect(p.text).toContain(marker);
    expect(p.base64).not.toBe("");
    expect(pushable(p)).toBe(true);
  });
  it("holds a large text file as bytes only, never as prompt text", async () => {
    const p = await prepareFile(new File(["x".repeat(400_001)], "big.csv"), "data");
    expect(p.text).toBeNull();
    expect(pushable(p)).toBe(true);
    expect(toRoutePayloadFile(p).kind).toBe("binary");
  });
  it("marks oversize binaries as not pushable", async () => {
    const big = new File([new Uint8Array(PUSH_LIMITS.fileBytes + 1)], "huge.bin");
    const p = await prepareFile(big, "other");
    expect(p.publish).toBe(false);
    expect(pushable(p)).toBe(false);
  });
});
