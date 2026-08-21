import { describe, expect, it } from "vitest";
import { fitWithin, dataUrlToBase64 } from "@/lib/portfolio/image";

describe("fitWithin", () => {
  it("scales the long side down to maxSide and keeps aspect", () => {
    expect(fitWithin(1024, 768, 512)).toEqual({ width: 512, height: 384 });
    expect(fitWithin(300, 900, 512)).toEqual({ width: 171, height: 512 });
    expect(fitWithin(200, 100, 512)).toEqual({ width: 200, height: 100 });
  });
});

describe("dataUrlToBase64", () => {
  it("strips the data URL prefix", () => {
    expect(dataUrlToBase64("data:image/jpeg;base64,AAAA")).toBe("AAAA");
  });
});
