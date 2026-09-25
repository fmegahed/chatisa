import { describe, expect, it } from "vitest";
import { splitOversize } from "@/lib/portfolio/intake";
import { formatSize } from "@/lib/portfolio/files";

/**
 * One rule for every way a file enters the Portfolio Builder (professor,
 * 2026-09-24): a file over 25 MB, the publishing limit for one file, is not
 * added, and the student is told why.
 */
const file = (name: string, bytes: number) => ({ name, size: bytes }) as File;

describe("splitOversize", () => {
  it("keeps files up to 25 MB and names the ones it refuses", () => {
    const ok = file("model.py", 1_000);
    const edge = file("edge.csv", 25 * 1024 * 1024);
    const big = file("raw.csv", 30 * 1024 * 1024);
    const out = splitOversize([ok, edge, big]);
    expect(out.accepted).toEqual([ok, edge]);
    expect(out.refused).toBe("raw.csv (30.0 MB) is over the 25 MB limit for one file on a published page, so it was not added.");
  });
  it("names several refused files in one message, and says nothing when all fit", () => {
    expect(splitOversize([file("a.bin", 40 * 1024 * 1024), file("b.bin", 26 * 1024 * 1024)]).refused).toBe(
      "a.bin (40.0 MB) and b.bin (26.0 MB) are over the 25 MB limit for one file on a published page, so they were not added.",
    );
    expect(splitOversize([file("a.py", 1)]).refused).toBeNull();
  });
});

describe("formatSize (review fix: one unit everywhere)", () => {
  it("shows binary megabytes with one decimal, so a file under the limit never reads as over it", () => {
    expect(formatSize(25 * 1024 * 1024)).toBe("25.0 MB");
    expect(formatSize(26_000_000)).toBe("24.8 MB");
    expect(formatSize(25.3 * 1024 * 1024)).toBe("25.3 MB");
    expect(formatSize(1_500)).toBe("2 KB");
    expect(formatSize(10)).toBe("1 KB");
  });
});
