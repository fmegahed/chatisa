import { describe, expect, it } from "vitest";
import {
  describePlot,
  mergePlotInfo,
  parsePlotInfo,
  plotAltText,
  plotDetails,
  plotInfoFromCode,
} from "@/lib/run/plot-alt";

// The Coding Studio's own examples (components/sandbox/Sandbox.tsx EXAMPLES).
const PY = `import pandas as pd
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for _, r in grades.iterrows():
    ax.plot([r["ISA 401"], r["ISA 444"]], [r["student"], r["student"]], color="#54585A")
ax.scatter(grades["ISA 401"], grades["student"], label="ISA 401")
ax.set(xlabel="Grade", ylabel="Student", title="ISA grades by student and course")
ax.legend(title="Course")
plt.show()`;

const R = `library(tidyverse)
ggplot(grades, aes(x = grade, y = student)) +
  geom_line(aes(group = student), color = "gray30", linewidth = 1) +
  geom_point(aes(color = course), size = 5) +
  labs(title = "ISA grades by student and course", x = "Grade", y = "Student", color = "Course")`;

const SQL = `-- SQL runs in your browser. DRAW nothing here
SELECT * FROM wide
VISUALISE student AS y
DRAW segment MAPPING g401 AS x, g444 AS xend, student AS yend
DRAW point MAPPING g401 AS x, 'ISA 401' AS fill
LABEL x => 'Grade', y => 'Student', fill => 'Course'`;

describe("plot descriptions (#23)", () => {
  it("reads titles, axes and chart calls from Python code", () => {
    const info = plotInfoFromCode(PY, "python");
    expect(info.title).toBe("ISA grades by student and course");
    expect(info.xLabel).toBe("Grade");
    expect(info.yLabel).toBe("Student");
    expect(info.kinds).toEqual(["lines", "points"]);
  });

  it("does not mistake a legend title for the chart title", () => {
    const info = plotInfoFromCode(
      'ax.legend(title="Course")\nax.bar(x, y)\nax.set_xlabel("Year")',
      "python",
    );
    expect(info.title).toBeUndefined();
    expect(info.kinds).toEqual(["bars"]);
  });

  it("reads ggplot2 labs() and geoms from R code", () => {
    const info = plotInfoFromCode(R, "r");
    expect(info).toMatchObject({
      title: "ISA grades by student and course",
      xLabel: "Grade",
      yLabel: "Student",
      kinds: ["lines", "points"],
    });
  });

  it("reads base R plot arguments", () => {
    const info = plotInfoFromCode(
      'hist(mtcars$mpg, main = "Miles per gallon", xlab = "MPG")',
      "r",
    );
    expect(info).toMatchObject({
      kinds: ["histogram"],
      title: "Miles per gallon",
      xLabel: "MPG",
    });
  });

  it("reads ggsql DRAW and LABEL clauses, ignoring comments", () => {
    const info = plotInfoFromCode(SQL, "sql");
    expect(info.kinds).toEqual(["segments", "points"]);
    expect(info.xLabel).toBe("Grade");
    expect(info.yLabel).toBe("Student");
  });

  it("writes a specific alt text with position, type, title and axes", () => {
    const alt = plotAltText({
      language: "r",
      info: plotInfoFromCode(R, "r"),
      index: 0,
      total: 2,
    });
    expect(alt).toBe(
      'Plot 1 of 2: chart with lines and points titled "ISA grades by student and course", x axis Grade, y axis Student. From your R code.',
    );
  });

  it("falls back honestly when nothing is known", () => {
    expect(plotAltText({ language: "python", info: {}, index: 0, total: 2 })).toBe(
      "Plot 1 of 2 from your Python code (no title)",
    );
    expect(plotAltText({ language: "r" })).toBe("Plot from your R code (no title)");
  });

  it("prefers the figure's facts, but keeps a specific chart word from the code", () => {
    const merged = mergePlotInfo(
      parsePlotInfo({
        kinds: ["bars"],
        title: "Heights",
        xLabel: "",
        yLabel: "Count",
        series: ["a", "b"],
        panels: 1,
      }),
      { kinds: ["histogram"], xLabel: "cm", source: "code" },
    );
    expect(merged).toMatchObject({
      kinds: ["histogram"],
      title: "Heights",
      xLabel: "cm",
      yLabel: "Count",
      series: ["a", "b"],
      source: "figure",
    });
  });

  it("ignores malformed worker facts", () => {
    expect(parsePlotInfo(null)).toBeUndefined();
    expect(parsePlotInfo({ kinds: ["nonsense", 3], title: 5 })).toMatchObject({
      kinds: [],
      title: undefined,
    });
    // describePlot survives junk and still uses the code.
    expect(describePlot(R, "r", "junk").title).toBe(
      "ISA grades by student and course",
    );
  });

  it("lists details for the Describe disclosure", () => {
    const rows = plotDetails(plotInfoFromCode(R, "r"), "r");
    expect(rows.map((r) => r.label)).toEqual([
      "Chart type",
      "Title",
      "X axis",
      "Y axis",
      "Language",
      "Described from",
    ]);
    expect(rows[0].value).toBe("Chart with lines and points");
  });
});
