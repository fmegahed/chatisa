/**
 * Working chart exemplars in the house style (design 2026-07-25), one per
 * language. These are the code the model starts from, so they are kept short
 * enough to read in full and are verified to RUN in the browser runtimes rather
 * than merely to look plausible.
 *
 * Both build the same figure: the ISA 401 vs ISA 444 dumbbell already used as
 * the Coding Studio starter, which exercises every rule at once. Two series, so
 * the red-on-charcoal highlight pair applies; few enough points to label, so the
 * annotations must not collide; and a title that states the finding.
 */

export const R_CHART_EXAMPLE = `# Miami house style, R. Two series, so red = focus and charcoal = context.
library(ggplot2)
library(dplyr)
library(ggtext)    # coloured words in the subtitle, replaces the legend
library(ggrepel)   # labels that do not sit on the geoms or each other

grades <- tibble::tibble(
  student = c("Amanda", "Bill", "Cara", "Dan"),
  isa_401 = c(93, 88, 74, 85),
  isa_444 = c(86, 81, 82, 78)
) |>
  mutate(student = reorder(student, isa_401))

long <- tidyr::pivot_longer(grades, c(isa_401, isa_444),
                            names_to = "course", values_to = "grade")

ggplot(long, aes(grade, student)) +
  geom_line(aes(group = student), colour = "#585E60", linewidth = 1.2) +
  geom_point(aes(colour = course, shape = course), size = 5) +
  # Labels are the secondary encoding: never rely on colour alone.
  geom_text_repel(aes(label = grade), size = 3.4, colour = "#000000",
                  min.segment.length = 0, box.padding = 0.6,
                  segment.colour = "#585E60", seed = 1) +
  scale_colour_manual(values = c(isa_401 = "#C3142D", isa_444 = "#585E60")) +
  scale_shape_manual(values = c(isa_401 = 16, isa_444 = 17)) +
  scale_x_continuous(limits = c(65, 100), expand = expansion(mult = 0.04)) +
  labs(
    # The title is the finding, not the variables.
    title = "ISA 401 grades ran higher for three of four students",
    subtitle = "Grade in <span style='color:#C3142D'>**ISA 401**</span> versus <span style='color:#585E60'>**ISA 444**</span>. Cara is the exception, up 8 points in ISA 444.",
    x = "Grade", y = "Student"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14, colour = "#000000"),
    plot.subtitle = element_markdown(colour = "#000000", lineheight = 1.3),
    plot.title.position = "plot",
    legend.position = "none",          # identity lives in the subtitle
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "#FFFFFF", colour = NA),
    plot.background = element_rect(fill = "#FFFFFF", colour = NA),
    axis.text = element_text(colour = "#000000")
  )
`;

export const PYTHON_CHART_EXAMPLE = `# Miami house style, matplotlib. Two series: red = focus, charcoal = context.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text   # labels that do not collide

RED, CHARCOAL, INK, BG = "#C3142D", "#585E60", "#000000", "#FFFFFF"
# Ask for a condensed sans and let it fall back; never hardcode one family.
plt.rcParams["font.family"] = ["Roboto Condensed", "DejaVu Sans", "sans-serif"]

students = ["Cara", "Dan", "Bill", "Amanda"]      # sorted by ISA 401
isa_401  = [74, 85, 88, 93]
isa_444  = [82, 78, 81, 86]

fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=BG)
ax.set_facecolor(BG)

texts = []
for i, (a, b) in enumerate(zip(isa_401, isa_444)):
    ax.plot([a, b], [i, i], color=CHARCOAL, linewidth=2, zorder=1)
    # Distinct markers as well as colour: the secondary encoding.
    ax.scatter(a, i, s=120, color=RED, marker="o", zorder=2)
    ax.scatter(b, i, s=120, color=CHARCOAL, marker="^", zorder=2)
    texts += [ax.text(a, i, str(a), color=INK, fontsize=9),
              ax.text(b, i, str(b), color=INK, fontsize=9)]

ax.set_yticks(range(len(students)), students)
ax.set_xlim(65, 100)
ax.set_xlabel("Grade", color=INK)
ax.set_ylabel("Student", color=INK)
# Title states the finding; subtitle carries the insight.
ax.set_title("ISA 401 grades ran higher for three of four students",
             fontsize=14, fontweight="bold", color=INK, loc="left", pad=26)
ax.text(0, 1.03, "Circles are ISA 401, triangles ISA 444. Cara is the exception, up 8 points in ISA 444.",
        transform=ax.transAxes, fontsize=10, color=INK, va="bottom")

ax.grid(axis="x", color="#E6E6E6", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(INK)
ax.tick_params(colors=INK)

adjust_text(texts, ax=ax, only_move={"text": "y"},
            arrowprops=dict(arrowstyle="-", color=CHARCOAL, lw=0.8))
fig.tight_layout()
plt.show()
`;

/**
 * The deck variant: the same rules with nothing that needs installing, for the
 * provider sandboxes. Kept separate from PYTHON_CHART_EXAMPLE so the model is
 * never tempted to carry an adjustText import into a container without network.
 */
export const HOSTED_CHART_EXAMPLE = `# Miami house style for a HOSTED sandbox: matplotlib only, no extra packages.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, CHARCOAL, INK, BG = "#C3142D", "#585E60", "#000000", "#FFFFFF"
plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

stages = ["Descriptive", "Diagnostic", "Predictive", "Prescriptive"]
share  = [42, 27, 21, 10]

fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=BG)
ax.set_facecolor(BG)
# One series, so Miami red alone, and no legend: the title names the series.
bars = ax.bar(stages, share, color=RED, width=0.62)
for bar, value in zip(bars, share):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value}%",
            ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")

ax.set_ylim(0, max(share) * 1.18)
ax.set_ylabel("Share of reported use (%)", color=INK)
ax.set_title("Most reported analytics work is still descriptive",
             fontsize=15, fontweight="bold", color=INK, loc="left", pad=26)
ax.text(0, 1.03, "Prescriptive work is the smallest slice at 10%, the gap this course targets.",
        transform=ax.transAxes, fontsize=10.5, color=INK, va="bottom")

ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(colors=INK)
fig.tight_layout()
fig.savefig("analytics_stages.png", dpi=200, facecolor=BG)
`;
