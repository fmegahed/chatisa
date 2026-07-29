import "server-only";
import { countScoutPostings, upsertScoutPosting } from "@/lib/db";
import { TAXONOMY_VERSION } from "./taxonomy";

/**
 * Deterministic feed for mock mode (CHATISA_MOCK_LLM=1), seeded at boot when
 * the table is empty so e2e tests have a feed without any network or model
 * call. Never runs outside mock mode: the real feed comes only from the
 * weekly harvest.
 */

const skills = (pairs: [string, "required" | "preferred"][]) =>
  JSON.stringify(pairs.map(([skillId, importance]) => ({ skillId, importance })));

const FIXTURES = [
  {
    externalId: "fix-001",
    title: "Data Analyst",
    company: "Queen City Insurance",
    locationCity: "Cincinnati",
    locationState: "OH",
    remote: false,
    category: "fulltime" as const,
    skillsJson: skills([
      ["sql", "required"],
      ["excel", "required"],
      ["data_visualization", "required"],
      ["tableau", "preferred"],
      ["communication", "preferred"],
    ]),
    description:
      "Queen City Insurance seeks a Data Analyst to build reporting for our personal lines business. Required: SQL, Excel, and clear data visualization for underwriting stakeholders. Preferred: Tableau. You will own weekly dashboards and ad hoc pricing analyses.",
  },
  {
    externalId: "fix-002",
    title: "Business Intelligence Analyst",
    company: "Riverbend Health",
    locationCity: "Columbus",
    locationState: "OH",
    remote: false,
    category: "fulltime" as const,
    visaSponsorship: "sponsors" as const,
    skillsJson: skills([
      ["business_intelligence", "required"],
      ["power_bi", "required"],
      ["sql", "required"],
      ["data_warehousing", "preferred"],
      ["dashboard_design", "preferred"],
    ]),
    description:
      "Riverbend Health is hiring a BI Analyst. Required qualifications: Power BI, SQL, and experience turning warehouse data into executive dashboards. Preferred: dimensional modeling and KPI design experience.",
  },
  {
    externalId: "fix-003",
    title: "Information Security Analyst",
    company: "Fort Washington Bank",
    locationCity: "Cincinnati",
    locationState: "OH",
    remote: false,
    category: "fulltime" as const,
    visaSponsorship: "no_sponsorship" as const,
    skillsJson: skills([
      ["cybersecurity", "required"],
      ["security_operations", "required"],
      ["risk_management", "preferred"],
      ["compliance", "preferred"],
      ["python", "preferred"],
    ]),
    description:
      "Fort Washington Bank seeks an Information Security Analyst for our SOC. Required: security monitoring, incident triage, SIEM familiarity. Preferred: risk frameworks, regulatory compliance exposure, and scripting in Python.",
  },
  {
    externalId: "fix-004",
    title: "Data Science Intern",
    company: "Lakefront Retail Group",
    locationCity: "Chicago",
    locationState: "IL",
    remote: false,
    category: "internship" as const,
    skillsJson: skills([
      ["python", "required"],
      ["machine_learning", "required"],
      ["statistical_analysis", "preferred"],
      ["data_visualization", "preferred"],
    ]),
    description:
      "Summer internship on the demand forecasting team. Required: Python and coursework in machine learning. Preferred: statistics, visualization, and curiosity about retail analytics.",
  },
  {
    externalId: "fix-005",
    title: "Management Analyst",
    company: "Department of the Treasury",
    locationCity: "Washington",
    locationState: "DC",
    remote: false,
    category: "federal" as const,
    skillsJson: skills([
      ["data_analysis", "required"],
      ["excel", "required"],
      ["communication", "required"],
      ["data_visualization", "preferred"],
    ]),
    description:
      "Serve as a Management Analyst supporting data-driven process improvement across bureau operations. Requires data analysis, spreadsheets, and strong written communication; dashboarding is desirable.",
  },
  {
    externalId: "fix-006",
    title: "Remote Analytics Consultant",
    company: "Bluegrass Advisory",
    locationCity: null,
    locationState: "KY",
    remote: true,
    category: "fulltime" as const,
    skillsJson: skills([
      ["consulting", "required"],
      ["data_analysis", "required"],
      ["stakeholder_management", "required"],
      ["r", "preferred"],
      ["forecasting", "preferred"],
    ]),
    description:
      "Client-facing analytics consulting, fully remote. Required: consulting mindset, end-to-end data analysis, stakeholder communication. Preferred: R and forecasting experience for our supply chain clients.",
  },
] as const;

export function seedScoutFixtures(): void {
  if (countScoutPostings() > 0) return;
  for (const f of FIXTURES) {
    upsertScoutPosting({
      source: "activejobs",
      externalId: f.externalId,
      fingerprint: `${f.company.toLowerCase()}|${f.title.toLowerCase()}|${f.locationState ?? ""}`,
      title: f.title,
      company: f.company,
      locationCity: f.locationCity,
      locationState: f.locationState,
      remote: f.remote,
      category: f.category,
      applyUrl: `https://careers.example.com/${f.externalId}`,
      description: f.description,
      postedAt: "2026-07-26",
      skillsJson: f.skillsJson,
      visaSponsorship:
        "visaSponsorship" in f ? (f.visaSponsorship as string) : "unknown",
      taxonomyVersion: TAXONOMY_VERSION,
    });
  }
}
