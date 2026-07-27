// tests/unit/project-db.test.ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-project-db-"));
process.env.CHATISA_DATA_DIR = dataDir;

const { closeDb, upsertUser } = await import("@/lib/db");
const {
  createProject,
  deleteProject,
  getAccessibleProject,
  isProjectMember,
  listOwnedProjects,
  listSharedProjects,
  listProjectMembers,
  addProjectMember,
  isProjectLead,
  removeProjectMember,
  updateProjectCoaches,
  fillMemberName,
  getDeliverable,
  getOrCreateDeliverable,
  saveDeliverableContent,
  saveDeliverableTranscript,
  listDeliverables,
} = await import("@/lib/db/projects");

const LEAD = "lead@miamioh.edu";
const MATE = "mate@miamioh.edu";
const STRANGER = "stranger@miamioh.edu";

beforeAll(() => {
  upsertUser(LEAD, "Team Lead");
  upsertUser(MATE, "Teammate");
  upsertUser(STRANGER, "Nobody");
  // The projects.owner_email foreign key requires the owner to exist. The
  // case-insensitivity check below creates a project for a mixed-case owner,
  // so seed that user too (emails are stored lowercased).
  upsertUser("Mixed.Case@Miamioh.edu", "Mixed");
});

afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

describe("project data layer", () => {
  it("creates a project and enrolls the owner as the sole lead", () => {
    const id = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "401/501",
      name: "Retail dashboard",
      organization: "Kroger",
      coachTypes: ["scoping", "premortem"],
    });
    const project = getAccessibleProject(id, LEAD);
    expect(project?.name).toBe("Retail dashboard");
    expect(project?.courseCode).toBe("401/501");
    expect(project?.coachTypes).toEqual(["scoping", "premortem"]);

    const members = listProjectMembers(id);
    expect(members).toHaveLength(1);
    expect(members[0].email).toBe(LEAD);
    expect(members[0].role).toBe("lead");
  });

  it("hides a project from a non-member", () => {
    const id = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "444/544",
      name: "Demand forecast",
      organization: "",
      coachTypes: ["scoping"],
    });
    expect(getAccessibleProject(id, STRANGER)).toBeUndefined();
    expect(isProjectMember(id, STRANGER)).toBe(false);
  });

  it("shares a project with an invited member and lists it for them", () => {
    const id = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "496",
      name: "Practicum project",
      organization: "Acme",
      coachTypes: ["scoping", "reflection"],
    });
    addProjectMember({ projectId: id, email: MATE, name: "Teammate", role: "member" });

    expect(isProjectMember(id, MATE)).toBe(true);
    expect(getAccessibleProject(id, MATE)?.name).toBe("Practicum project");

    // Owner sees it under "owned", member sees it under "shared".
    expect(listOwnedProjects(LEAD).some((p) => p.id === id)).toBe(true);
    expect(listOwnedProjects(MATE).some((p) => p.id === id)).toBe(false);
    expect(listSharedProjects(MATE).some((p) => p.id === id)).toBe(true);
    expect(listSharedProjects(LEAD).some((p) => p.id === id)).toBe(false);
  });

  it("treats emails case-insensitively", () => {
    const id = createProject({
      ownerEmail: "Mixed.Case@Miamioh.edu",
      ownerName: "Mixed",
      courseCode: "225",
      name: "Casing",
      organization: "",
      coachTypes: [],
    });
    expect(getAccessibleProject(id, "mixed.case@miamioh.edu")?.name).toBe("Casing");
  });
});

describe("deliverable data layer", () => {
  it("creates a deliverable once and returns the same row thereafter", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "401/501",
      name: "Deliverable project",
      organization: "",
      coachTypes: ["scoping"],
    });

    const first = getOrCreateDeliverable(projectId, "scoping");
    const second = getOrCreateDeliverable(projectId, "scoping");
    expect(second.id).toBe(first.id);
    expect(first.contentJson).toBe("{}");
    expect(first.transcriptJson).toBe("[]");
  });

  it("saves content and transcript with a last-updated-by name", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "444/544",
      name: "Save project",
      organization: "",
      coachTypes: ["scoping"],
    });
    getOrCreateDeliverable(projectId, "scoping");

    saveDeliverableContent({
      projectId,
      coachType: "scoping",
      contentJson: JSON.stringify({ organizationName: "Kroger" }),
      updatedBy: "Team Lead",
    });
    saveDeliverableTranscript({
      projectId,
      coachType: "scoping",
      transcriptJson: JSON.stringify([{ role: "user", text: "hi" }]),
      updatedBy: "Team Lead",
    });

    const row = getDeliverable(projectId, "scoping");
    expect(row?.contentJson).toContain("Kroger");
    expect(row?.transcriptJson).toContain("hi");
    expect(row?.lastUpdatedBy).toBe("Team Lead");
  });

  it("saving before a get-or-create still creates the row", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "225",
      name: "Lazy project",
      organization: "",
      coachTypes: ["scoping"],
    });
    saveDeliverableContent({
      projectId,
      coachType: "scoping",
      contentJson: JSON.stringify({ contacts: "Jo" }),
      updatedBy: null,
    });
    expect(getDeliverable(projectId, "scoping")?.contentJson).toContain("Jo");
  });

  it("lists deliverables for a project", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "496",
      name: "List project",
      organization: "",
      coachTypes: ["scoping", "reflection"],
    });
    getOrCreateDeliverable(projectId, "scoping");
    getOrCreateDeliverable(projectId, "reflection");
    const types = listDeliverables(projectId).map((d) => d.coachType).sort();
    expect(types).toEqual(["reflection", "scoping"]);
  });
});

describe("team management", () => {
  it("identifies the lead as the owner only", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "401/501",
      name: "Lead check", organization: "", coachTypes: [],
    });
    addProjectMember({ projectId, email: MATE, name: "Teammate", role: "member" });
    expect(isProjectLead(projectId, LEAD)).toBe(true);
    expect(isProjectLead(projectId, MATE)).toBe(false);
    expect(isProjectLead(projectId, STRANGER)).toBe(false);
  });

  it("removes a member but never the owner", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "444/544",
      name: "Remove", organization: "", coachTypes: [],
    });
    addProjectMember({ projectId, email: MATE, name: "Teammate", role: "member" });

    expect(removeProjectMember(projectId, MATE)).toBe(true);
    expect(isProjectMember(projectId, MATE)).toBe(false);
    // The owner cannot be removed.
    expect(removeProjectMember(projectId, LEAD)).toBe(false);
    expect(isProjectMember(projectId, LEAD)).toBe(true);
  });

  it("updates which coaches a project includes", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "225",
      name: "Coaches", organization: "", coachTypes: ["scoping"],
    });
    updateProjectCoaches(projectId, ["premortem", "reflection"]);
    expect(getAccessibleProject(projectId, LEAD)?.coachTypes).toEqual(["premortem", "reflection"]);
  });

  it("fills a member name only when it was empty", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "496",
      name: "Names", organization: "", coachTypes: [],
    });
    addProjectMember({ projectId, email: MATE, name: null, role: "member" });
    fillMemberName(projectId, MATE, "Real Name");
    expect(listProjectMembers(projectId).find((m) => m.email === MATE)?.name).toBe("Real Name");
    // Does not overwrite an existing name.
    fillMemberName(projectId, MATE, "Different");
    expect(listProjectMembers(projectId).find((m) => m.email === MATE)?.name).toBe("Real Name");
  });
});

describe("deleting a project", () => {
  it("lets only the owner delete, cascading to members and deliverables", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "401/501",
      name: "To delete", organization: "", coachTypes: ["scoping"],
    });
    addProjectMember({ projectId, email: MATE, name: "Teammate", role: "member" });
    getOrCreateDeliverable(projectId, "scoping");

    // A member cannot delete it.
    expect(deleteProject(projectId, MATE)).toBe(false);
    expect(getAccessibleProject(projectId, LEAD)).toBeDefined();

    // The owner can, and it takes the members and deliverables with it.
    expect(deleteProject(projectId, LEAD)).toBe(true);
    expect(getAccessibleProject(projectId, LEAD)).toBeUndefined();
    expect(listProjectMembers(projectId)).toHaveLength(0);
    expect(listDeliverables(projectId)).toHaveLength(0);
  });
});
