import {
  index,
  integer,
  real,
  sqliteTable,
  text,
  uniqueIndex,
} from "drizzle-orm/sqlite-core";

/**
 * Users are identified by their Miami email. `role` is reserved for a
 * future admin dashboard (ADR-004); everyone is a student for now.
 * Timestamps are ISO-8601 UTC strings.
 */
export const users = sqliteTable("users", {
  email: text("email").primaryKey(),
  name: text("name"),
  role: text("role").notNull().default("student"),
  firstSeenAt: text("first_seen_at").notNull(),
  lastSeenAt: text("last_seen_at").notNull(),
});

/**
 * An uploaded study document. The PDF itself is never written to disk.
 *
 * Retention (ADR-015): page text lives in `examDocumentPages` only long enough
 * to build an exam from it, then it is deleted. What survives is this metadata
 * plus the short quotes actually cited by questions, so the server never holds
 * a standing copy of a textbook.
 */
export const examDocuments = sqliteTable(
  "exam_documents",
  {
    id: text("id").primaryKey(),
    userEmail: text("user_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    /** Display only. Never used as a filesystem path. */
    filename: text("filename").notNull(),
    sizeBytes: integer("size_bytes").notNull(),
    pageCount: integer("page_count").notNull(),
    /** Pages read directly from the PDF's own text layer. */
    textPageCount: integer("text_page_count").notNull(),
    /** Pages transcribed by a vision model (ADR-014). */
    visionPageCount: integer("vision_page_count").notNull().default(0),
    charCount: integer("char_count").notNull(),
    classification: text("classification").notNull(),
    warningsJson: text("warnings_json").notNull().default("[]"),
    /** Set when the transient page text has been purged. */
    textPurgedAt: text("text_purged_at"),
    createdAt: text("created_at").notNull(),
  },
  (t) => [index("exam_documents_user_created").on(t.userEmail, t.createdAt)],
);

/**
 * Transient per-page text. Page granularity is what makes page-range scoping
 * and page citations possible. Rows are deleted once an exam has been built,
 * and swept on a time limit even if that never happens.
 */
export const examDocumentPages = sqliteTable(
  "exam_document_pages",
  {
    id: text("id").primaryKey(),
    documentId: text("document_id")
      .notNull()
      .references(() => examDocuments.id, { onDelete: "cascade" }),
    pageNumber: integer("page_number").notNull(),
    text: text("text").notNull(),
    charCount: integer("char_count").notNull(),
    /** "text" or "vision", so provenance can be shown to the student. */
    source: text("source").notNull(),
  },
  (t) => [
    uniqueIndex("exam_document_pages_doc_page").on(t.documentId, t.pageNumber),
  ],
);

export const exams = sqliteTable(
  "exams",
  {
    id: text("id").primaryKey(),
    userEmail: text("user_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    documentId: text("document_id")
      .notNull()
      .references(() => examDocuments.id, { onDelete: "cascade" }),
    modelId: text("model_id").notNull(),
    /** generating | ready | in_progress | completed | failed */
    status: text("status").notNull(),
    failureReason: text("failure_reason"),
    /** practice (immediate feedback) or exam (feedback withheld). */
    examMode: text("exam_mode").notNull(),
    questionType: text("question_type").notNull(),
    requestedCount: integer("requested_count").notNull(),
    deliveredCount: integer("delivered_count").notNull().default(0),
    /** Questions discarded because they could not be traced to the document. */
    droppedCount: integer("dropped_count").notNull().default(0),
    scopeFromPage: integer("scope_from_page").notNull(),
    scopeToPage: integer("scope_to_page").notNull(),
    /** What was actually covered, reported to the student verbatim. */
    coverageJson: text("coverage_json").notNull().default("{}"),
    currentPosition: integer("current_position").notNull().default(0),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
    completedAt: text("completed_at"),
  },
  (t) => [
    index("exams_user_updated").on(t.userEmail, t.updatedAt),
    index("exams_document").on(t.documentId),
  ],
);

export const examQuestions = sqliteTable(
  "exam_questions",
  {
    id: text("id").primaryKey(),
    examId: text("exam_id")
      .notNull()
      .references(() => exams.id, { onDelete: "cascade" }),
    position: integer("position").notNull(),
    type: text("type").notNull(),
    stem: text("stem").notNull(),
    optionsJson: text("options_json"),
    correctIndex: integer("correct_index"),
    modelAnswer: text("model_answer").notNull(),
    rubricJson: text("rubric_json").notNull(),
    explanation: text("explanation").notNull(),
    topic: text("topic").notNull(),
    bloom: text("bloom").notNull(),
    /** Short verbatim quote proving the question came from the document. */
    sourceQuote: text("source_quote").notNull(),
    sourcePage: integer("source_page").notNull(),
    groundingStatus: text("grounding_status").notNull(),
    pointsPossible: integer("points_possible").notNull(),
  },
  (t) => [
    uniqueIndex("exam_questions_exam_position").on(t.examId, t.position),
  ],
);

export const examAnswers = sqliteTable(
  "exam_answers",
  {
    id: text("id").primaryKey(),
    examId: text("exam_id")
      .notNull()
      .references(() => exams.id, { onDelete: "cascade" }),
    questionId: text("question_id")
      .notNull()
      .references(() => examQuestions.id, { onDelete: "cascade" }),
    selectedIndex: integer("selected_index"),
    responseText: text("response_text"),
    confidence: text("confidence"),
    /** local | model | failed. Never scored zero for our own failures. */
    gradedBy: text("graded_by"),
    graderModelId: text("grader_model_id"),
    isCorrect: integer("is_correct"),
    pointsAwarded: real("points_awarded"),
    criteriaJson: text("criteria_json"),
    feedback: text("feedback"),
    createdAt: text("created_at").notNull(),
    gradedAt: text("graded_at"),
  },
  (t) => [uniqueIndex("exam_answers_question").on(t.questionId)],
);

/**
 * One job a student is applying for.
 *
 * Set up once and shared by both halves of JobApp Assistant: tailoring the
 * application, then practising the interview. The student should not have to
 * describe the same job twice.
 *
 * The posting text is stored because the student may edit it and will come back
 * to it. The resume is not: only the derived brief is kept, consistent with
 * ADR-015, so no standing copy of a student's personal history accumulates.
 */
export const jobApplications = sqliteTable(
  "job_applications",
  {
    id: text("id").primaryKey(),
    userEmail: text("user_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    company: text("company").notNull(),
    positionTitle: text("position_title").notNull(),
    jobUrl: text("job_url"),
    /** fetched | pasted | none. Recorded so the UI can be honest about where
     * the description came from and whether reading the link worked. */
    descriptionSource: text("description_source").notNull().default("none"),
    postingText: text("posting_text"),
    /** Short derived summaries. Never the resume or the full posting. */
    roleBrief: text("role_brief"),
    candidateBrief: text("candidate_brief"),
    /** Kept so tailoring can quote the student's own lines back at them. */
    resumeText: text("resume_text"),
    resumeFilename: text("resume_filename"),
    resumePurgedAt: text("resume_purged_at"),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [index("job_applications_user").on(t.userEmail, t.createdAt)],
);

/**
 * A tailored resume or cover letter.
 *
 * `contentJson` holds the structured document, not prose, so the same record
 * renders to both the print page and the .docx without the two drifting.
 * `reviewedAt` is set only once the student has been through it: nothing
 * exports until then, because the student is the one making the claims.
 */
export const tailoredDocuments = sqliteTable(
  "tailored_documents",
  {
    id: text("id").primaryKey(),
    applicationId: text("application_id")
      .notNull()
      .references(() => jobApplications.id, { onDelete: "cascade" }),
    userEmail: text("user_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    /** resume | cover_letter */
    kind: text("kind").notNull(),
    /** FSB template 1, 2 or 3. */
    template: integer("template").notNull().default(2),
    modelId: text("model_id").notNull(),
    contentJson: text("content_json").notNull(),
    /** Bullets the generator could not trace to the student's own resume. */
    ungroundedJson: text("ungrounded_json"),
    reviewedAt: text("reviewed_at"),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [index("tailored_documents_application").on(t.applicationId, t.kind)],
);

/**
 * A mock interview session.
 *
 * The legacy module kept everything in Streamlit session state, so a refresh,
 * a reconnect, or navigating away destroyed the interview and the student was
 * left with nothing: the spoken feedback was said once and gone. Persisting
 * the session is what turns this from a disposable demo into something a
 * student can finish, return to, and keep.
 *
 * The resume is deliberately NOT stored. Only the short derived brief the
 * interviewer actually needs is kept, so a standing copy of a student's
 * personal history does not accumulate on the server.
 */
export const interviews = sqliteTable(
  "interviews",
  {
    id: text("id").primaryKey(),
    userEmail: text("user_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    modelId: text("model_id").notNull(),
    /** Set when the interview was started from a saved job application. */
    applicationId: text("application_id").references(() => jobApplications.id, {
      onDelete: "set null",
    }),
    /** behavioral | technical | case | mixed */
    interviewType: text("interview_type").notNull(),
    /** in_progress | completed | abandoned */
    status: text("status").notNull(),
    jobTitle: text("job_title").notNull(),
    /** Short derived summary of the target role. Never the full posting. */
    roleBrief: text("role_brief"),
    /** Short derived summary of the student's background. Never the resume. */
    candidateBrief: text("candidate_brief"),
    gradeLevel: text("grade_level"),
    major: text("major"),
    plannedQuestions: integer("planned_questions").notNull(),
    askedCount: integer("asked_count").notNull().default(0),
    /** Set only when the whole interview is finished and reviewed. */
    summaryJson: text("summary_json"),
    createdAt: text("created_at").notNull(),
    completedAt: text("completed_at"),
  },
  (t) => [index("interviews_user_created").on(t.userEmail, t.createdAt)],
);

/**
 * One question and the student's answer.
 *
 * `answerSource` records whether the student spoke or typed. Speech is an
 * enhancement, never a requirement: the interview is fully answerable by
 * typing, which the legacy module made impossible.
 */
export const interviewTurns = sqliteTable(
  "interview_turns",
  {
    id: text("id").primaryKey(),
    interviewId: text("interview_id")
      .notNull()
      .references(() => interviews.id, { onDelete: "cascade" }),
    ordinal: integer("ordinal").notNull(),
    question: text("question").notNull(),
    /** What the question is probing, for the per-topic summary at the end. */
    topic: text("topic"),
    answerText: text("answer_text"),
    /** spoken | typed | skipped */
    answerSource: text("answer_source"),
    /** Seconds the student spent answering. Null when typed. */
    answerSeconds: integer("answer_seconds"),
    /** Per-criterion judgements, from the model, bound to a fixed rubric. */
    criteriaJson: text("criteria_json"),
    strength: text("strength"),
    improvement: text("improvement"),
    askedAt: text("asked_at").notNull(),
    answeredAt: text("answered_at"),
  },
  (t) => [uniqueIndex("interview_turns_ordinal").on(t.interviewId, t.ordinal)],
);

/**
 * Privacy-filtered analytics replacing logs/activity_log.json (ADR-006).
 * Never stores prompt or response text: lengths and counts only.
 */
export const usageEvents = sqliteTable(
  "usage_events",
  {
    id: text("id").primaryKey(),
    createdAt: text("created_at").notNull(),
    userEmail: text("user_email"),
    module: text("module").notNull(),
    eventType: text("event_type").notNull(),
    modelId: text("model_id"),
    provider: text("provider"),
    inputTokens: integer("input_tokens"),
    outputTokens: integer("output_tokens"),
    costUsd: real("cost_usd"),
    latencyMs: integer("latency_ms"),
    promptChars: integer("prompt_chars"),
    responseChars: integer("response_chars"),
    outcome: text("outcome"),
  },
  (t) => [index("usage_events_created").on(t.createdAt)],
);

/**
 * Job Scout's weekly-harvested postings (design 2026-07-28). Public employer
 * content only: no student column exists here on purpose, and none may be
 * added — which jobs a student views or saves lives in their browser
 * (local-first decision, 2026-07-28).
 */
export const scoutPostings = sqliteTable(
  "scout_postings",
  {
    id: text("id").primaryKey(),
    /** jsearch | usajobs */
    source: text("source").notNull(),
    /** The source's own job id; uniqueness is per source. */
    externalId: text("external_id").notNull(),
    /** lower(company)|lower(title)|state — cross-source duplicate detection. */
    fingerprint: text("fingerprint").notNull(),
    title: text("title").notNull(),
    company: text("company").notNull(),
    locationCity: text("location_city"),
    locationState: text("location_state"),
    remote: integer("remote", { mode: "boolean" }).notNull().default(false),
    /** fulltime | internship | federal */
    category: text("category").notNull(),
    /** The employer's own application URL. Job Scout never proxies applying. */
    applyUrl: text("apply_url").notNull(),
    description: text("description").notNull(),
    postedAt: text("posted_at"),
    harvestedAt: text("harvested_at").notNull(),
    /** Bumped each harvest that still sees the posting; drives retirement. */
    lastSeenAt: text("last_seen_at").notNull(),
    /** [{skillId, importance: "required"|"preferred"}] from the tagging pass. */
    skillsJson: text("skills_json").notNull().default("[]"),
    /**
     * sponsors | no_sponsorship | unknown, from the tagging pass
     * (user request 2026-07-28). "sponsors"/"no_sponsorship" only on the
     * posting's own explicit words; everything else stays unknown, because a
     * wrong "sponsors" wastes an international student's application.
     */
    visaSponsorship: text("visa_sponsorship").notNull().default("unknown"),
    /** Vocabulary version the tags were made with (lib/scout/taxonomy.ts). */
    taxonomyVersion: integer("taxonomy_version").notNull(),
    active: integer("active", { mode: "boolean" }).notNull().default(true),
  },
  (t) => [
    uniqueIndex("scout_postings_source_external").on(t.source, t.externalId),
    index("scout_postings_active_category").on(t.active, t.category),
  ],
);

/**
 * One row per harvest run: the scheduler's persisted memory (a restart must
 * derive "is a run due" from here, never from an in-memory timer) and the
 * operator's cost/coverage record.
 */
export const scoutRuns = sqliteTable(
  "scout_runs",
  {
    id: text("id").primaryKey(),
    startedAt: text("started_at").notNull(),
    finishedAt: text("finished_at"),
    /** running | completed | partial | failed */
    status: text("status").notNull(),
    /** schedule | manual */
    trigger: text("trigger").notNull(),
    jsearchRequests: integer("jsearch_requests").notNull().default(0),
    jsearchFound: integer("jsearch_found").notNull().default(0),
    usajobsRequests: integer("usajobs_requests").notNull().default(0),
    usajobsFound: integer("usajobs_found").notNull().default(0),
    dedupedCount: integer("deduped_count").notNull().default(0),
    taggedCount: integer("tagged_count").notNull().default(0),
    costUsd: real("cost_usd").notNull().default(0),
    /** Per-source failure notes, JSON {jsearch?: string, usajobs?: string}. */
    sourceErrorsJson: text("source_errors_json").notNull().default("{}"),
    error: text("error"),
  },
  (t) => [index("scout_runs_started").on(t.startedAt)],
);

/**
 * A team project workspace, scoped to a course the student selects (ADR-010).
 * "Course" is a label only; no enrollment data is looked up or stored.
 * `coachTypesJson` is the subset of coaches the lead enabled, as a JSON array.
 */
export const projects = sqliteTable(
  "projects",
  {
    id: text("id").primaryKey(),
    courseCode: text("course_code").notNull(),
    name: text("name").notNull(),
    organization: text("organization").notNull().default(""),
    ownerEmail: text("owner_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    coachTypesJson: text("coach_types_json").notNull().default("[]"),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [index("projects_owner_updated").on(t.ownerEmail, t.updatedAt)],
);

/**
 * Team membership. Identifies teammates by their authenticated Google name and
 * `@miamioh.edu` email only. No student or Banner ID. The `email` index powers
 * the "shared with me" query for an invited student.
 */
export const projectMembers = sqliteTable(
  "project_members",
  {
    id: text("id").primaryKey(),
    projectId: text("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    email: text("email").notNull(),
    /** Known once the member has signed in at least once; may be null before. */
    name: text("name"),
    /** lead | member. Exactly one lead: the creator. */
    role: text("role").notNull(),
    createdAt: text("created_at").notNull(),
  },
  (t) => [
    uniqueIndex("project_members_project_email").on(t.projectId, t.email),
    index("project_members_email").on(t.email),
  ],
);

/**
 * One deliverable per enabled coach per project, created lazily on first use.
 * `contentJson` is the structured deliverable (schema per coach); `transcriptJson`
 * is the coach chat. `lastUpdatedBy` is a display name (or email) for the
 * "last updated by <name>" note, since edits are shared last-save-wins.
 */
export const deliverables = sqliteTable(
  "deliverables",
  {
    id: text("id").primaryKey(),
    projectId: text("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    coachType: text("coach_type").notNull(),
    contentJson: text("content_json").notNull().default("{}"),
    transcriptJson: text("transcript_json").notNull().default("[]"),
    lastUpdatedBy: text("last_updated_by"),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [
    uniqueIndex("deliverables_project_coach").on(t.projectId, t.coachType),
  ],
);
