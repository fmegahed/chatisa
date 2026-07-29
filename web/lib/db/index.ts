import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";
import Database from "better-sqlite3";
import { drizzle, type BetterSQLite3Database } from "drizzle-orm/better-sqlite3";
import { migrate } from "drizzle-orm/better-sqlite3/migrator";
import { and, desc, eq, isNull, lt, sql } from "drizzle-orm";
import * as schema from "./schema";

/**
 * SQLite client (ADR-002). Single file under web/data/ (gitignored +
 * locally excluded). Import lazily from server code only: never from
 * proxy.ts or client components.
 */
type Db = BetterSQLite3Database<typeof schema>;

const globalForDb = globalThis as unknown as {
  __chatisaDb?: Db;
  __chatisaDbClient?: Database.Database;
  __chatisaScoutDb?: Db;
  __chatisaScoutDbClient?: Database.Database;
};

function createDb(): Db {
  const dataDir =
    process.env.CHATISA_DATA_DIR ?? path.join(process.cwd(), "data");
  fs.mkdirSync(dataDir, { recursive: true });
  const client = new Database(path.join(dataDir, "chatisa.db"));
  client.pragma("journal_mode = WAL");
  client.pragma("foreign_keys = ON");
  globalForDb.__chatisaDbClient = client;
  const db = drizzle(client, { schema });
  migrate(db, { migrationsFolder: path.join(process.cwd(), "drizzle") });
  return db;
}

export function getDb(): Db {
  if (!globalForDb.__chatisaDb) globalForDb.__chatisaDb = createDb();
  return globalForDb.__chatisaDb;
}

/**
 * Job Scout's OWN database file, scout.db beside chatisa.db (ADR-027).
 * Separate on purpose: the 2026-07-29 source swap (JSearch out, Active Jobs
 * DB in) must not mix aggregator-era postings with the employer-direct feed,
 * and the user chose to keep the old rows rather than delete them. They stay
 * in chatisa.db's scout tables, which nothing reads anymore; this fresh file
 * starts empty and only ever holds the new sources. Postings are a public
 * cache with their own lifecycle, so a future source swap can reset this
 * file without touching users or usage events.
 *
 * Tables are created inline rather than through drizzle migrations: the main
 * migration chain targets chatisa.db, and a second chain for two tables
 * would be more machinery than the DDL it manages.
 */
const SCOUT_DDL = `
CREATE TABLE IF NOT EXISTS scout_postings (
  id text PRIMARY KEY NOT NULL,
  source text NOT NULL,
  external_id text NOT NULL,
  fingerprint text NOT NULL,
  title text NOT NULL,
  company text NOT NULL,
  location_city text,
  location_state text,
  remote integer DEFAULT false NOT NULL,
  category text NOT NULL,
  apply_url text NOT NULL,
  description text NOT NULL,
  posted_at text,
  harvested_at text NOT NULL,
  last_seen_at text NOT NULL,
  skills_json text DEFAULT '[]' NOT NULL,
  taxonomy_version integer NOT NULL,
  active integer DEFAULT true NOT NULL,
  visa_sponsorship text DEFAULT 'unknown' NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS scout_postings_source_external
  ON scout_postings (source, external_id);
CREATE INDEX IF NOT EXISTS scout_postings_active_category
  ON scout_postings (active, category);
CREATE TABLE IF NOT EXISTS scout_runs (
  id text PRIMARY KEY NOT NULL,
  started_at text NOT NULL,
  finished_at text,
  status text NOT NULL,
  trigger text NOT NULL,
  activejobs_requests integer DEFAULT 0 NOT NULL,
  activejobs_found integer DEFAULT 0 NOT NULL,
  usajobs_requests integer DEFAULT 0 NOT NULL,
  usajobs_found integer DEFAULT 0 NOT NULL,
  deduped_count integer DEFAULT 0 NOT NULL,
  tagged_count integer DEFAULT 0 NOT NULL,
  cost_usd real DEFAULT 0 NOT NULL,
  source_errors_json text DEFAULT '{}' NOT NULL,
  error text
);
CREATE INDEX IF NOT EXISTS scout_runs_started ON scout_runs (started_at);
`;

function createScoutDb(): Db {
  const dataDir =
    process.env.CHATISA_DATA_DIR ?? path.join(process.cwd(), "data");
  fs.mkdirSync(dataDir, { recursive: true });
  const client = new Database(path.join(dataDir, "scout.db"));
  client.pragma("journal_mode = WAL");
  client.exec(SCOUT_DDL);
  globalForDb.__chatisaScoutDbClient = client;
  return drizzle(client, { schema });
}

export function getScoutDb(): Db {
  if (!globalForDb.__chatisaScoutDb)
    globalForDb.__chatisaScoutDb = createScoutDb();
  return globalForDb.__chatisaScoutDb;
}

/**
 * Releases the database file. Needed for tidy shutdown and for tests, since
 * Windows will not delete a file while a handle is open.
 */
export function closeDb(): void {
  globalForDb.__chatisaDbClient?.close();
  globalForDb.__chatisaDbClient = undefined;
  globalForDb.__chatisaDb = undefined;
  globalForDb.__chatisaScoutDbClient?.close();
  globalForDb.__chatisaScoutDbClient = undefined;
  globalForDb.__chatisaScoutDb = undefined;
}

/** Record a successful sign-in, creating the user on first visit. */
export function upsertUser(email: string, name: string | null): void {
  const now = new Date().toISOString();
  getDb()
    .insert(schema.users)
    .values({
      email: email.toLowerCase(),
      name,
      firstSeenAt: now,
      lastSeenAt: now,
    })
    .onConflictDoUpdate({
      target: schema.users.email,
      set: { lastSeenAt: now, ...(name ? { name } : {}) },
    })
    .run();
}

/** Cheap connectivity probe for the readiness check. */
export function dbReady(): boolean {
  try {
    getDb().run(sql`select 1`);
    return true;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------- exam ally

export interface StoredPage {
  pageNumber: number;
  text: string;
  charCount: number;
  source: string;
}

/** Stores document metadata plus its transient page text in one transaction. */
export function createExamDocument(params: {
  userEmail: string;
  filename: string;
  sizeBytes: number;
  pageCount: number;
  textPageCount: number;
  visionPageCount: number;
  charCount: number;
  classification: string;
  warnings: string[];
  pages: StoredPage[];
}): string {
  const db = getDb();
  const id = randomUUID();
  const now = new Date().toISOString();
  db.transaction((tx) => {
    tx.insert(schema.examDocuments)
      .values({
        id,
        userEmail: params.userEmail.toLowerCase(),
        filename: params.filename,
        sizeBytes: params.sizeBytes,
        pageCount: params.pageCount,
        textPageCount: params.textPageCount,
        visionPageCount: params.visionPageCount,
        charCount: params.charCount,
        classification: params.classification,
        warningsJson: JSON.stringify(params.warnings),
        createdAt: now,
      })
      .run();
    for (const page of params.pages) {
      if (page.text.length === 0) continue;
      tx.insert(schema.examDocumentPages)
        .values({
          id: randomUUID(),
          documentId: id,
          pageNumber: page.pageNumber,
          text: page.text,
          charCount: page.charCount,
          source: page.source,
        })
        .run();
    }
  });
  return id;
}

/** Returns the document only when it belongs to this user. */
export function getOwnedExamDocument(id: string, userEmail: string) {
  return getDb()
    .select()
    .from(schema.examDocuments)
    .where(
      and(
        eq(schema.examDocuments.id, id),
        eq(schema.examDocuments.userEmail, userEmail.toLowerCase()),
      ),
    )
    .get();
}

export function listExamDocuments(userEmail: string) {
  return getDb()
    .select()
    .from(schema.examDocuments)
    .where(eq(schema.examDocuments.userEmail, userEmail.toLowerCase()))
    .orderBy(desc(schema.examDocuments.createdAt))
    .all();
}

/** Page text within a scope. Empty once the text has been purged. */
export function getDocumentPages(
  documentId: string,
  fromPage?: number,
  toPage?: number,
) {
  const rows = getDb()
    .select()
    .from(schema.examDocumentPages)
    .where(eq(schema.examDocumentPages.documentId, documentId))
    .orderBy(schema.examDocumentPages.pageNumber)
    .all();
  if (fromPage === undefined || toPage === undefined) return rows;
  return rows.filter(
    (r) => r.pageNumber >= fromPage && r.pageNumber <= toPage,
  );
}

/**
 * Drops the transient page text once it is no longer needed (ADR-015), so the
 * server keeps metadata and short cited quotes rather than whole textbooks.
 */
export function purgeDocumentText(documentId: string): void {
  const db = getDb();
  db.delete(schema.examDocumentPages)
    .where(eq(schema.examDocumentPages.documentId, documentId))
    .run();
  db.update(schema.examDocuments)
    .set({ textPurgedAt: new Date().toISOString() })
    .where(eq(schema.examDocuments.id, documentId))
    .run();
}

/** Safety net: purge page text for documents older than the cutoff. */
export function purgeExpiredDocumentText(olderThanHours = 24): number {
  const cutoff = new Date(
    Date.now() - olderThanHours * 60 * 60 * 1000,
  ).toISOString();
  const stale = getDb()
    .select({ id: schema.examDocuments.id })
    .from(schema.examDocuments)
    .where(
      and(
        lt(schema.examDocuments.createdAt, cutoff),
        isNull(schema.examDocuments.textPurgedAt),
      ),
    )
    .all();
  for (const row of stale) purgeDocumentText(row.id);
  return stale.length;
}

export function deleteExamDocument(id: string, userEmail: string): boolean {
  if (!getOwnedExamDocument(id, userEmail)) return false;
  getDb()
    .delete(schema.examDocuments)
    .where(eq(schema.examDocuments.id, id))
    .run();
  return true;
}

export interface NewExamQuestion {
  type: string;
  stem: string;
  options: string[] | null;
  correctIndex: number | null;
  modelAnswer: string;
  rubric: { criterion: string; points: number }[];
  explanation: string;
  topic: string;
  bloom: string;
  sourceQuote: string;
  sourcePage: number;
  groundingStatus: string;
  pointsPossible: number;
}

/** Creates the exam and its questions in one transaction. */
export function createExamWithQuestions(params: {
  userEmail: string;
  documentId: string;
  modelId: string;
  examMode: string;
  questionType: string;
  requestedCount: number;
  droppedCount: number;
  scopeFromPage: number;
  scopeToPage: number;
  coverage: unknown;
  questions: NewExamQuestion[];
}): string {
  const db = getDb();
  const id = randomUUID();
  const now = new Date().toISOString();
  db.transaction((tx) => {
    tx.insert(schema.exams)
      .values({
        id,
        userEmail: params.userEmail.toLowerCase(),
        documentId: params.documentId,
        modelId: params.modelId,
        status: "ready",
        examMode: params.examMode,
        questionType: params.questionType,
        requestedCount: params.requestedCount,
        deliveredCount: params.questions.length,
        droppedCount: params.droppedCount,
        scopeFromPage: params.scopeFromPage,
        scopeToPage: params.scopeToPage,
        coverageJson: JSON.stringify(params.coverage ?? {}),
        createdAt: now,
        updatedAt: now,
      })
      .run();

    params.questions.forEach((q, index) => {
      tx.insert(schema.examQuestions)
        .values({
          id: randomUUID(),
          examId: id,
          position: index,
          type: q.type,
          stem: q.stem,
          optionsJson: q.options ? JSON.stringify(q.options) : null,
          correctIndex: q.correctIndex,
          modelAnswer: q.modelAnswer,
          rubricJson: JSON.stringify(q.rubric),
          explanation: q.explanation,
          topic: q.topic,
          bloom: q.bloom,
          sourceQuote: q.sourceQuote,
          sourcePage: q.sourcePage,
          groundingStatus: q.groundingStatus,
          pointsPossible: q.pointsPossible,
        })
        .run();
    });
  });
  return id;
}

export function getExamQuestions(examId: string) {
  return getDb()
    .select()
    .from(schema.examQuestions)
    .where(eq(schema.examQuestions.examId, examId))
    .orderBy(schema.examQuestions.position)
    .all();
}

export function getExamAnswers(examId: string) {
  return getDb()
    .select()
    .from(schema.examAnswers)
    .where(eq(schema.examAnswers.examId, examId))
    .all();
}

export function getAnswerForQuestion(questionId: string) {
  return getDb()
    .select()
    .from(schema.examAnswers)
    .where(eq(schema.examAnswers.questionId, questionId))
    .get();
}

/** One answer per question: a resubmission updates rather than duplicates. */
export function saveAnswer(params: {
  examId: string;
  questionId: string;
  selectedIndex: number | null;
  responseText: string | null;
  confidence: string | null;
  gradedBy: string | null;
  graderModelId: string | null;
  isCorrect: boolean | null;
  pointsAwarded: number | null;
  criteria: unknown;
  feedback: string | null;
}): void {
  const now = new Date().toISOString();
  const values = {
    id: randomUUID(),
    examId: params.examId,
    questionId: params.questionId,
    selectedIndex: params.selectedIndex,
    responseText: params.responseText,
    confidence: params.confidence,
    gradedBy: params.gradedBy,
    graderModelId: params.graderModelId,
    isCorrect:
      params.isCorrect === null ? null : params.isCorrect ? 1 : 0,
    pointsAwarded: params.pointsAwarded,
    criteriaJson: params.criteria ? JSON.stringify(params.criteria) : null,
    feedback: params.feedback,
    createdAt: now,
    gradedAt: params.gradedBy ? now : null,
  };
  getDb()
    .insert(schema.examAnswers)
    .values(values)
    .onConflictDoUpdate({
      target: schema.examAnswers.questionId,
      set: {
        selectedIndex: values.selectedIndex,
        responseText: values.responseText,
        confidence: values.confidence,
        gradedBy: values.gradedBy,
        graderModelId: values.graderModelId,
        isCorrect: values.isCorrect,
        pointsAwarded: values.pointsAwarded,
        criteriaJson: values.criteriaJson,
        feedback: values.feedback,
        gradedAt: values.gradedAt,
      },
    })
    .run();
}

/** Records progress and marks the exam finished when it is. */
export function advanceExam(params: {
  examId: string;
  position: number;
  complete: boolean;
}): void {
  const now = new Date().toISOString();
  getDb()
    .update(schema.exams)
    .set({
      currentPosition: params.position,
      status: params.complete ? "completed" : "in_progress",
      updatedAt: now,
      ...(params.complete ? { completedAt: now } : {}),
    })
    .where(eq(schema.exams.id, params.examId))
    .run();
}

export function deleteExam(id: string, userEmail: string): boolean {
  if (!getOwnedExam(id, userEmail)) return false;
  getDb().delete(schema.exams).where(eq(schema.exams.id, id)).run();
  return true;
}

/** Returns the exam only when it belongs to this user. */
export function getOwnedExam(id: string, userEmail: string) {
  return getDb()
    .select()
    .from(schema.exams)
    .where(
      and(
        eq(schema.exams.id, id),
        eq(schema.exams.userEmail, userEmail.toLowerCase()),
      ),
    )
    .get();
}

export function listExams(userEmail: string) {
  return getDb()
    .select()
    .from(schema.exams)
    .where(eq(schema.exams.userEmail, userEmail.toLowerCase()))
    .orderBy(desc(schema.exams.updatedAt))
    .all();
}

// ---------------------------------------------------------------- analytics

/**
 * Privacy-filtered usage record (ADR-006). Callers must pass lengths, never
 * prompt or response text.
 */
export function recordUsageEvent(event: {
  userEmail?: string | null;
  module: string;
  eventType: string;
  modelId?: string | null;
  provider?: string | null;
  inputTokens?: number | null;
  outputTokens?: number | null;
  costUsd?: number | null;
  latencyMs?: number | null;
  promptChars?: number | null;
  responseChars?: number | null;
  outcome?: string | null;
}): void {
  try {
    getDb()
      .insert(schema.usageEvents)
      .values({
        id: randomUUID(),
        createdAt: new Date().toISOString(),
        userEmail: event.userEmail?.toLowerCase() ?? null,
        module: event.module,
        eventType: event.eventType,
        modelId: event.modelId ?? null,
        provider: event.provider ?? null,
        inputTokens: event.inputTokens ?? null,
        outputTokens: event.outputTokens ?? null,
        costUsd: event.costUsd ?? null,
        latencyMs: event.latencyMs ?? null,
        promptChars: event.promptChars ?? null,
        responseChars: event.responseChars ?? null,
        outcome: event.outcome ?? null,
      })
      .run();
  } catch {
    // Analytics must never break a student's conversation.
  }
}

/**
 * Deep-health probe (2026-07-25): a REAL write and read-back through the
 * native driver, throwing on failure. recordUsageEvent deliberately swallows
 * errors, so it cannot serve as a health signal; this can. The probe row is
 * deleted immediately so health checks never accumulate in analytics.
 */
export function dbWriteProbe(): boolean {
  const id = randomUUID();
  getDb()
    .insert(schema.usageEvents)
    .values({
      id,
      createdAt: new Date().toISOString(),
      userEmail: null,
      module: "health",
      eventType: "deep_health_probe",
    })
    .run();
  const rows = getDb()
    .select({ id: schema.usageEvents.id })
    .from(schema.usageEvents)
    .where(eq(schema.usageEvents.id, id))
    .all();
  getDb().delete(schema.usageEvents).where(eq(schema.usageEvents.id, id)).run();
  return rows.length === 1;
}

/** Total spend for a user in the current module, for the cost display. */
export function sumUserCost(userEmail: string, module?: string): number {
  const rows = getDb()
    .select({ cost: schema.usageEvents.costUsd })
    .from(schema.usageEvents)
    .where(
      module
        ? and(
            eq(schema.usageEvents.userEmail, userEmail.toLowerCase()),
            eq(schema.usageEvents.module, module),
          )
        : eq(schema.usageEvents.userEmail, userEmail.toLowerCase()),
    )
    .all();
  return rows.reduce((sum, r) => sum + (r.cost ?? 0), 0);
}

/* ---------------------------------------------------------------- interviews */

export function createInterview(params: {
  userEmail: string;
  modelId: string;
  interviewType: string;
  jobTitle: string;
  roleBrief: string | null;
  candidateBrief: string | null;
  gradeLevel: string | null;
  major: string | null;
  plannedQuestions: number;
  /** Set when started from a saved application (JobApp handoff, 2026-07-28).
   * Callers must have ownership-checked it; this function trusts them. */
  applicationId?: string | null;
}): string {
  const db = getDb();
  const id = randomUUID();
  db.insert(schema.interviews)
    .values({
      id,
      userEmail: params.userEmail,
      modelId: params.modelId,
      interviewType: params.interviewType,
      status: "in_progress",
      jobTitle: params.jobTitle,
      roleBrief: params.roleBrief,
      candidateBrief: params.candidateBrief,
      gradeLevel: params.gradeLevel,
      major: params.major,
      plannedQuestions: params.plannedQuestions,
      applicationId: params.applicationId ?? null,
      askedCount: 0,
      createdAt: new Date().toISOString(),
    })
    .run();
  return id;
}

/** Ownership-checked read. Returns undefined for another student's interview,
 * which callers turn into the same not-found response as a bad id. */
export function getOwnedInterview(id: string, userEmail: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.interviews)
    .where(
      and(
        eq(schema.interviews.id, id),
        eq(schema.interviews.userEmail, userEmail),
      ),
    )
    .get();
}

export function listInterviews(userEmail: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.interviews)
    .where(eq(schema.interviews.userEmail, userEmail))
    .orderBy(desc(schema.interviews.createdAt))
    .limit(20)
    .all();
}

export function getInterviewTurns(interviewId: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.interviewTurns)
    .where(eq(schema.interviewTurns.interviewId, interviewId))
    .orderBy(schema.interviewTurns.ordinal)
    .all();
}

export function appendInterviewQuestion(params: {
  interviewId: string;
  ordinal: number;
  question: string;
  topic: string | null;
}): string {
  const db = getDb();
  const id = randomUUID();
  db.transaction((tx) => {
    tx.insert(schema.interviewTurns)
      .values({
        id,
        interviewId: params.interviewId,
        ordinal: params.ordinal,
        question: params.question,
        topic: params.topic,
        askedAt: new Date().toISOString(),
      })
      .run();
    tx.update(schema.interviews)
      .set({ askedCount: params.ordinal })
      .where(eq(schema.interviews.id, params.interviewId))
      .run();
  });
  return id;
}

export function saveInterviewAnswer(params: {
  turnId: string;
  answerText: string | null;
  answerSource: string;
  answerSeconds: number | null;
  criteriaJson: string | null;
  strength: string | null;
  improvement: string | null;
}): void {
  const db = getDb();
  db.update(schema.interviewTurns)
    .set({
      answerText: params.answerText,
      answerSource: params.answerSource,
      answerSeconds: params.answerSeconds,
      criteriaJson: params.criteriaJson,
      strength: params.strength,
      improvement: params.improvement,
      answeredAt: new Date().toISOString(),
    })
    .where(eq(schema.interviewTurns.id, params.turnId))
    .run();
}

export function completeInterview(id: string, summaryJson: string): void {
  const db = getDb();
  db.update(schema.interviews)
    .set({
      status: "completed",
      summaryJson,
      completedAt: new Date().toISOString(),
    })
    .where(eq(schema.interviews.id, id))
    .run();
}

export function deleteInterview(id: string, userEmail: string): boolean {
  const db = getDb();
  const result = db
    .delete(schema.interviews)
    .where(
      and(
        eq(schema.interviews.id, id),
        eq(schema.interviews.userEmail, userEmail),
      ),
    )
    .run();
  return result.changes > 0;
}


/* -------------------------------------------------------- job applications */

export function createJobApplication(params: {
  userEmail: string;
  company: string;
  positionTitle: string;
  jobUrl: string | null;
  descriptionSource: string;
  postingText: string | null;
  resumeText: string | null;
  resumeFilename: string | null;
  roleBrief: string | null;
  candidateBrief: string | null;
}): string {
  const db = getDb();
  const id = randomUUID();
  const now = new Date().toISOString();
  db.insert(schema.jobApplications)
    .values({ id, ...params, createdAt: now, updatedAt: now })
    .run();
  return id;
}

export function getOwnedApplication(id: string, userEmail: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.jobApplications)
    .where(
      and(
        eq(schema.jobApplications.id, id),
        eq(schema.jobApplications.userEmail, userEmail),
      ),
    )
    .get();
}

export function listApplications(userEmail: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.jobApplications)
    .where(eq(schema.jobApplications.userEmail, userEmail))
    .orderBy(desc(schema.jobApplications.createdAt))
    .limit(20)
    .all();
}

export function updateApplication(
  id: string,
  userEmail: string,
  patch: Partial<{
    company: string;
    positionTitle: string;
    jobUrl: string | null;
    descriptionSource: string;
    postingText: string | null;
    resumeText: string | null;
    resumeFilename: string | null;
    roleBrief: string | null;
    candidateBrief: string | null;
  }>,
): boolean {
  const db = getDb();
  const result = db
    .update(schema.jobApplications)
    .set({ ...patch, updatedAt: new Date().toISOString() })
    .where(
      and(
        eq(schema.jobApplications.id, id),
        eq(schema.jobApplications.userEmail, userEmail),
      ),
    )
    .run();
  return result.changes > 0;
}

/**
 * Releases the stored resume once it is no longer needed for review. The
 * derived briefs stay, so an interview can still be tailored, but the personal
 * document itself does not sit on the server indefinitely (ADR-015).
 */
export function purgeApplicationResume(id: string): void {
  const db = getDb();
  db.update(schema.jobApplications)
    .set({ resumeText: null, resumePurgedAt: new Date().toISOString() })
    .where(eq(schema.jobApplications.id, id))
    .run();
}

export function deleteApplication(id: string, userEmail: string): boolean {
  const db = getDb();
  const result = db
    .delete(schema.jobApplications)
    .where(
      and(
        eq(schema.jobApplications.id, id),
        eq(schema.jobApplications.userEmail, userEmail),
      ),
    )
    .run();
  return result.changes > 0;
}

/* ------------------------------------------------------ tailored documents */

export function upsertTailoredDocument(params: {
  applicationId: string;
  userEmail: string;
  kind: string;
  template: number;
  modelId: string;
  contentJson: string;
  ungroundedJson: string | null;
}): string {
  const db = getDb();
  const now = new Date().toISOString();
  const existing = db
    .select()
    .from(schema.tailoredDocuments)
    .where(
      and(
        eq(schema.tailoredDocuments.applicationId, params.applicationId),
        eq(schema.tailoredDocuments.kind, params.kind),
      ),
    )
    .get();

  if (existing) {
    // Regenerating replaces the draft and clears the review, because the
    // student has not seen this version yet.
    db.update(schema.tailoredDocuments)
      .set({
        template: params.template,
        modelId: params.modelId,
        contentJson: params.contentJson,
        ungroundedJson: params.ungroundedJson,
        reviewedAt: null,
        updatedAt: now,
      })
      .where(eq(schema.tailoredDocuments.id, existing.id))
      .run();
    return existing.id;
  }

  const id = randomUUID();
  db.insert(schema.tailoredDocuments)
    .values({ id, ...params, createdAt: now, updatedAt: now })
    .run();
  return id;
}

export function getOwnedDocument(id: string, userEmail: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.tailoredDocuments)
    .where(
      and(
        eq(schema.tailoredDocuments.id, id),
        eq(schema.tailoredDocuments.userEmail, userEmail),
      ),
    )
    .get();
}

export function listDocumentsForApplication(applicationId: string) {
  const db = getDb();
  return db
    .select()
    .from(schema.tailoredDocuments)
    .where(eq(schema.tailoredDocuments.applicationId, applicationId))
    .all();
}

/** Saves the student's edits. Editing counts as reviewing. */
export function saveDocumentContent(params: {
  id: string;
  userEmail: string;
  contentJson: string;
  markReviewed: boolean;
}): boolean {
  const db = getDb();
  const now = new Date().toISOString();
  const result = db
    .update(schema.tailoredDocuments)
    .set({
      contentJson: params.contentJson,
      updatedAt: now,
      ...(params.markReviewed ? { reviewedAt: now } : {}),
    })
    .where(
      and(
        eq(schema.tailoredDocuments.id, params.id),
        eq(schema.tailoredDocuments.userEmail, params.userEmail),
      ),
    )
    .run();
  return result.changes > 0;
}

// ---------------------------------------------------------------- job scout

/** A posting as the harvest pipeline hands it over, pre-tagging. */
export interface ScoutPostingInput {
  source: "activejobs" | "usajobs";
  externalId: string;
  fingerprint: string;
  title: string;
  company: string;
  locationCity: string | null;
  locationState: string | null;
  remote: boolean;
  category: "fulltime" | "internship" | "federal";
  applyUrl: string;
  description: string;
  postedAt: string | null;
  skillsJson: string;
  /** sponsors | no_sponsorship | unknown (tagging pass, 2026-07-28). */
  visaSponsorship: string;
  taxonomyVersion: number;
}

/**
 * Insert-or-refresh on (source, externalId): a posting seen again keeps its
 * id (so a student's locally saved id stays valid week to week) and gets its
 * lastSeenAt and tags refreshed.
 */
export function upsertScoutPosting(input: ScoutPostingInput): string {
  const db = getScoutDb();
  const now = new Date().toISOString();
  const existing = db
    .select({ id: schema.scoutPostings.id })
    .from(schema.scoutPostings)
    .where(
      and(
        eq(schema.scoutPostings.source, input.source),
        eq(schema.scoutPostings.externalId, input.externalId),
      ),
    )
    .get();
  if (existing) {
    db.update(schema.scoutPostings)
      .set({
        title: input.title,
        company: input.company,
        locationCity: input.locationCity,
        locationState: input.locationState,
        remote: input.remote,
        category: input.category,
        applyUrl: input.applyUrl,
        description: input.description,
        postedAt: input.postedAt,
        lastSeenAt: now,
        skillsJson: input.skillsJson,
        visaSponsorship: input.visaSponsorship,
        taxonomyVersion: input.taxonomyVersion,
        active: true,
      })
      .where(eq(schema.scoutPostings.id, existing.id))
      .run();
    return existing.id;
  }
  const id = randomUUID();
  db.insert(schema.scoutPostings)
    .values({ id, ...input, harvestedAt: now, lastSeenAt: now, active: true })
    .run();
  return id;
}

/** True when any active posting already carries this cross-source fingerprint. */
export function scoutFingerprintExists(fingerprint: string): boolean {
  return Boolean(
    getScoutDb()
      .select({ id: schema.scoutPostings.id })
      .from(schema.scoutPostings)
      .where(
        and(
          eq(schema.scoutPostings.fingerprint, fingerprint),
          eq(schema.scoutPostings.active, true),
        ),
      )
      .get(),
  );
}

export interface ScoutFeedFilters {
  category?: "fulltime" | "internship" | "federal";
  state?: string;
  remote?: boolean;
  limit: number;
  offset: number;
}

/**
 * The feed query. Descriptions are excluded: cards do not need them, and a
 * page of full employer text would be most of a megabyte.
 */
export function listScoutPostings(filters: ScoutFeedFilters) {
  const conditions = [eq(schema.scoutPostings.active, true)];
  if (filters.category)
    conditions.push(eq(schema.scoutPostings.category, filters.category));
  if (filters.state)
    conditions.push(eq(schema.scoutPostings.locationState, filters.state));
  if (filters.remote !== undefined)
    conditions.push(eq(schema.scoutPostings.remote, filters.remote));
  return getScoutDb()
    .select({
      id: schema.scoutPostings.id,
      source: schema.scoutPostings.source,
      title: schema.scoutPostings.title,
      company: schema.scoutPostings.company,
      locationCity: schema.scoutPostings.locationCity,
      locationState: schema.scoutPostings.locationState,
      remote: schema.scoutPostings.remote,
      category: schema.scoutPostings.category,
      applyUrl: schema.scoutPostings.applyUrl,
      postedAt: schema.scoutPostings.postedAt,
      skillsJson: schema.scoutPostings.skillsJson,
      visaSponsorship: schema.scoutPostings.visaSponsorship,
      taxonomyVersion: schema.scoutPostings.taxonomyVersion,
    })
    .from(schema.scoutPostings)
    .where(and(...conditions))
    .orderBy(desc(schema.scoutPostings.postedAt))
    .limit(filters.limit)
    .offset(filters.offset)
    .all();
}

export function countScoutPostings(): number {
  const row = getScoutDb()
    .select({ n: sql<number>`count(*)` })
    .from(schema.scoutPostings)
    .where(eq(schema.scoutPostings.active, true))
    .get();
  return row?.n ?? 0;
}

/** Full row, description included, for the expanded card and the handoff. */
export function getScoutPosting(id: string) {
  return (
    getScoutDb()
      .select()
      .from(schema.scoutPostings)
      .where(eq(schema.scoutPostings.id, id))
      .get() ?? null
  );
}

/**
 * Retirement per design §4.3: postings unseen since the cutoff go inactive;
 * rows older than the purge cutoff are deleted outright.
 */
export function retireScoutPostings(params: {
  unseenSinceIso: string;
  purgeBeforeIso: string;
  /** Postings whose own post date is older than this go inactive too
   * (user decision 2026-07-28: nothing older than a month stays listed). */
  postedBeforeIso: string;
}): { deactivated: number; purged: number } {
  const db = getScoutDb();
  const deactivated = db
    .update(schema.scoutPostings)
    .set({ active: false })
    .where(
      and(
        eq(schema.scoutPostings.active, true),
        sql`(${schema.scoutPostings.lastSeenAt} < ${params.unseenSinceIso}
             or (${schema.scoutPostings.postedAt} is not null
                 and ${schema.scoutPostings.postedAt} < ${params.postedBeforeIso}))`,
      ),
    )
    .run().changes;
  const purged = db
    .delete(schema.scoutPostings)
    .where(lt(schema.scoutPostings.lastSeenAt, params.purgeBeforeIso))
    .run().changes;
  return { deactivated, purged };
}

export function createScoutRun(trigger: "schedule" | "manual"): string {
  const id = randomUUID();
  getScoutDb()
    .insert(schema.scoutRuns)
    .values({
      id,
      startedAt: new Date().toISOString(),
      status: "running",
      trigger,
    })
    .run();
  return id;
}

export function finishScoutRun(
  id: string,
  patch: {
    status: "completed" | "partial" | "failed";
    activejobsRequests?: number;
    activejobsFound?: number;
    usajobsRequests?: number;
    usajobsFound?: number;
    dedupedCount?: number;
    taggedCount?: number;
    costUsd?: number;
    sourceErrorsJson?: string;
    error?: string | null;
  },
): void {
  getScoutDb()
    .update(schema.scoutRuns)
    .set({ finishedAt: new Date().toISOString(), ...patch })
    .where(eq(schema.scoutRuns.id, id))
    .run();
}

/** The scheduler's persisted memory: when did a harvest last succeed. */
export function latestSuccessfulScoutRun() {
  return (
    getScoutDb()
      .select()
      .from(schema.scoutRuns)
      .where(
        sql`${schema.scoutRuns.status} in ('completed', 'partial')`,
      )
      .orderBy(desc(schema.scoutRuns.startedAt))
      .limit(1)
      .get() ?? null
  );
}

/**
 * Guards the manual trigger: only one harvest at a time. A "running" row
 * older than two hours is treated as dead, not in progress: a killed
 * process never writes finishScoutRun, and without this cutoff one crash
 * would block every future harvest (guard added 2026-07-28).
 */
export function scoutRunInProgress(): boolean {
  const staleCutoff = new Date(Date.now() - 2 * 3_600_000).toISOString();
  return Boolean(
    getScoutDb()
      .select({ id: schema.scoutRuns.id })
      .from(schema.scoutRuns)
      .where(
        and(
          eq(schema.scoutRuns.status, "running"),
          sql`${schema.scoutRuns.startedAt} > ${staleCutoff}`,
        ),
      )
      .get(),
  );
}

/**
 * The whole active feed in one shot (card fields only, no descriptions).
 * Exists because client-side matching needs every posting's tags anyway,
 * and one compact response beats a pagination loop: at 2,000 postings this
 * is roughly 600 KB before gzip (user scale question, 2026-07-29).
 */
export function listAllScoutPostings() {
  return getScoutDb()
    .select({
      id: schema.scoutPostings.id,
      source: schema.scoutPostings.source,
      title: schema.scoutPostings.title,
      company: schema.scoutPostings.company,
      locationCity: schema.scoutPostings.locationCity,
      locationState: schema.scoutPostings.locationState,
      remote: schema.scoutPostings.remote,
      category: schema.scoutPostings.category,
      applyUrl: schema.scoutPostings.applyUrl,
      postedAt: schema.scoutPostings.postedAt,
      skillsJson: schema.scoutPostings.skillsJson,
      visaSponsorship: schema.scoutPostings.visaSponsorship,
      taxonomyVersion: schema.scoutPostings.taxonomyVersion,
    })
    .from(schema.scoutPostings)
    .where(eq(schema.scoutPostings.active, true))
    .orderBy(desc(schema.scoutPostings.postedAt))
    .all();
}
