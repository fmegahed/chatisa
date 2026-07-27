import { z } from "zod";

/**
 * Two schemas on purpose.
 *
 * The model-facing schema is permissive: constraints live in field
 * descriptions, every property is present (strict structured output requires
 * it) and nullable rather than optional, and there are no unions, because
 * several providers emit poor JSON Schema for them. A single strict schema
 * would let one malformed question throw away a whole batch of good ones.
 *
 * The strict schema is then applied per question on the server, so bad
 * questions are dropped individually.
 */

export const QUESTION_TYPES = [
  "multiple_choice",
  "short_answer",
  "code_understanding",
  "data_analysis",
] as const;

export type QuestionType = (typeof QUESTION_TYPES)[number];

export const BLOOM_LEVELS = [
  "remember",
  "understand",
  "apply",
  "analyze",
  "evaluate",
  "create",
] as const;

export const generatedQuestionSchema = z.object({
  type: z.enum(QUESTION_TYPES),
  stem: z
    .string()
    .describe("The question as the student reads it. Self contained."),
  options: z
    .array(z.string())
    .nullable()
    .describe(
      "Exactly 4 answer options for multiple_choice. Null for every other type.",
    ),
  correctIndex: z
    .number()
    .int()
    .nullable()
    .describe(
      "Zero-based index of the one correct option. Null for every other type.",
    ),
  modelAnswer: z
    .string()
    .describe("A complete correct answer, two to six sentences."),
  rubric: z
    .array(
      z.object({
        criterion: z
          .string()
          .describe("One observable thing a correct answer must contain."),
        points: z.number().int().describe("Points for this criterion, 1 to 10."),
      }),
    )
    .describe("Two to four criteria whose points total 10."),
  explanation: z
    .string()
    .describe("Why the answer is right, written for a student who got it wrong."),
  topic: z
    .string()
    .describe("A two to five word topic label, reused across related questions."),
  bloom: z.enum(BLOOM_LEVELS),
  sourceQuote: z
    .string()
    .describe(
      "40 to 400 characters copied VERBATIM from the excerpt that make this question answerable.",
    ),
  sourcePage: z
    .number()
    .int()
    .describe("The page number labelled on the excerpt containing the quote."),
});

export type GeneratedQuestion = z.infer<typeof generatedQuestionSchema>;

export const examGenerationSchema = z.object({
  questions: z.array(generatedQuestionSchema),
});

/** Applied per question after generation. Failures drop one question, not all. */
export const strictQuestionSchema = generatedQuestionSchema
  .extend({
    stem: z.string().min(15).max(1200),
    modelAnswer: z.string().min(10).max(2000),
    explanation: z.string().min(10).max(1500),
    topic: z.string().min(2).max(60),
    sourceQuote: z.string().min(30).max(400),
    sourcePage: z.number().int().min(1),
    rubric: z
      .array(
        z.object({
          criterion: z.string().min(3).max(200),
          points: z.number().int().min(1).max(10),
        }),
      )
      .min(1)
      .max(5),
  })
  .superRefine((q, ctx) => {
    const mcq = q.type === "multiple_choice";
    if (mcq && (q.options?.length !== 4 || q.correctIndex === null)) {
      ctx.addIssue({ code: "custom", message: "mcq_shape", path: ["options"] });
    }
    if (
      mcq &&
      q.correctIndex !== null &&
      (q.correctIndex < 0 || q.correctIndex > 3)
    ) {
      ctx.addIssue({
        code: "custom",
        message: "mcq_index",
        path: ["correctIndex"],
      });
    }
    if (!mcq && q.options !== null) {
      ctx.addIssue({ code: "custom", message: "open_shape", path: ["options"] });
    }
  });

/** Grading of one written answer. No total score: the server does the sums. */
export const gradeSchema = z.object({
  criteria: z.array(
    z.object({
      criterion: z.string(),
      met: z.enum(["yes", "partial", "no"]),
      justification: z.string(),
    }),
  ),
  feedback: z.string(),
});

export type GradeResult = z.infer<typeof gradeSchema>;
