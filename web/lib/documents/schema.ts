import { z } from "zod";

/**
 * The structured shape of a tailored document.
 *
 * Stored as data rather than prose so the print page and the .docx render from
 * one source and cannot drift apart, and so the student can edit one bullet
 * without the model rewriting everything around it.
 */

export const resumeBulletSchema = z.object({
  text: z.string().describe("One impact-statement bullet, at most two lines."),
  /**
   * The line from the student's own resume this came from. The whole guard
   * rail rests on this: a bullet with no source is a bullet the model invented.
   */
  sourceLine: z
    .string()
    .nullable()
    .describe(
      "The exact line from the student's resume this bullet is based on. Null if there is none.",
    ),
});

export const resumeEntrySchema = z.object({
  organization: z.string(),
  title: z.string(),
  location: z.string().nullable(),
  dates: z.string().nullable(),
  bullets: z.array(resumeBulletSchema),
});

export const resumeSectionSchema = z.object({
  heading: z.string(),
  entries: z.array(resumeEntrySchema),
});

export const resumeContentSchema = z.object({
  name: z.string(),
  contact: z.object({
    email: z.string().nullable(),
    phone: z.string().nullable(),
    linkedin: z.string().nullable(),
  }),
  education: z.object({
    school: z.string(),
    location: z.string(),
    degree: z.string().nullable(),
    majorMinor: z.string().nullable(),
    graduation: z.string().nullable(),
    gpa: z.string().nullable(),
    honors: z.array(z.string()),
  }),
  sections: z.array(resumeSectionSchema),
  skills: z.array(z.string()),
});

export type ResumeContent = z.infer<typeof resumeContentSchema>;
export type ResumeBullet = z.infer<typeof resumeBulletSchema>;

export const coverLetterContentSchema = z.object({
  name: z.string(),
  contact: z.object({
    email: z.string().nullable(),
    phone: z.string().nullable(),
    linkedin: z.string().nullable(),
  }),
  date: z.string().nullable(),
  recipient: z.object({
    name: z.string().nullable(),
    company: z.string(),
    address: z.string().nullable(),
  }),
  salutation: z.string(),
  /** Introduction, three matched body paragraphs, then the summary. */
  paragraphs: z.array(
    z.object({
      text: z.string(),
      /** Which requirement from the posting this paragraph answers. */
      addresses: z.string().nullable(),
      sourceLine: z.string().nullable(),
    }),
  ),
  closing: z.string(),
});

export type CoverLetterContent = z.infer<typeof coverLetterContentSchema>;

/**
 * Model-facing schemas are deliberately looser than these: a single malformed
 * field must not lose the whole document, and the catalog spans providers whose
 * JSON Schema support varies. The strict shapes above are applied afterwards.
 */
export const generatedResumeSchema = z.object({
  education: z.object({
    degree: z.string().nullable(),
    majorMinor: z.string().nullable(),
    graduation: z.string().nullable(),
    gpa: z.string().nullable(),
    honors: z.array(z.string()),
  }),
  sections: z.array(
    z.object({
      heading: z.string(),
      entries: z.array(
        z.object({
          organization: z.string(),
          title: z.string(),
          location: z.string().nullable(),
          dates: z.string().nullable(),
          bullets: z.array(
            z.object({
              text: z.string(),
              sourceLine: z.string().nullable(),
            }),
          ),
        }),
      ),
    }),
  ),
  skills: z.array(z.string()),
});

export const generatedCoverLetterSchema = z.object({
  salutation: z.string(),
  paragraphs: z.array(
    z.object({
      text: z.string(),
      addresses: z.string().nullable(),
      sourceLine: z.string().nullable(),
    }),
  ),
  closing: z.string(),
});
