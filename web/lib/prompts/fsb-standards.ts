/**
 * Farmer School of Business resume and cover letter standards.
 *
 * Encoded from the six PDFs in `resume_and_cover_letters_examples_and_guidelines/`.
 * The three "Standards" are three typographic variants of one content standard,
 * not three different standards, so the rules live here once and `Template`
 * carries only what actually differs between them.
 *
 * Quoted rules are the documents' own wording. Where the source is silent this
 * file says so rather than inventing a rule: several things people assume are
 * specified (page limit, margins, font size, references) are not stated
 * anywhere in the FSB materials.
 */

export type TemplateId = 1 | 2 | 3;

export interface TemplateStyle {
  id: TemplateId;
  label: string;
  /** Standard 1 omits the school; 2 and 3 include it. */
  schoolLine: string;
  /** Whether a horizontal rule sits under each section header. */
  sectionRules: boolean;
  headingFont: "serif" | "sans";
  bodyFont: "serif" | "sans";
  /**
   * Standards 1 and 2 lead an experience entry with the organization; Standard
   * 3 leads with the position title and folds city and state onto the org line.
   */
  entryOrder: "organization-first" | "title-first";
}

export const TEMPLATES: Record<TemplateId, TemplateStyle> = {
  1: {
    id: 1,
    label: "Standard 1",
    schoolLine: "Miami University",
    sectionRules: true,
    headingFont: "sans",
    bodyFont: "serif",
    entryOrder: "organization-first",
  },
  2: {
    id: 2,
    label: "Standard 2",
    schoolLine: "Miami University, Farmer School of Business",
    sectionRules: true,
    headingFont: "serif",
    bodyFont: "serif",
    entryOrder: "organization-first",
  },
  3: {
    id: 3,
    label: "Standard 3",
    schoolLine: "Miami University, Farmer School of Business",
    sectionRules: false,
    headingFont: "sans",
    bodyFont: "sans",
    entryOrder: "title-first",
  },
};

export const RESUME_SECTIONS = [
  "EDUCATION",
  "RELEVANT EXPERIENCE",
  "COURSES & PROJECT EXPERIENCE",
  "ACTIVITIES",
  "SKILLS / CERTIFICATIONS",
] as const;

export type ResumeSection = (typeof RESUME_SECTIONS)[number];

/** The ACTIVITIES header may be renamed, per the templates themselves. */
export const ACTIVITIES_ALTERNATIVES = [
  "ACTIVITIES",
  "EXTRACURRICULARS",
  "INVOLVEMENT",
  "PROFESSIONAL ASSOCIATIONS",
];

/** Bullet rules, in the documents' own words where they are prescriptive. */
export const BULLET_RULES = [
  "Start each bullet with a strong action verb from the approved list.",
  "Formulate each bullet as an impact statement: WHAT you did, HOW you did it, and WHY you did it or HOW it added value.",
  "Showcase the lead impact or result in the very first bullet of an entry.",
  "Include 2 to 4 bullet points for each entry.",
  "Bullets are no more than two lines long, and avoid leaving one or two orphan words on the second line.",
  "Write in past tense unless the position or class is current, in which case present tense.",
  "Include quantification wherever possible: numbers, dollar figures and percentages.",
  "Match language from the job description, including how soft skills are described.",
  "Use only one action verb per section; do not repeat a verb within a section.",
] as const;

/** Explicit prohibitions. Each is stated in the source materials. */
export const BULLET_PROHIBITIONS = [
  'Do not start a bullet with "Responsible for".',
  'Do not use "other duties as assigned".',
  "Do not list routine duties.",
  'Do not end a bullet with "etc.".',
  'Do not use personal pronouns such as "I", "my", "our" or "we"; the resume is third person.',
] as const;

/** Education block conventions specific to Miami and FSB. */
export const EDUCATION_RULES = [
  "Location is Oxford, OH, right aligned opposite the school name.",
  'Graduation is phrased "Expected Graduation 20XX": the year only, no month.',
  "Include GPA only if it is above 3.0. Omit it entirely otherwise.",
  "Major and minor share one line, separated by a slash.",
  "Remove high school after the first year.",
  "Study abroad or prior universities get their own line with their own dates.",
  'Refer to a course by its title, never its course number (for example "Business Analytics", not "ISA 401").',
] as const;

export const CONTACT_LINE_FORMAT = "Email | Phone | LinkedIn";

/**
 * Action verbs, grouped by the transferable-skill categories the FSB handout
 * uses. The handout's own framing: the category headings are the transferable
 * skills, so this is one list serving both purposes.
 */
export const ACTION_VERBS: Record<string, string[]> = {
  Creativity: ["Act","Compose","Conceptualize","Create","Customize","Design","Develop","Direct","Display","Draw","Entertain","Establish","Fashion","Formulate","Generate","Illustrate","Imagine","Improve","Initiate","Innovate","Introduce","Invent","Modify","Originate","Perform","Revise","Revitalize","Shape","Visualize"],
  "Teaching Skills": ["Advise","Assess","Coach","Communicate","Develop","Educate","Evaluate","Explain","Facilitate","Guide","Influence","Initiate","Inspire","Instruct","Monitor","Persuade","Provide","Show","Teach","Tutor"],
  "Analytical/Financial Skills": ["Adjust","Allocate","Analyze","Appraise","Assess","Balance","Budget","Calculate","Compare","Compute","Conserve","Estimate","Evaluate","Examine","Forecast","Inspect","Interpret","Investigate","Manage","Measure","Net","Plan","Prepare","Program","Project","Quantify","Reconcile","Record","Reduce","Research","Retrieve","Review","Survey"],
  "Teamwork/Team-building Skills": ["Assist","Collaborate","Contribute","Cooperate","Coordinate","Help","Involve","Participate","Share","Support","Uphold"],
  "Organizational Skills": ["Arrange","Categorize","Chart","Collect","Compile","Coordinate","Correct","Distribute","Execute","File","Follow-through","Log","Maintain","Map out","Monitor","Obtain","Operate","Order","Organize","Plan","Prepare","Prioritize","Process","Provide","Purchase","Record","Review","Schedule","Submit","Supply","Systematize","Update","Verify"],
  "Adaptability/Flexibility": ["Acclimate","Adapt","Adjust","Alter","Anticipate","Change","Comply","Evolve","Learn","Modify","Revise","Rework"],
  "Communication/Interpersonal Skills": ["Address","Arbitrate","Articulate","Author","Clarify","Communicate","Compose","Condense","Connect","Consult","Contact","Convey","Convince","Correspond","Debate","Define","Direct","Discuss","Draft","Edit","Explain","Express","Influence","Interact","Interpret","Interview","Lecture","Listen","Mediate","Moderate","Motivate","Negotiate","Observe","Outline","Persuade","Present","Propose","Reason","Reconcile","Report","Resolve","Respond","Speak","Specify","Suggest","Summarize","Translate","Write"],
  "Helping Skills": ["Administer","Advocate","Aide","Alleviate","Answer","Arrange","Assess","Assist","Attend to","Benefit","Clarify","Coach","Collaborate","Contribute","Cooperate","Counsel","Demonstrate","Diagnose","Educate","Encourage","Ensure","Expedite","Facilitate","Further","Give","Guide","Help","Intervene","Listen","Motivate","Prevent","Provide","Refer","Relieve","Represent","Resolve","Serve","Support","Treat","Volunteer"],
  Detail: ["Arrange","Categorize","Classify","Compare","Examine","Inspect","Organize","Process","Record","Sort","Systematize"],
  "Leadership/Management Skills": ["Administer","Appoint","Approve","Assign","Attain","Authorize","Chair","Contract","Control","Coordinate","Decide","Delegate","Develop","Direct","Eliminate","Emphasize","Enforce","Enhance","Establish","Evaluate","Execute","Facilitate","Handle","Hire","Improve","Incorporate","Increase","Initiate","Lead","Manage","Motivate","Multi-task","Navigate","Organize","Oversee","Plan","Preside","Prioritize","Produce","Recommend","Restore","Review","Schedule","Secure","Select","Streamline","Strengthen","Supervise","Terminate"],
  "Research Skills": ["Analyze","Clarify","Collect","Compare","Conduct","Critique","Detect","Evaluate","Find","Highlight","Persuade","Propose","Prove","Simulate","Quantify","Stimulate","Study","Test","Train","Transmit"],
  "PR/Advertising": ["Advertise","Communicate","Contact","Correspond","Develop","Elicit","Enlist","Influence","Involve","Market","Persuade","Present","Promote","Propose","Publicize","Recruit","Sell","Show","Solicit"],
  Quantifying: ["Cut","Decrease","Eliminate","Increase","Lessen","Lower","Maximize","Minimize","Raise","Reduce"],
  "Improvement/Achievement": ["Accelerate","Accomplish","Achieve","Advance","Boost","Change","Correct","Enhance","Expedite","Fix","Further","Improve","Overhaul","Rectify","Repair","Resolve","Restore","Revamp","Revitalize","Save","Secure","Solve","Streamline","Strengthen","Update","Upgrade"],
  Initiative: ["Conceptualize","Create","Design","Develop","Devise","Establish","Found","Generate","Implement","Innovate","Institute","Introduce","Launch","Lead","Motivate","Originate","Pioneer","Produce","Propose","Set up","Spearhead","Start"],
  "Technical Skills": ["Apply","Assemble","Build","Calculate","Compute","Conserve","Construct","Convert","Debug","Design","Determine","Develop","Engineer","Fabricate","Fortify","Install","Maintain","Operate","Overhaul","Print","Program","Rectify","Regulate","Remodel","Repair","Replace","Restore","Solve","Specialize","Standardize","Study","Troubleshoot","Upgrade","Utilize"],
};

/** Every approved verb, lowercased, for checking a generated bullet. */
export const ALL_ACTION_VERBS: ReadonlySet<string> = new Set(
  Object.values(ACTION_VERBS)
    .flat()
    .map((v) => v.toLowerCase()),
);

/** Cover letter structure, paragraph by paragraph. */
export const COVER_LETTER_STRUCTURE = [
  {
    name: "Introduction",
    rules: [
      "Give a brief overview of your credentials. It is not necessary to state your name.",
      "Explain why you want to work for that organization, highlighting something specific to them.",
      "Name the position and say how you heard about it. Miami students commonly cite Handshake.",
      "Mention any personal connection to the organization.",
    ],
  },
  {
    name: "Body",
    rules: [
      "Pick three elements of the job description most relevant to your qualifications.",
      "For each, give a specific accomplishment that shows you can meet that need.",
      "Prefer a few in-depth examples over a list.",
      "Quantify and show results wherever possible.",
    ],
  },
  {
    name: "Summary",
    rules: [
      "Reference your attached resume.",
      "Thank them for their time.",
      'If you say you will follow up, do not give a specific date; write "in two weeks" or "the week of". Omit this entirely if the posting says not to make contact.',
    ],
  },
] as const;

export const COVER_LETTER_PROHIBITIONS = [
  'Do not write "To whom it may concern". Use "Dear Hiring Manager" or "Dear Search Committee Chair" when no name is available.',
  'Do not use an informal greeting such as "Hey Ms. Grams".',
  'Do not close with "Cheers" or "Talk with you soon". Use Sincerely, Regards, Warm regards or Best regards.',
] as const;

/**
 * The cover letter is first person, the opposite of the resume. Getting this
 * backwards is the most likely voice mistake, so it is stated explicitly rather
 * than left implicit in the two rule sets.
 */
export const VOICE = {
  resume: "third person, no personal pronouns",
  coverLetter: "first person",
} as const;

/**
 * Things people assume are specified and are not, anywhere in the FSB
 * materials. Recorded so nobody later invents a rule and attributes it to the
 * school: page limit, margins, font name or size, line spacing, guidance on
 * photos or personal identifiers, references, and any ATS advice. The templates
 * are all one page in practice, which is why the generator targets one page,
 * but that is an observation rather than a stated rule.
 */
export const NOT_SPECIFIED = [
  "page limit",
  "margins",
  "font name or size",
  "line spacing",
  "photos or personal identifiers",
  "references",
  "applicant tracking system advice",
] as const;

/** The rules block shared by both document prompts. */
export function resumeRulesForPrompt(template: TemplateId): string {
  const style = TEMPLATES[template];
  return `FARMER SCHOOL OF BUSINESS RESUME STANDARD

Voice: ${VOICE.resume}.
School line: ${style.schoolLine}, with Oxford, OH right aligned opposite it.
Sections, in this order: ${RESUME_SECTIONS.join(", ")}.

Bullet rules:
${BULLET_RULES.map((r) => `- ${r}`).join("\n")}

Never:
${BULLET_PROHIBITIONS.map((r) => `- ${r}`).join("\n")}

Education:
${EDUCATION_RULES.map((r) => `- ${r}`).join("\n")}

Contact line: ${CONTACT_LINE_FORMAT}
The whole resume fits on one page.`;
}

export function coverLetterRulesForPrompt(): string {
  return `FARMER SCHOOL OF BUSINESS COVER LETTER STANDARD

Voice: ${VOICE.coverLetter}. This is the opposite of the resume, which is third person.

Structure:
${COVER_LETTER_STRUCTURE.map(
  (p) => `${p.name}:\n${p.rules.map((r) => `  - ${r}`).join("\n")}`,
).join("\n")}

Never:
${COVER_LETTER_PROHIBITIONS.map((r) => `- ${r}`).join("\n")}

One page, roughly 250 to 350 words.`;
}

/**
 * Page and type metrics, read out of the Word originals the user supplied
 * rather than inferred from the rendered PDFs.
 *
 * Twips are the unit Word uses: 1440 to the inch. Font sizes here are points;
 * the docx library takes half-points, so they are doubled at the point of use.
 */
export const LAYOUT = {
  /** US Letter. */
  pageWidthTwips: 12_240,
  pageHeightTwips: 15_840,
  /** 0.5 inch on all four sides, which is what the FSB templates use and what
   * the user confirmed is acceptable for an undergraduate one-pager. */
  marginTwips: 720,
  /** Usable width, and therefore where a right-aligned tab stop belongs. */
  get contentWidthTwips() {
    return this.pageWidthTwips - this.marginTwips * 2;
  },
  nameSizePt: 14,
  contactSizePt: 11,
  headingSizePt: 11,
  bodySizePt: 10,
  headingFont: "Arial",
  /** Standard 1 sets body in a serif face; Standard 3 is Arial throughout. */
  resumeBodyFont: "Times New Roman",
  coverLetterFont: "Arial",
} as const;

/**
 * Undergraduate resumes are expected to be a single page (user instruction,
 * 2026-07-21). This is a hard target for generation, not a suggestion: an
 * over-long resume is the most common way a student's application is dismissed
 * before it is read.
 */
export const RESUME_PAGE_TARGET = 1;

/**
 * Measured from the finished sample letter on page 3 of the FSB cover letter
 * document: four body paragraphs totalling 205 words, running 34, 52, 58 and
 * 61 words. Generation aims at this shape rather than at a vague "one page",
 * because the sample is the actual artifact the school hands out.
 */
export const COVER_LETTER_SHAPE = {
  bodyParagraphs: 4,
  targetWords: 205,
  maxWords: 320,
  /** The sample uses a colon, as does the other worked example. The annotated
   * template shows a comma; the finished letters are the better guide. */
  salutationPunctuation: ":",
} as const;
