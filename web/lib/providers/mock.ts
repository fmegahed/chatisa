import { simulateReadableStream } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import type {
  LanguageModelV4CallOptions,
  LanguageModelV4StreamPart,
} from "@ai-sdk/provider";

/**
 * Deterministic stand-in model for automated tests, enabled only by
 * CHATISA_MOCK_LLM=1. It covers both call styles the app uses: streaming chat
 * (doStream) and structured generation (doGenerate), so tests never call a
 * provider or spend sponsored budget.
 *
 * CHATISA_MOCK_LLM_MODE selects a behavior for failure-path tests:
 *   ok        (default) valid, well-formed output
 *   invalid   output that does not satisfy the requested schema
 *   illegible transcription reporting an unreadable page
 */
const CHUNKS = [
  "Here is how to read a CSV in both languages, plus a SQL check.\n\n",
  "```r\nif(require(readr)==FALSE) install.packages('readr')\n",
  "sales <- readr::read_csv('sales.csv')\n```\n\n",
  "```python\nimport pandas as pd\n\nsales = (\n    pd.read_csv('sales.csv')\n)\n```\n\n",
  // A SQL block, the one language runnable in the browser today, so tests can
  // assert the Run button appears on runnable blocks and not on the R/Python
  // ones above.
  "```sql\nSELECT 1 AS n;\n```\n\n",
  "What does your dataset look like?",
];

const USAGE = {
  inputTokens: {
    total: 120,
    noCache: 120,
    cacheRead: undefined,
    cacheWrite: undefined,
  },
  outputTokens: { total: 80, text: 80, reasoning: undefined },
};

function textParts(chunks: string[]): LanguageModelV4StreamPart[] {
  return [
    { type: "text-start", id: "0" },
    ...chunks.map(
      (delta): LanguageModelV4StreamPart => ({
        type: "text-delta",
        id: "0",
        delta,
      }),
    ),
    { type: "text-end", id: "0" },
    {
      type: "finish",
      finishReason: { unified: "stop", raw: "stop" },
      usage: USAGE,
    },
  ];
}

/** Index of the last user message in the prompt (-1 when there is none). */
function lastUserIndex(options: LanguageModelV4CallOptions): number {
  const prompt = options.prompt ?? [];
  for (let i = prompt.length - 1; i >= 0; i--) {
    if ((prompt[i] as { role?: string }).role === "user") return i;
  }
  return -1;
}

/** The text of the LAST USER message only. Trigger phrases must match what
 * the student typed, never the system prompt (which legitimately mentions
 * attachments, datasets, and Miami style) or earlier turns. */
function lastUserText(options: LanguageModelV4CallOptions): string {
  const index = lastUserIndex(options);
  if (index < 0) return "";
  const content = (options.prompt?.[index] as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (part as { text?: string }).text ?? "")
    .join("\n");
}

/** True when the LAST USER message carries a file part with the given media
 * prefix (a natively attached image or PDF on the current turn). */
function lastUserHasFilePart(
  options: LanguageModelV4CallOptions,
  mediaPrefix: string,
): boolean {
  const index = lastUserIndex(options);
  if (index < 0) return false;
  const content = (options.prompt?.[index] as { content?: unknown }).content;
  if (!Array.isArray(content)) return false;
  return content.some((part) => {
    const p = part as { type?: string; mediaType?: string };
    return p.type === "file" && p.mediaType?.startsWith(mediaPrefix) === true;
  });
}

/** The last tool result belonging to the CURRENT turn (after the last user
 * message), stringified, or null. Results from earlier turns stay in the
 * history and must not re-trigger the acknowledgement branch. */
function lastToolResult(options: LanguageModelV4CallOptions): string | null {
  const prompt = options.prompt ?? [];
  const from = lastUserIndex(options) + 1;
  let found: string | null = null;
  for (const message of prompt.slice(from)) {
    const { role, content } = message as { role?: string; content?: unknown };
    if (role !== "tool" || !Array.isArray(content)) continue;
    for (const part of content) {
      const p = part as { type?: string; output?: unknown; result?: unknown };
      if (p.type === "tool-result") {
        found = JSON.stringify(p.output ?? p.result ?? null);
      }
    }
  }
  return found;
}

/** Unique-enough tool call ids: the Ask Anything client dedupes executions by
 * id, so two scripted calls in one chat must never collide. */
let mockToolCallCounter = 0;

function toolCall(
  toolName: string,
  input: object,
): LanguageModelV4StreamPart[] {
  mockToolCallCounter += 1;
  return [
    {
      type: "tool-call",
      toolCallId: `mock-tool-${mockToolCallCounter}`,
      toolName,
      input: JSON.stringify(input),
    },
    {
      type: "finish",
      finishReason: { unified: "tool-calls", raw: "tool_calls" },
      usage: USAGE,
    },
  ];
}

/**
 * Streaming behavior, scripted so the Ask Anything loops are testable end to
 * end without a provider:
 * 1. The prompt already carries a tool result: acknowledge it in text (with an
 *    excerpt), the loop's terminal turn.
 * 2. Scripted tool calls by trigger phrase: "describe the dataset" runs
 *    print(<var>.shape) on the announced DataFrame (real Pyodide), "find
 *    papers" calls search_papers (server-executed fixtures), "miami style"
 *    calls get_miami_style, "use python" runs print(6 * 7).
 * 3. Attachment acknowledgements: FILE_ACK echoes an attached block's start,
 *    PDF_ACK and IMAGE_ACK confirm native file parts arrived.
 * 4. Anything else: the canned CSV answer the chat tests rely on.
 */
function mockStreamParts(
  options: LanguageModelV4CallOptions,
): LanguageModelV4StreamPart[] {
  const toolResult = lastToolResult(options);
  if (toolResult) {
    return textParts([
      "The tool run finished. ",
      `RESULT_ACK ${toolResult.slice(0, 200)}`,
    ]);
  }
  const userText = lastUserText(options);
  const hasTools = (options.tools ?? []).length > 0;

  // A snippet needing a package that CANNOT exist in the browser, so the Run
  // button gate (2026-07-26) has something deterministic to act on. statsforecast
  // is in KNOWN_UNAVAILABLE_PYTHON because it needs compiling; the default canned
  // answer deliberately uses readr and pandas, which are both available, so the
  // tests asserting Run buttons appear are unaffected.
  if (/statsforecast/i.test(userText)) {
    return textParts([
      "Here is a forecast with statsforecast.\n\n",
      "```python\nimport pandas as pd\nfrom statsforecast import StatsForecast\n",
      "sf = StatsForecast(models=[], freq='M')\n```\n\n",
      "Run it locally, since it needs compiled dependencies.",
    ]);
  }
  if (hasTools && /power\s?point|slide deck|presentation/i.test(userText)) {
    // Hosted execution (slice E): the provider runs the tool inside the same
    // response, so the call, its result (with a created file), and the
    // model's continuation all ride in one stream. The tool input arrives as
    // STREAMED DELTAS, exactly like a real provider sends it; a bug that only
    // shows under input streaming must be reproducible here.
    mockToolCallCounter += 1;
    const hostedInput = JSON.stringify({
      type: "programmatic-tool-call",
      code:
        "from pptx import Presentation\nprs = Presentation('miami_template_by_fadel_megahed.pptx')\n" +
        "# build slides from the template exemplars\n".repeat(20) +
        "prs.save('miami-deck.pptx')",
    });
    const inputDeltas: LanguageModelV4StreamPart[] = [];
    for (let i = 0; i < hostedInput.length; i += 4) {
      inputDeltas.push({
        type: "tool-input-delta",
        id: `mock-hosted-${mockToolCallCounter}`,
        delta: hostedInput.slice(i, i + 4),
      });
    }
    return [
      {
        type: "tool-input-start",
        id: `mock-hosted-${mockToolCallCounter}`,
        toolName: "code_execution",
        providerExecuted: true,
      },
      ...inputDeltas,
      {
        type: "tool-input-end",
        id: `mock-hosted-${mockToolCallCounter}`,
      },
      {
        type: "tool-call",
        toolCallId: `mock-hosted-${mockToolCallCounter}`,
        toolName: "code_execution",
        providerExecuted: true,
        input: hostedInput,
      },
      {
        type: "tool-result",
        toolCallId: `mock-hosted-${mockToolCallCounter}`,
        toolName: "code_execution",
        result: {
          type: "code_execution_result",
          stdout: "Saved miami-deck.pptx built from the Miami template.",
          stderr: "",
          return_code: 0,
          content: [
            { type: "code_execution_output", file_id: "file_mockdeck1" },
          ],
        },
      },
      ...textParts([
        "Your deck is ready, built from the Miami template. ",
        "DECK_ACK It ran on Anthropic's servers; download it under the card above.",
      ]),
    ];
  }
  if (hasTools && /describe the dataset/i.test(userText)) {
    const varName = /DataFrame `([a-z0-9_]+)`/.exec(userText)?.[1] ?? "df";
    return toolCall("run_python", { code: `print(${varName}.shape)` });
  }
  if (hasTools && /find papers|search the literature/i.test(userText)) {
    return toolCall("search_papers", { query: "mock topic", limit: 3 });
  }
  if (hasTools && /miami style/i.test(userText)) {
    return toolCall("get_miami_style", { kind: "tikz" });
  }
  if (hasTools && /scrape the fsb/i.test(userText)) {
    // The live-gated ask e2e: the scripted code is exactly what a real model
    // would write, and it executes on the real worker (requests through the
    // py-proxy, bs4 with the auto-loaded lxml parser).
    return toolCall("run_python", {
      code: [
        "import requests",
        "from bs4 import BeautifulSoup",
        "r = requests.get('https://miamioh.edu/fsb/directory/?up=/query/all/all/Information_Systems_and_Analytics/all')",
        "soup = BeautifulSoup(r.text, 'lxml')",
        "print('FSB_ROWS', len(soup.find('table').find_all('tr')))",
      ].join("\n"),
    });
  }
  if (hasTools && /use python/i.test(userText)) {
    return toolCall("run_python", { code: "print(6 * 7)" });
  }
  if (/show me math|quadratic formula/i.test(userText)) {
    // TeX in both delimiter styles models actually emit, so the KaTeX
    // rendering path (incl. bracket-form normalization) is e2e-testable.
    return textParts([
      "The quadratic formula is ",
      "\\(x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}\\) for any ",
      "\\(a \\neq 0\\).\n\nA famous identity, displayed:\n\n",
      "$$\ne^{i\\pi} + 1 = 0\n$$\n\n",
      "And code stays code: `\\(not math\\)`.",
    ]);
  }
  const attachedAt = userText.indexOf("[Attached");
  if (attachedAt >= 0) {
    return textParts([
      "I read your file. ",
      `FILE_ACK ${userText.slice(attachedAt, attachedAt + 160)}`,
    ]);
  }
  if (lastUserHasFilePart(options, "application/pdf")) {
    return textParts(["That document came through. ", "PDF_ACK"]);
  }
  if (lastUserHasFilePart(options, "image/")) {
    return textParts(["I can see your image. ", "IMAGE_ACK"]);
  }
  return textParts(CHUNKS);
}

function mode(): string {
  return process.env.CHATISA_MOCK_LLM_MODE ?? "ok";
}

/** Names of the top-level properties the caller's JSON schema asks for. */
function requestedKeys(options: LanguageModelV4CallOptions): string[] {
  const format = options.responseFormat;
  if (!format || format.type !== "json" || !format.schema) return [];
  const schema = format.schema as { properties?: Record<string, unknown> };
  return Object.keys(schema.properties ?? {});
}

/**
 * Produces schema-shaped JSON for the structured calls this app makes. The
 * shape is chosen from the requested schema's own property names, so one mock
 * serves transcription, generation and grading without hard-coded coupling.
 */
function mockObjectFor(options: LanguageModelV4CallOptions): string {
  const keys = requestedKeys(options);

  if (mode() === "invalid") return JSON.stringify({ unexpected: true });

  // Page transcription: { text, legible }
  if (keys.includes("legible")) {
    if (mode() === "illegible") {
      return JSON.stringify({ text: "", legible: false });
    }
    return JSON.stringify({
      text: "Normalization removes transitive dependencies.\nA relation in 3NF has no transitive dependency on the primary key.",
      legible: true,
    });
  }

  // Exam generation: quote the real document from the prompt so the grounding
  // check is genuinely exercised rather than bypassed.
  if (keys.includes("questions")) {
    return JSON.stringify({ questions: mockQuestions(options) });
  }

  // Tailored resume. Quotes real lines out of the resume in the prompt, so the
  // grounding check is genuinely exercised rather than bypassed, and includes
  // one deliberately ungrounded bullet so the warning path is covered too.
  if (keys.includes("sections") && keys.includes("skills")) {
    const prompt = promptText(options);
    const resume = /<resume nonce="[^"]*">\n([\s\S]*?)\n<\/resume/.exec(prompt)?.[1] ?? "";
    const lines = resume
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 25);
    const first = lines[0] ?? "Built weekly reports";
    const second = lines[1] ?? first;
    return JSON.stringify({
      education: {
        degree: "Bachelors of Science",
        majorMinor: "Business Analytics / Statistics",
        graduation: "Expected Graduation 2027",
        gpa: "3.6",
        honors: [],
      },
      sections: [
        {
          heading: "RELEVANT EXPERIENCE",
          entries: [
            {
              organization: "Acme Logistics",
              title: "Data Analytics Intern",
              location: "Cincinnati, OH",
              dates: "Summer 2025",
              bullets: [
                { text: first, sourceLine: first },
                { text: second, sourceLine: second },
                {
                  // Ungrounded on purpose: exercises the warning the student
                  // can override.
                  text: "Directed a team of forty consultants across three continents",
                  sourceLine: null,
                },
              ],
            },
          ],
        },
      ],
      skills: ["R", "Python", "SQL"],
    });
  }

  // Cover letter.
  if (keys.includes("salutation") && keys.includes("paragraphs")) {
    const prompt = promptText(options);
    const resume = /<resume nonce="[^"]*">\n([\s\S]*?)\n<\/resume/.exec(prompt)?.[1] ?? "";
    const line = resume
      .split("\n")
      .map((l) => l.trim())
      .find((l) => l.length > 25) ?? "Built weekly reports in Excel and SQL";
    return JSON.stringify({
      salutation: "Dear Hiring Manager",
      paragraphs: [
        {
          text: "Through Miami University's career management system, Handshake, I learned of this position.",
          addresses: null,
          sourceLine: null,
        },
        { text: line, addresses: "analytics experience", sourceLine: line },
        {
          text: "Your team's work on operations analytics is what drew me to apply. After a review of my resume, I welcome further conversation.",
          addresses: null,
          sourceLine: null,
        },
      ],
      closing: "Sincerely,",
    });
  }

  // Interview: the next question. Varies with how many have been asked, so a
  // test can tell turn 3 from turn 1 and the duplicate check is meaningful.
  if (keys.includes("question") && keys.includes("topic")) {
    const prompt = promptText(options);
    const asked = Number(/Ask question (\d+) of/.exec(prompt)?.[1] ?? 1);
    const bank = [
      "Tell me about a project where the data did not behave as you expected.",
      "Walk me through how you would decide whether a difference between two groups is real.",
      "Describe a time you had to change your approach after getting feedback.",
      "How would you explain a regression coefficient to someone in marketing?",
      "Tell me about a time you disagreed with a teammate about an analysis.",
      "What would you check first if a dashboard suddenly showed impossible numbers?",
    ];
    const topics = [
      "handling messy data",
      "statistical judgement",
      "responding to feedback",
      "explaining results",
      "teamwork",
      "debugging",
    ];
    const index = Math.max(0, asked - 1) % bank.length;
    return JSON.stringify({ question: bank[index], topic: topics[index] });
  }

  // Interview: judging one answer against the fixed rubric. Reads the answer so
  // an empty one is genuinely marked down rather than always passing.
  if (keys.includes("criteria") && keys.includes("improvement")) {
    const prompt = promptText(options);
    const answer = /<answer nonce="[^"]*">\n([\s\S]*?)\n<\/answer/.exec(prompt)?.[1] ?? "";
    const substantive = answer.trim().length > 40;
    const verdict = substantive ? "met" : "not_met";
    return JSON.stringify({
      criteria: [
        { id: "answered_the_question", verdict },
        { id: "specific_evidence", verdict: substantive ? "partly" : "not_met" },
        { id: "structure", verdict },
        { id: "reasoning", verdict: substantive ? "partly" : "not_met" },
      ],
      strength: substantive
        ? "You gave a clear situation and stayed on the question that was asked."
        : "You attempted the question.",
      improvement: substantive
        ? "Say what the result was, since you described the actions but not the outcome."
        : "Give an actual example, even a short one, rather than a general statement.",
    });
  }

  // Interview: the closing report.
  if (keys.includes("didWell") && keys.includes("workOn")) {
    return JSON.stringify({
      didWell: [
        "You answered the question that was asked rather than the one you wanted.",
        "Your examples came from real work you had done.",
        "You explained your reasoning instead of only stating conclusions.",
      ],
      workOn: [
        "Name the outcome of each story, not just the actions.",
        "Lead with the situation in one sentence before the detail.",
        "Quantify results where you can, even roughly.",
      ],
      overall: "You came across as thoughtful and honest, with room to be more concrete.",
    });
  }

  // Interview: condensing a resume and posting into short briefs.
  if (keys.includes("candidateBrief") && keys.includes("roleBrief")) {
    return JSON.stringify({
      candidateBrief:
        "A business analytics student with coursework in R and SQL and one internship building reports.",
      roleBrief:
        "An analyst role focused on turning operational data into recommendations for non-technical stakeholders.",
    });
  }

  // Grading of a written answer.
  if (keys.includes("criteria")) {
    return JSON.stringify({
      criteria: [
        {
          criterion: "Names the concept",
          met: "yes",
          justification: "The answer names it directly.",
        },
      ],
      feedback: "Clear answer that identifies the key idea.",
    });
  }

  return JSON.stringify(Object.fromEntries(keys.map((k) => [k, null])));
}

/** Everything the caller put in the prompt, as one string. */
function promptText(options: LanguageModelV4CallOptions): string {
  const chunks: string[] = [];
  for (const message of options.prompt ?? []) {
    const content = (message as { content?: unknown }).content;
    if (typeof content === "string") chunks.push(content);
    else if (Array.isArray(content)) {
      for (const part of content) {
        const text = (part as { text?: string }).text;
        if (typeof text === "string") chunks.push(text);
      }
    }
  }
  return chunks.join("\n");
}

/**
 * Builds questions whose sourceQuote is real text lifted out of the document
 * in the prompt, and whose sourcePage is a page label from it. That way the
 * mock passes grounding for the same reason a good model would.
 */
function mockQuestions(options: LanguageModelV4CallOptions) {
  const prompt = promptText(options);
  const requested = Number(/Number of questions:\s*(\d+)/.exec(prompt)?.[1] ?? 3);
  // Read the requested type from its own field. Matching the whole prompt
  // would also match the phrase "multiple choice" inside the instructions.
  const typeLine = /Question type:\s*(.+)/.exec(prompt)?.[1] ?? "";
  const wantsMcq = /Multiple Choice/i.test(typeLine);

  // Page bodies, keyed by the [page N] markers the prompt uses.
  const segments: { page: number; body: string }[] = [];
  const re = /\[page (\d+)\]\n([\s\S]*?)(?=\n\[page \d+\]|$)/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(prompt)) !== null) {
    const body = match[2].trim();
    if (body.length >= 60) segments.push({ page: Number(match[1]), body });
  }
  if (segments.length === 0) return [];

  const count = Math.max(1, Math.min(requested, 20));
  return Array.from({ length: count }, (_, i) => {
    const seg = segments[i % segments.length];
    // A verbatim slice, offset per question so stems are not duplicates.
    const start = Math.min(i * 7, Math.max(0, seg.body.length - 120));
    const quote = seg.body.slice(start, start + 120).trim();
    return {
      type: wantsMcq ? "multiple_choice" : "short_answer",
      // Wording follows the quoted fragment, so stems differ from each
      // other the way real generated questions do.
      stem: `Explain, in your own words, what the material means by "${quote.slice(0, 60).trim()}" (page ${seg.page}).`,
      options: wantsMcq
        ? [`Answer ${i + 1}A`, `Answer ${i + 1}B`, `Answer ${i + 1}C`, `Answer ${i + 1}D`]
        : null,
      // Always position 0, deliberately. Real models cluster the correct
      // answer in one slot, which is the defect reported on 2026-07-21, and a
      // mock that spreads answers evenly would let a regression in
      // lib/exam/answer-positions.ts pass unnoticed.
      correctIndex: wantsMcq ? 0 : null,
      modelAnswer:
        "A complete answer restates the concept and applies it to an example.",
      rubric: [
        { criterion: "Names the concept accurately", points: 5 },
        { criterion: "Applies it to an example", points: 5 },
      ],
      explanation:
        "The material defines the concept and shows how it is applied.",
      topic: `Topic ${i + 1}`,
      bloom: "understand",
      sourceQuote: quote,
      sourcePage: seg.page,
    };
  });
}

export function getMockModel() {
  return new MockLanguageModelV4({
    doStream: async (options) => {
      const chunks = mockStreamParts(options);
      // Tool-input deltas stream at real-provider speed (bursts, no delay):
      // update-frequency bugs must be reproducible in tests. Text answers
      // keep the slow cadence the stop-mid-stream tests depend on.
      const burst = chunks.some((c) => c.type === "tool-input-delta");
      return {
        stream: simulateReadableStream<LanguageModelV4StreamPart>({
          chunkDelayInMs: burst ? 0 : 120,
          initialDelayInMs: 60,
          chunks,
        }),
      };
    },
    doGenerate: async (options) => ({
      content: [{ type: "text" as const, text: mockObjectFor(options) }],
      finishReason: { unified: "stop" as const, raw: "stop" },
      usage: USAGE,
      warnings: [],
    }),
  });
}
