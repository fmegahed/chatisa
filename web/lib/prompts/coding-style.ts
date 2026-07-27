/**
 * The single source of truth for ChatISA's code-style rules, so the Coding
 * Tutor, the Sandbox chat, and inline completions all teach and generate code
 * the same way (DRY). This is the exact style block the legacy Coding Companion
 * prompt used; reusing it keeps that prompt byte-for-byte unchanged.
 */
export const CODING_STYLE_RULES = `When you show R code, you must use:
  (a) library_name::function_name() syntax as this avoids conflicts in function names and makes it clear to the student where the function is imported from when there are multiple packages loaded. Based on this, do NOT use library() in the beginning of your code chunk and use if(require(library)==FALSE) install.packages(library), and
  (b) use the native pipe |> as your pipe operator.

On the other hand for Python, break chained methods into multiple lines using parentheses; for example, do NOT write df.groupby('Region')['Sales'].agg('sum') on one line.`;
