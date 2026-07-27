import type { StopCondition, ToolSet } from "ai";

/**
 * Stop the server loop when a tool call is waiting on the student's browser.
 *
 * Ask Anything mixes three kinds of tool in one conversation:
 *
 *   - server-executed (search_papers, read_url, get_miami_style): declared WITH
 *     an execute, so their result lands in the same step.
 *   - provider-executed (code_execution, code_interpreter): the provider runs
 *     them and returns the result inside its own response.
 *   - browser-executed (run_python, run_r, run_sql): declared WITHOUT an
 *     execute. The call streams to the page, the student's runtime produces the
 *     result, and the client sends it back on the NEXT request.
 *
 * Only the third kind ends the server's turn, and nothing was telling the loop
 * that. Measured on Claude Opus 5, 2026-07-26:
 *
 *   step 1  code_execution (provider) + 2x search_papers (server)  -> 3 results
 *   step 2  run_python (browser)                                   -> 0 results
 *   then    AI_APICallError: AI_MissingToolResultsError:
 *           Tool result is missing for tool call toolu_01ShWGRu83...
 *
 * Because step 1 resolved every call, the loop kept going; step 2 left a browser
 * call unresolved, and the attempt at step 3 failed because that call has no
 * result and never will on the server. The student saw "That response failed".
 *
 * It looked model-specific only because it needs a model that mixes a
 * provider-executed tool with a browser tool in one turn. Opus 5 does; Sonnet 5
 * answered the same class of question with browser tools alone and never tripped
 * it. Nothing about the model or the code_execution tool version is at fault.
 */
export function awaitsBrowserTool<TOOLS extends ToolSet>(): StopCondition<TOOLS> {
  return ({ steps }) => {
    const last = steps.at(-1);
    if (!last) return false;
    const resolved = new Set(last.toolResults.map((result) => result.toolCallId));
    return last.toolCalls.some(
      (call) => !call.providerExecuted && !resolved.has(call.toolCallId),
    );
  };
}
