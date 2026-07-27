import { describe, expect, it } from "vitest";
import { APICallError } from "ai";
import { classifyProviderFailure } from "@/lib/providers/errors";

function apiError(message: string, statusCode?: number, body?: string) {
  return new APICallError({
    message,
    url: "https://example.test/v1/chat",
    requestBodyValues: {},
    statusCode,
    responseBody: body,
  });
}

describe("classifyProviderFailure", () => {
  // Both strings below are copied verbatim from the user's production logs on
  // 2026-07-21. Previously each of these told the student "you can try again",
  // which is advice that could never succeed.
  it("treats an exhausted account as permanent and needing an operator", () => {
    const failure = classifyProviderFailure(
      apiError(
        "Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.",
      ),
    );
    expect(failure.kind).toBe("account_problem");
    expect(failure.retryable).toBe(false);
    expect(failure.operatorAction).toBe(true);
    expect(failure.message).not.toMatch(/try again/i);
  });

  it("treats a model no provider serves as permanent and needing an operator", () => {
    const failure = classifyProviderFailure(
      apiError(
        "The requested model 'meta-llama/Llama-4-Maverick-17B-128E-Instruct' is not supported by any provider you have enabled.",
      ),
    );
    expect(failure.kind).toBe("model_unavailable");
    expect(failure.retryable).toBe(false);
    expect(failure.operatorAction).toBe(true);
    expect(failure.message).not.toMatch(/try again/i);
  });

  it("never leaks provider internals to the student", () => {
    const failure = classifyProviderFailure(
      apiError("Invalid api key sk-ant-secret-value-here", 401),
    );
    expect(failure.message).not.toMatch(/sk-ant/);
    expect(failure.message).not.toMatch(/api key/i);
  });

  it("says waiting helps when the model is merely busy", () => {
    const failure = classifyProviderFailure(
      apiError("Rate limit reached for this model", 429),
    );
    expect(failure.kind).toBe("rate_limited");
    expect(failure.retryable).toBe(true);
    expect(failure.operatorAction).toBe(false);
  });

  it("tells the student to start a new chat when the context is full", () => {
    const failure = classifyProviderFailure(
      apiError("This model's maximum context length is 32768 tokens", 400),
    );
    expect(failure.kind).toBe("context_too_long");
    expect(failure.retryable).toBe(false);
    expect(failure.message).toMatch(/new chat|context window/i);
  });

  it("reads the provider explanation from the response body too", () => {
    const failure = classifyProviderFailure(
      apiError("Request failed", 400, '{"error":{"message":"insufficient_quota"}}'),
    );
    expect(failure.kind).toBe("account_problem");
  });

  it("treats a server fault as worth retrying", () => {
    const failure = classifyProviderFailure(apiError("Bad gateway", 502));
    expect(failure.kind).toBe("transient");
    expect(failure.retryable).toBe(true);
  });

  it("treats a dropped connection as worth retrying", () => {
    const failure = classifyProviderFailure(new Error("fetch failed"));
    expect(failure.kind).toBe("transient");
    expect(failure.retryable).toBe(true);
  });

  it("falls back safely for an error it does not recognise", () => {
    const failure = classifyProviderFailure(new Error("something odd"));
    expect(failure.kind).toBe("unknown");
    expect(failure.message).toMatch(/try again/i);
  });
});
