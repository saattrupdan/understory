import { describe, it, expect } from "vitest";
import { TraceRecorder } from "../src/agent/trace.js";

describe("trace usage (#15)", () => {
  it("records token usage on the finalized trace", () => {
    const r = new TraceRecorder();
    r.record("read_concept", "/a.md", ["/a.md"]);
    const trace = r.finalize("query", "q", "a", "success", ["openai:m"], {
      inputTokens: 3200,
      outputTokens: 410,
    });
    expect(trace.usage).toEqual({ inputTokens: 3200, outputTokens: 410 });
  });

  it("records numeric timing metadata without prompt or response content", async () => {
    const r = new TraceRecorder();
    const trace = r.finalize("query", "sensitive question", "sensitive answer", "success", ["openai:model"], undefined, undefined, undefined, {
      promptContextMs: 12,
      generationMs: 34,
      modelCalls: [{ model: "openai:model", durationMs: 30, inputTokens: 8, outputTokens: 3 }],
    });
    const saved = JSON.stringify(trace.timing);
    expect(trace.timing?.modelCalls).toEqual([
      { model: "openai:model", durationMs: 30, inputTokens: 8, outputTokens: 3 },
    ]);
    expect(saved).not.toContain("sensitive");
  });

  it("leaves usage undefined when the provider reports none", () => {
    const r = new TraceRecorder();
    const trace = r.finalize("query", "q", "a");
    expect(trace.usage).toBeUndefined();
  });
});
