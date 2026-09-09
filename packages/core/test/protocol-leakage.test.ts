import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const generateTextMock = vi.hoisted(() => vi.fn());
vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return { ...actual, generateText: generateTextMock };
});

import { runMutation, runQuery, type QueryResult } from "../src/agent/agent.js";
import {
  clearHotMemory,
  hotLookup,
  recordHotWrite,
} from "../src/agent/hot-memory.js";
import { runRecall } from "../src/agent/recall.js";
import { clearQueryCache, runQueryCached } from "../src/agent/query-cache.js";
import { isMalformedAnswer } from "../src/agent/answer-validation.js";
import { TraceStore } from "../src/agent/trace.js";
import { KnowledgeBase } from "../src/okf/index.js";

let root: string;
let kb: KnowledgeBase;

beforeEach(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-protocol-"));
  kb = new KnowledgeBase(root);
  clearHotMemory();
  clearQueryCache();
  vi.stubEnv("LLM_API_FORMAT", "openai");
  vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
  vi.stubEnv("LLM_API_KEY", "test");
  vi.stubEnv("LLM_MODEL", "test-model");
});

afterEach(async () => {
  await fs.rm(root, { recursive: true, force: true });
  generateTextMock.mockReset();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("textual tool-call answer validation", () => {
  it("recognises complete, truncated, and prefaced protocol leakage", () => {
    expect(isMalformedAnswer("<|tool_call_start|>[read_concept(path='x')]<|tool_call_end|>")).toBe(true);
    expect(isMalformedAnswer("<|tool_call_start|>[read_concept(path='x')]")).toBe(true);
    expect(isMalformedAnswer("Here is the call: [read_concept(path='x')]")).toBe(true);
    expect(isMalformedAnswer("Here's the call: [read_concept(path='x')]")).toBe(true);
    expect(isMalformedAnswer("I will use:\n[read_concept(path='x'")).toBe(true);
    expect(isMalformedAnswer("[read_concept(path='x'")).toBe(true);
  });

  it("allows prose and quoted examples containing tool syntax", () => {
    expect(isMalformedAnswer("The read_concept tool is used to inspect a concept.")).toBe(false);
    expect(isMalformedAnswer("We document write_concept(path='x') in the runbook.")).toBe(false);
    expect(isMalformedAnswer("Use [read_concept(path='x')] when the answer needs a concept.")).toBe(false);
    expect(isMalformedAnswer("The literal marker <|tool_call_start|> is described here.")).toBe(false);
    expect(isMalformedAnswer("`<|tool_call_start|>[read_concept(path='x')]` is a marker example.")).toBe(false);
    expect(isMalformedAnswer('The example "[read_concept(path=\'x\')]" is quoted.')).toBe(false);
    expect(isMalformedAnswer("The example '[read_concept(path=\"x\")]' is quoted.")).toBe(false);
  });
});

describe("fast-path answer validation", () => {
  it("declines malformed hot-memory output", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact" }, "alpha", "add");
    recordHotWrite("/facts/a.md");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "<|tool_call_start|>[read_concept(path='/facts/a.md')]",
      finishReason: "stop" as const,
    }));

    await expect(hotLookup(kb, "what is alpha?", {}, generate)).resolves.toBeNull();
    expect(logged).toHaveBeenCalled();
    expect(String(logged.mock.calls[0][0])).toContain("hot memory declined");
  });

  it("declines malformed recall output and preserves candidates", async () => {
    await kb.writeConcept(
      "/facts/a.md",
      { type: "Fact", title: "Alpha" },
      "alpha value",
      "add"
    );
    vi.stubEnv("RECALL_MIN_SCORE", "0");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "SUFFICIENT\n[read_concept(path='/facts/a.md')",
      finishReason: "stop" as const,
    }));

    const result = await runRecall(kb, "alpha value", {}, generate);
    expect(result.answer).toBeNull();
    expect(result.paths).toContain("/facts/a.md");
    expect(logged).toHaveBeenCalledWith(expect.stringContaining("recall declined"));
  });
});

describe("deep agent answer validation", () => {
  const step = { toolCalls: [] as unknown[] };

  it("repairs once with the same model and generated context", async () => {
    generateTextMock
      .mockResolvedValueOnce({
        text: "<|tool_call_start|>[read_concept(path='x')]",
        steps: [
          {
            ...step,
            response: {
              messages: [
                {
                  role: "assistant",
                  content: [
                    {
                      type: "tool-call",
                      toolCallId: "call-1",
                      toolName: "read_concept",
                      input: { path: "x" },
                    },
                  ],
                },
                {
                  role: "tool",
                  content: [
                    {
                      type: "tool-result",
                      toolCallId: "call-1",
                      toolName: "read_concept",
                      output: { path: "x", body: "alpha" },
                    },
                  ],
                },
              ],
            },
          },
        ],
        // The combined v5 response has the successful transcript plus the
        // malformed final assistant message. The latter must be excluded.
        response: {
          messages: [
            {
              role: "assistant",
              content: [
                {
                  type: "tool-call",
                  toolCallId: "call-1",
                  toolName: "read_concept",
                  input: { path: "x" },
                },
              ],
            },
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "call-1",
                  toolName: "read_concept",
                  output: { path: "x", body: "alpha" },
                },
              ],
            },
            {
              role: "assistant",
              content: [{ type: "text", text: "<|tool_call_start|>[read_concept(path='x')]" }],
            },
          ],
        },
      })
      .mockResolvedValueOnce({ text: "The answer is alpha.", steps: [step] });

    const result = await runQuery(kb, "What is alpha?");
    expect(result.answer).toBe("The answer is alpha.");
    expect(generateTextMock).toHaveBeenCalledTimes(2);
    expect(generateTextMock.mock.calls[1][0].tools).toEqual({});
    const repairMessages = generateTextMock.mock.calls[1][0].messages;
    expect(repairMessages).toHaveLength(3);
    expect(JSON.stringify(repairMessages)).not.toContain("<|tool_call_start|>");
    expect(repairMessages[1]).toMatchObject({ role: "assistant", content: [{ type: "tool-call" }] });
    expect(repairMessages[2]).toMatchObject({ role: "tool", content: [{ type: "tool-result" }] });
    expect(await new TraceStore(root).list()).toEqual(
      expect.arrayContaining([expect.objectContaining({ outcome: "success", answer: result.answer })])
    );
  });

  it("fails and records no successful trace when repair is malformed", async () => {
    generateTextMock
      .mockResolvedValueOnce({
        text: "[read_concept(path='x')",
        steps: [step],
        response: { messages: [] },
      })
      .mockResolvedValueOnce({ text: "read_concepts(paths=['x'])", steps: [step] });

    await expect(runQuery(kb, "What is alpha?")).rejects.toThrow("protocol leakage");
    const traces = await new TraceStore(root).list();
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ outcome: "failed" });
    expect(traces[0].answer).not.toContain("<|tool_call");
    expect(traces[0].answer).not.toContain("read_concepts(");
  });
});

describe("mutation answer validation", () => {
  it("does not replay a write when its final summary is malformed", async () => {
    generateTextMock.mockImplementation(async (request: {
      tools?: Record<string, { execute?: (input: unknown) => Promise<unknown> }>;
    }) => {
      const writeTool = request.tools?.write_concept;
      await writeTool?.execute?.({
        path: "/facts/new.md",
        frontmatter: { type: "Fact" },
        body: "written once",
        log_summary: "create test fact",
      });
      return { text: "[write_concept(path='/facts/new.md')", steps: [step] };
    });

    const result = await runMutation(kb, "Create the test fact.");
    expect(result).toMatchObject({
      ok: false,
      status: "partial",
      filesChanged: ["/facts/new.md"],
    });
    expect(generateTextMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept("/facts/new.md")).body).toBe("written once\n");
    const traces = await new TraceStore(root).list();
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ outcome: "partial" });
    expect(traces[0].answer).not.toContain("write_concept(");
  });
});

describe("query cache validation", () => {
  it("does not cache or return a malformed deep result", async () => {
    const runner = vi.fn(async (): Promise<QueryResult> => ({
      answer: "<|tool_call_end|>",
      steps: 1,
      traceId: "t",
    }));
    const noHot = async () => null;
    const noRecall = async () => ({ answer: null, paths: [] as string[] });

    await expect(runQueryCached(kb, "q?", {}, runner, noHot, noRecall)).rejects.toThrow(
      "protocol leakage"
    );
    await expect(runQueryCached(kb, "q?", {}, runner, noHot, noRecall)).rejects.toThrow(
      "protocol leakage"
    );
    expect(runner).toHaveBeenCalledTimes(2);
  });
});
