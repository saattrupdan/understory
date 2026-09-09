import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const providerState = vi.hoisted(() => {
  let calls = 0;
  const model = {
    specificationVersion: "v2" as const,
    provider: "test",
    modelId: "test-model",
    supportedUrls: {},
    doStream: async () => {
      calls += 1;
      const chunks =
        calls === 1
          ? [
              { type: "stream-start", warnings: [] },
              {
                type: "tool-call",
                toolCallId: "write-1",
                toolName: "write_concept",
                input: JSON.stringify({
                  path: "/facts/new.md",
                  frontmatter: { type: "Fact" },
                  body: "written once",
                  log_summary: "create test fact",
                }),
              },
              {
                type: "finish",
                finishReason: "tool-calls",
                usage: { inputTokens: 1, outputTokens: 1 },
              },
            ]
          : [
              { type: "stream-start", warnings: [] },
              { type: "text-start", id: "text-1" },
              {
                type: "text-delta",
                id: "text-1",
                delta: 'Calling write_concept(path="/facts/new.md")',
              },
              { type: "text-end", id: "text-1" },
              {
                type: "finish",
                finishReason: "stop",
                usage: { inputTokens: 1, outputTokens: 1 },
              },
            ];
      return {
        stream: new ReadableStream({
          start(controller) {
            for (const chunk of chunks) controller.enqueue(chunk);
            controller.close();
          },
        }),
      };
    },
  };
  return { model, get calls() { return calls; } };
});

vi.mock("../src/providers/index.js", async () => {
  const actual = await vi.importActual<typeof import("../src/providers/index.js")>(
    "../src/providers/index.js"
  );
  return {
    ...actual,
    createModel: vi.fn(async () => providerState.model),
    resolveModelConfig: () => ({
      baseURL: "http://test.invalid/v1",
      apiKey: "test",
      format: "openai" as const,
      model: "test-model",
    }),
    resolveFallbackConfig: () => null,
  };
});

import { streamChat } from "../src/agent/agent.js";
import { TraceStore } from "../src/agent/trace.js";
import { KnowledgeBase } from "../src/okf/index.js";

let root: string | undefined;

afterEach(async () => {
  vi.unstubAllEnvs();
  if (root) await fs.rm(root, { recursive: true, force: true });
  root = undefined;
});

describe("chat protocol guard through AI SDK UI stream", () => {
  it("reports a partial mutation before any successful completion marker", async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-chat-api-"));
    const kb = new KnowledgeBase(root);
    const { result } = await streamChat(kb, [{ role: "user", content: "create a fact" }]);

    const response = result.toUIMessageStreamResponse({
      onError: (error) => (error instanceof Error ? error.message : String(error)),
    });
    const body = await response.text();
    const errorIndex = body.indexOf('"type":"error"');
    const finishStepIndex = body.lastIndexOf('"type":"finish-step"');
    const finishIndex = body.lastIndexOf('"type":"finish"');

    expect(body).not.toContain('Calling write_concept(path="/facts/new.md")');
    expect(errorIndex).toBeGreaterThanOrEqual(0);
    expect(body).toContain("/facts/new.md");
    expect(errorIndex).toBeLessThan(finishStepIndex);
    expect(errorIndex).toBeLessThan(finishIndex);
    expect(providerState.calls).toBe(2);
    expect((await kb.readConcept("/facts/new.md")).body).toBe("written once\n");

    const traces = await new TraceStore(root).list();
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ outcome: "partial" });
    expect(traces[0].answer).toContain("/facts/new.md");
  });
});
