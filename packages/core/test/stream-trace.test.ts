import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const streamTextMock = vi.hoisted(() => vi.fn());
vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return { ...actual, streamText: streamTextMock };
});

import { streamChat } from "../src/agent/agent.js";
import { createProtocolLeakageGuard } from "../src/agent/answer-validation.js";
import { TraceStore } from "../src/agent/trace.js";
import { KnowledgeBase } from "../src/okf/index.js";

let root: string | undefined;

afterEach(async () => {
  streamTextMock.mockReset();
  vi.unstubAllEnvs();
  if (root) await fs.rm(root, { recursive: true, force: true });
  root = undefined;
});

describe("stream chat traces", () => {
  it("suppresses malformed text before the client-visible stream", async () => {
    const guard = createProtocolLeakageGuard();
    const stopStream = vi.fn();
    const transformed = guard.transform({ tools: {}, stopStream });
    const writer = transformed.writable.getWriter();
    const reader = transformed.readable.getReader();
    const outputPromise = (async () => {
      const output: unknown[] = [];
      while (true) {
        const next = await reader.read();
        if (next.done) return output;
        output.push(next.value);
      }
    })();
    await writer.write({ type: "text-start", id: "text-1" });
    await writer.write({ type: "text-delta", id: "text-1", text: "Here is: [read_concept(" });
    await writer.write({ type: "text-delta", id: "text-1", text: "path='x')]" });
    await writer.write({ type: "text-end", id: "text-1" });
    await writer.close();

    const output = await outputPromise;
    expect(output.filter((part: any) => part.type === "text-delta")).toEqual([]);
    expect(guard.wasMalformed()).toBe(true);
    expect(stopStream).toHaveBeenCalledOnce();
  });

  it("releases ordinary text after the bounded text-part gate", async () => {
    const guard = createProtocolLeakageGuard();
    const transformed = guard.transform({ tools: {}, stopStream: vi.fn() });
    const writer = transformed.writable.getWriter();
    const reader = transformed.readable.getReader();
    const outputPromise = (async () => {
      const output: any[] = [];
      while (true) {
        const next = await reader.read();
        if (next.done) return output;
        output.push(next.value);
      }
    })();
    await writer.write({ type: "text-start", id: "text-1" });
    await writer.write({ type: "text-delta", id: "text-1", text: "Ordinary answer." });
    await writer.write({ type: "text-end", id: "text-1" });
    await writer.close();

    const output = await outputPromise;
    expect(output.map((part) => part.type)).toEqual(["text-start", "text-delta", "text-end"]);
    expect(output[1].text).toBe("Ordinary answer.");
    expect(guard.wasMalformed()).toBe(false);
  });

  it("finalises one failed trace when asynchronous callbacks race", async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-stream-"));
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_MODEL", "test-model");
    let callbacks: Record<string, (event?: unknown) => Promise<void>> | undefined;
    streamTextMock.mockImplementation((options: Record<string, unknown>) => {
      callbacks = options as typeof callbacks;
      return {};
    });

    await streamChat(new KnowledgeBase(root), [{ role: "user", content: "hello" }]);
    await Promise.all([
      callbacks!.onError!({ error: new Error("provider failed") }),
      callbacks!.onAbort!(),
    ]);

    const traces = await new TraceStore(root).list();
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ outcome: "failed", answer: "provider failed" });
  });

  it("rejects malformed streamed text instead of tracing success", async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-stream-"));
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_MODEL", "test-model");
    let onFinish: ((event: unknown) => Promise<void>) | undefined;
    let experimentalTransform: unknown;
    streamTextMock.mockImplementation((options: Record<string, unknown>) => {
      onFinish = options.onFinish as typeof onFinish;
      experimentalTransform = options.experimental_transform;
      return {};
    });

    await streamChat(new KnowledgeBase(root), [{ role: "user", content: "hello" }]);
    expect(experimentalTransform).toEqual(expect.any(Function));
    await expect(
      onFinish!({
        text: "<|tool_call_start|>[read_concept(path='x')]",
        totalUsage: {},
        steps: [{ toolCalls: [] }],
      })
    ).rejects.toThrow("protocol leakage");

    const traces = await new TraceStore(root).list();
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ outcome: "failed" });
    expect(traces[0].answer).not.toContain("<|tool_call");
  });

  it("traces a synthesis assertion failure from onFinish", async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-stream-"));
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_MODEL", "test-model");
    let onFinish: ((event: unknown) => Promise<void>) | undefined;
    streamTextMock.mockImplementation((options: Record<string, unknown>) => {
      onFinish = options.onFinish as typeof onFinish;
      return {};
    });

    await streamChat(new KnowledgeBase(root), [{ role: "user", content: "hello" }]);
    await expect(
      onFinish!({ text: "", totalUsage: {}, steps: [{ toolCalls: [{}] }] })
    ).rejects.toThrow("step limit");

    const traces = await new TraceStore(root).list();
    expect(traces).toHaveLength(1);
    expect(traces[0]).toMatchObject({ outcome: "failed" });
  });
});
