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
