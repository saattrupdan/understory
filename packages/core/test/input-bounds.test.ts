import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const generateTextMock = vi.hoisted(() => vi.fn());
const streamTextMock = vi.hoisted(() => vi.fn());
vi.mock("ai", async () => {
  const actual = await vi.importActual<typeof import("ai")>("ai");
  return { ...actual, generateText: generateTextMock, streamText: streamTextMock };
});

import { runMutation, runQuery, streamChat } from "../src/agent/agent.js";
import { inputLength } from "../src/agent/limits.js";
import { KnowledgeBase } from "../src/okf/index.js";
import { TraceStore } from "../src/agent/trace.js";

let root: string | undefined;

afterEach(async () => {
  vi.unstubAllEnvs();
  generateTextMock.mockReset();
  streamTextMock.mockReset();
  if (root) await fs.rm(root, { recursive: true, force: true });
  root = undefined;
});

async function knowledgeBase(): Promise<KnowledgeBase> {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-input-bounds-"));
  return new KnowledgeBase(root);
}

describe("agent input bounds", () => {
  it("rejects million-character query and mutation before model or trace", async () => {
    const kb = await knowledgeBase();
    vi.stubEnv("AGENT_MAX_INPUT_CHARS", "32");
    const giant = "x".repeat(1_000_000);

    await expect(runQuery(kb, giant)).rejects.toThrow("AGENT_MAX_INPUT_CHARS");
    await expect(runMutation(kb, giant)).rejects.toThrow("AGENT_MAX_INPUT_CHARS");
    expect(generateTextMock).not.toHaveBeenCalled();
    expect(await new TraceStore(root!).list()).toHaveLength(0);
  });

  it("allows the exact string bound and rejects the next character", async () => {
    const kb = await knowledgeBase();
    vi.stubEnv("AGENT_MAX_INPUT_CHARS", "3");
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    generateTextMock.mockResolvedValue({ text: "ok", steps: [] });

    await expect(runQuery(kb, "abc")).resolves.toMatchObject({ answer: "ok" });
    await expect(runQuery(kb, "abcd")).rejects.toThrow("AGENT_MAX_INPUT_CHARS");
    expect(generateTextMock).toHaveBeenCalledTimes(1);
  });

  it("uses a separate bounded step budget for chat", async () => {
    const kb = await knowledgeBase();
    vi.stubEnv("AGENT_MAX_STEPS", "2");
    vi.stubEnv("AGENT_CHAT_MAX_STEPS", "10");
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    streamTextMock.mockReturnValue({});

    await streamChat(kb, [{ role: "user", content: "hello" }]);
    const options = streamTextMock.mock.calls[0]?.[0] as {
      prepareStep: (step: { stepNumber: number }) => unknown;
    };
    expect(options.prepareStep({ stepNumber: 8 })).toBeUndefined();
    expect(options.prepareStep({ stepNumber: 9 })).toEqual({ activeTools: [] });
  });

  it("uses a separate finite chat-history bound", async () => {
    const kb = await knowledgeBase();
    vi.stubEnv("AGENT_MAX_INPUT_CHARS", "32");
    vi.stubEnv("AGENT_CHAT_MAX_INPUT_CHARS", "40000");
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    streamTextMock.mockReturnValue({});

    await expect(
      streamChat(kb, [{ role: "user", content: "x".repeat(30_000) }])
    ).resolves.toBeDefined();
    await expect(
      streamChat(kb, [{ role: "user", content: "x".repeat(50_000) }])
    ).rejects.toThrow("AGENT_CHAT_MAX_INPUT_CHARS");
  });

  it("bounds the complete serialized chat history before streaming", async () => {
    const kb = await knowledgeBase();
    const messages = [{ role: "user" as const, content: "hello" }];
    vi.stubEnv("AGENT_CHAT_MAX_INPUT_CHARS", String(inputLength(messages)));
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    streamTextMock.mockReturnValue({});

    await expect(streamChat(kb, messages)).resolves.toBeDefined();
    await expect(streamChat(kb, [{ role: "user", content: "x".repeat(1_000_000) }])).rejects.toThrow(
      "AGENT_CHAT_MAX_INPUT_CHARS"
    );
    expect(streamTextMock).toHaveBeenCalledTimes(1);
  });
});
