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

  it("does not impose an application-level tool-step cutoff on chat", async () => {
    const kb = await knowledgeBase();
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    streamTextMock.mockReturnValue({});

    await streamChat(kb, [{ role: "user", content: "hello" }]);
    const options = streamTextMock.mock.calls[0]?.[0] as Record<string, unknown>;
    // The empty condition list delegates continuation to the AI SDK/model and
    // catches a bounded stopWhen callback or a final-step prepareStep hook.
    expect(options).not.toHaveProperty("prepareStep");
    expect(options.stopWhen).toEqual([]);
  });

  it("passes chat history beyond the one-shot input bound to streamText", async () => {
    const kb = await knowledgeBase();
    vi.stubEnv("AGENT_MAX_INPUT_CHARS", "32");
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    streamTextMock.mockReturnValue({});
    const messages = [{ role: "user" as const, content: "x".repeat(40) }];

    await expect(streamChat(kb, messages)).resolves.toBeDefined();
    expect(streamTextMock).toHaveBeenCalledTimes(1);
    expect((streamTextMock.mock.calls[0]?.[0] as { messages: unknown }).messages).toEqual(messages);
  });

  it("does not apply configured data budgets to interactive chat", async () => {
    const kb = await knowledgeBase();
    const longBody = "document-body-".repeat(80);
    for (let index = 0; index < 16; index += 1) {
      await kb.writeConcept(
        `/facts/context-${index}.md`,
        { type: `UniqueContextType${index}`, title: `Context ${index}` },
        index === 0 ? longBody : `context ${index}`,
        "Added context."
      );
    }
    vi.stubEnv("AGENT_MAX_DOCUMENT_CHARS", "4");
    vi.stubEnv("AGENT_MAX_TOOL_RESULT_CHARS", "300");
    vi.stubEnv("AGENT_MAX_SYSTEM_CONTEXT_CHARS", "240");
    vi.stubEnv("AGENT_MAX_INPUT_CHARS", "180");
    vi.stubEnv("LLM_API_FORMAT", "openai");
    vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
    vi.stubEnv("LLM_API_KEY", "test");
    vi.stubEnv("LLM_MODEL", "test-model");
    streamTextMock.mockReturnValue({});

    await streamChat(kb, [{ role: "user", content: "inspect and update" }]);
    const options = streamTextMock.mock.calls[0]?.[0] as {
      system: string;
      tools: Record<
        string,
        {
          inputSchema: { safeParse(value: unknown): { success: boolean } };
          execute(args: any, context: any): Promise<any>;
        }
      >;
    };
    const toolContext = { toolCallId: "test", messages: [] };

    expect(options.system).toContain("UniqueContextType15");
    expect(options.system).toContain("context-15.md");
    expect(options.system).not.toContain("system types truncated");
    expect(options.system).not.toContain("system tree truncated");

    const page = await options.tools.read_concept.execute(
      { path: "/facts/context-0.md", offset: 0 },
      toolContext
    );
    expect(page.body).toBe(longBody + "\n");
    expect(page).toMatchObject({ truncated: false, next_offset: null });

    const firstTree = await options.tools.list_directory.execute({}, toolContext);
    const secondTree = await options.tools.list_directory.execute({}, toolContext);
    expect(firstTree).toContain("context-15.md");
    expect(secondTree).toBe(firstTree);
    expect(JSON.stringify(page).length + firstTree.length + secondTree.length).toBeGreaterThan(300);

    const giantWrite = {
      path: "/facts/schema-unbounded.md",
      frontmatter: { type: "Fact", payload: "x".repeat(500) },
      body: "x".repeat(500),
      log_summary: "Added schema-unbounded fact.",
    };
    expect(options.tools.write_concept.inputSchema.safeParse(giantWrite).success).toBe(true);

    for (const suffix of ["a", "b"]) {
      await options.tools.write_concept.execute(
        {
          path: `/facts/write-${suffix}.md`,
          frontmatter: { type: "Fact" },
          body: suffix.repeat(80),
          log_summary: `Added write ${suffix}.`,
        },
        toolContext
      );
    }
    await expect(kb.readConcept("/facts/write-a.md")).resolves.toBeDefined();
    await expect(kb.readConcept("/facts/write-b.md")).resolves.toBeDefined();
  });
});
