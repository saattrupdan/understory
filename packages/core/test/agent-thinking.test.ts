import { afterEach, describe, expect, it, vi } from "vitest";

const createModelMock = vi.hoisted(() => vi.fn());
vi.mock("../src/providers/index.js", async () => {
  const actual = await vi.importActual<typeof import("../src/providers/index.js")>(
    "../src/providers/index.js"
  );
  return { ...actual, createModel: createModelMock };
});

import { resolveAgentModel } from "../src/agent/agent.js";

const configuredEnv = (extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv => ({
  LLM_API_BASE_URL: "http://primary/v1",
  LLM_MODEL: "primary-model",
  LLM_MAX_OUTPUT_TOKENS: "500",
  LLM_FALLBACK_API_BASE_URL: "http://fallback/v1",
  LLM_FALLBACK_MODEL: "fallback-model",
  LLM_FALLBACK_MAX_OUTPUT_TOKENS: "300",
  ...extra,
});

describe("deep-query thinking configuration", () => {
  afterEach(() => createModelMock.mockReset());

  it("disables thinking for primary and fallback query models while preserving other body fields", async () => {
    createModelMock.mockResolvedValue({});
    await resolveAgentModel({}, "query", configuredEnv({ QUERY_ENABLE_THINKING: "false" }));

    expect(createModelMock.mock.calls.map(([config]) => config.extraBody)).toEqual([
      { max_tokens: 500, chat_template_kwargs: { enable_thinking: false } },
      { max_tokens: 300, chat_template_kwargs: { enable_thinking: false } },
    ]);
  });

  it.each(["mutate", "chat"] as const)(
    "leaves %s model request settings unchanged by query opt-out",
    async (mode) => {
      createModelMock.mockResolvedValue({});
      await resolveAgentModel({}, mode, configuredEnv({ QUERY_ENABLE_THINKING: "false" }));
      expect(createModelMock.mock.calls.map(([config]) => config.extraBody)).toEqual([
        { max_tokens: 500 },
        { max_tokens: 300 },
      ]);
    }
  );

  it("disables thinking for mutation primary and fallback only when opted out", async () => {
    createModelMock.mockResolvedValue({});
    await resolveAgentModel({}, "mutate", configuredEnv({ MUTATION_ENABLE_THINKING: "false" }));

    expect(createModelMock.mock.calls.map(([config]) => config.extraBody)).toEqual([
      { max_tokens: 500, chat_template_kwargs: { enable_thinking: false } },
      { max_tokens: 300, chat_template_kwargs: { enable_thinking: false } },
    ]);
  });

  it("keeps thinking enabled unless explicitly opted out", async () => {
    createModelMock.mockResolvedValue({});
    await resolveAgentModel({}, "query", configuredEnv());
    expect(createModelMock.mock.calls.map(([config]) => config.extraBody)).toEqual([
      { max_tokens: 500 },
      { max_tokens: 300 },
    ]);
  });

  it("records separate numeric provider calls for loop steps and repair", async () => {
    const model = {
      specificationVersion: "v1",
      provider: "private-provider",
      modelId: "private-model",
      doGenerate: vi.fn(async () => ({
        content: [{ type: "text", text: "private response" }],
        usage: { inputTokens: 11, outputTokens: 4 },
      })),
    };
    createModelMock.mockResolvedValue(model);
    const providerCalls: Array<Record<string, unknown>> = [];
    const resolved = await resolveAgentModel({}, "query", {
      LLM_API_BASE_URL: "http://private-host/v1",
      LLM_MODEL: "private-model",
    }, (call) => providerCalls.push(call));
    const wrapped = resolved.model as typeof model;
    await wrapped.doGenerate({} as never);
    await wrapped.doGenerate({} as never);
    await (resolved.synthesisModel as typeof model).doGenerate({} as never);

    expect(providerCalls).toHaveLength(3);
    expect(providerCalls.every((call) =>
      typeof call.durationMs === "number" &&
      call.inputTokens === 11 && call.outputTokens === 4
    )).toBe(true);
    const serialized = JSON.stringify(providerCalls);
    expect(serialized).not.toContain("private response");
    expect(serialized).not.toContain("private-host");
  });

  it("does not mask a provider failure when recording timing", async () => {
    const original = new Error("private provider response");
    const model = {
      specificationVersion: "v1",
      provider: "private-provider",
      modelId: "private-model",
      doGenerate: vi.fn(async () => { throw original; }),
    };
    createModelMock.mockResolvedValue(model);
    const resolved = await resolveAgentModel({}, "query", {
      LLM_API_BASE_URL: "http://private-host/v1",
      LLM_MODEL: "private-model",
    }, () => { throw new Error("telemetry failure"); });
    await expect((resolved.model as typeof model).doGenerate({} as never)).rejects.toBe(original);
  });
});
