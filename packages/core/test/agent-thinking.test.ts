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
    "leaves %s model request settings unchanged",
    async (mode) => {
      createModelMock.mockResolvedValue({});
      await resolveAgentModel({}, mode, configuredEnv({ QUERY_ENABLE_THINKING: "false" }));
      expect(createModelMock.mock.calls.map(([config]) => config.extraBody)).toEqual([
        { max_tokens: 500 },
        { max_tokens: 300 },
      ]);
    }
  );

  it("keeps thinking enabled unless explicitly opted out", async () => {
    createModelMock.mockResolvedValue({});
    await resolveAgentModel({}, "query", configuredEnv());
    expect(createModelMock.mock.calls.map(([config]) => config.extraBody)).toEqual([
      { max_tokens: 500 },
      { max_tokens: 300 },
    ]);
  });
});
