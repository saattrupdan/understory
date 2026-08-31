import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { KnowledgeBase } from "../src/okf/index.js";
import { runRecall } from "../src/agent/recall.js";
import { clearHotMemory } from "../src/agent/hot-memory.js";
import { clearQueryCache, runQueryCached, withCandidateHint } from "../src/agent/query-cache.js";
import { buildReadTools } from "../src/agent/tools.js";
import { resolveModelConfig, thinkingBudgetBody } from "../src/providers/index.js";
import type { QueryResult } from "../src/agent/agent.js";

let root: string;
let kb: KnowledgeBase;

beforeEach(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-recall-"));
  kb = new KnowledgeBase(root);
  clearHotMemory();
  clearQueryCache();
});

afterEach(async () => {
  await fs.rm(root, { recursive: true, force: true });
  for (const k of ["RECALL", "RECALL_SEEDS", "RECALL_CANDIDATES", "RECALL_MIN_SCORE",
    "LLM_THINKING_BUDGET"]) {
    delete process.env[k];
  }
});

describe("runRecall", () => {
  it("answers from retrieved concepts in a single generation", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays, after the review meeting.",
      "add"
    );
    const generate = vi.fn(async () => "Fridays, after the review. Sources: /facts/deploy.md");

    const result = await runRecall(kb, "when do we deploy cadence?", {}, generate);

    expect(result.answer).toContain("Fridays");
    expect(result.paths).toContain("/facts/deploy.md");
    expect(generate).toHaveBeenCalledTimes(1);
    const prompt = generate.mock.calls[0][1] as string;
    expect(prompt).toContain("CONCEPT /facts/deploy.md");
    expect(generate.mock.calls[0][3]).toMatchObject({ maxOutputTokens: 900 });
  });

  it("declines when the model reports the excerpts are not enough", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const generate = vi.fn(async () => "UNKNOWN");
    const result = await runRecall(kb, "deploy day cadence", {}, generate);
    expect(result.answer).toBeNull();
    expect(result.paths).toContain("/facts/deploy.md"); // still handed to the deep run
  });

  it("does not spend a generation when the literal match is too weak", async () => {
    await kb.writeConcept("/facts/odd.md", { type: "Fact", title: "Odd note" }, "A zebra once passed by.", "add");
    const generate = vi.fn(async () => "should not run");
    // One incidental body mention scores far below the confidence gate.
    const result = await runRecall(kb, "zebra", {}, generate);
    expect(result.answer).toBeNull();
    expect(result.paths).toEqual([]);
    expect(generate).not.toHaveBeenCalled();
  });

  it("widens to linked concepts the keywords never named", async () => {
    // The answer lives in a concept the question shares no words with; only the
    // link out of the matched concept can reach it.
    await kb.writeConcept(
      "/facts/a.md",
      { type: "Fact", title: "Retrieval note", description: "how retrieval works" },
      "Retrieval is described further in [the recall design](/facts/b.md).",
      "add"
    );
    await kb.writeConcept(
      "/facts/b.md",
      { type: "Fact", title: "Recall design" },
      "Recall widens by walking one hop of the link graph.",
      "add"
    );
    const generate = vi.fn(async () => "It walks one hop of the link graph. Sources: /facts/b.md");

    const result = await runRecall(kb, "how does retrieval work?", {}, generate);

    expect(result.paths).toContain("/facts/b.md");
    expect((generate.mock.calls[0][1] as string)).toContain("Recall widens by walking");
  });

  it("is disabled by RECALL=false", async () => {
    await kb.writeConcept("/facts/deploy.md", { type: "Fact", title: "Deploy" }, "Fridays.", "add");
    process.env.RECALL = "false";
    const generate = vi.fn(async () => "should not run");
    expect((await runRecall(kb, "deploy?", {}, generate)).answer).toBeNull();
    expect(generate).not.toHaveBeenCalled();
  });
});

describe("runQueryCached layer order", () => {
  function deep(answer: string) {
    return vi.fn(async (): Promise<QueryResult> => ({ answer, steps: 9, traceId: "t" }));
  }
  const noHot = async () => null;

  it("answers from recall without starting the deep agent", async () => {
    const runner = deep("deep answer");
    const recall = vi.fn(async () => ({ answer: "recall answer", paths: ["/facts/a.md"] }));

    const result = await runQueryCached(kb, "q?", {}, runner, noHot, recall);

    expect(result.source).toBe("recall");
    expect(result.answer).toBe("recall answer");
    expect(runner).not.toHaveBeenCalled();
    // The repeat is now cached, and recall answered it without a deep run.
    const again = await runQueryCached(kb, "q?", {}, runner, noHot, recall);
    expect(again.cached).toBe(true);
    expect(recall).toHaveBeenCalledTimes(1);
  });

  it("escalates to the deep agent when recall declines, and hands it the candidates", async () => {
    const runner = deep("deep answer");
    const recall = vi.fn(async () => ({ answer: null, paths: ["/facts/a.md", "/facts/b.md"] }));

    const result = await runQueryCached(kb, "q?", {}, runner, noHot, recall);

    expect(result.source).toBe("deep");
    expect(result.answer).toBe("deep answer");
    expect(recall).toHaveBeenCalledTimes(1);
    expect(runner).toHaveBeenCalledTimes(1);
    // The declined retrieval is not thrown away: the deep run is told to read it.
    expect(runner.mock.calls[0][1]).toContain("/facts/a.md");
    expect(runner.mock.calls[0][1]).toContain("q?");
  });

  it("traces a recall answer so its latency is attributable", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const runner = deep("should not run");
    const generate = vi.fn(async () => "Fridays. Sources: /facts/deploy.md");

    const result = await runQueryCached(
      kb,
      "deploy cadence day?",
      {},
      runner,
      noHot,
      (k, q, o) => runRecall(k, q, o, generate)
    );

    expect(result.source).toBe("recall");
    const stored = await new (await import("../src/agent/trace.js")).TraceStore(root).list();
    const trace = stored.find((t) => t.id === result.traceId);
    expect(trace?.notation).toContain("recall");
    expect(runner).not.toHaveBeenCalled();
  });
});

describe("read_concepts", () => {
  it("reads several concepts in one call", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "body a", "add");
    await kb.writeConcept("/facts/b.md", { type: "Fact", title: "B" }, "body b", "add");

    const tools = buildReadTools(kb);
    const out = await tools.read_concepts!.execute!(
      { paths: ["/facts/a.md", "/facts/b.md", "/facts/gone.md"] },
      { toolCallId: "c", messages: [] }
    );

    expect((out as { read: unknown[] }).read).toHaveLength(2);
    expect((out as { missing: string[] }).missing).toEqual(["/facts/gone.md"]);
  });
});

describe("withCandidateHint", () => {
  it("leaves the question untouched when nothing was found", () => {
    expect(withCandidateHint("q?", [])).toBe("q?");
  });
});

describe("thinking budget wiring", () => {
  it("builds the vLLM request body for a reasoning budget", () => {
    expect(thinkingBudgetBody(256)).toEqual({
      chat_template_kwargs: { thinking: true, thinking_budget: 256 },
    });
    expect(thinkingBudgetBody(NaN)).toEqual({});
  });

  it("picks the budget up from the environment", () => {
    const base = { LLM_API_FORMAT: "openai", LLM_API_BASE_URL: "http://x/v1", LLM_MODEL: "m" };
    expect(resolveModelConfig(base).extraBody).toBeUndefined();
    expect(resolveModelConfig({ ...base, LLM_THINKING_BUDGET: "300" }).extraBody).toEqual({
      chat_template_kwargs: { thinking: true, thinking_budget: 300 },
    });
    expect(resolveModelConfig({ ...base, LLM_MAX_OUTPUT_TOKENS: "1200" }).extraBody).toEqual({
      max_tokens: 1200,
    });
  });
});
