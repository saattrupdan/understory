import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { KnowledgeBase } from "../src/okf/index.js";
import {
  clearHotMemory,
  hotLookup,
  type HotGenerate,
  recordHotDelete,
  recordHotQuery,
  recordHotWrite,
} from "../src/agent/hot-memory.js";
import { clearQueryCache, runQueryCached } from "../src/agent/query-cache.js";
import type { AgentOptions, QueryResult } from "../src/agent/agent.js";

let root: string;
let kb: KnowledgeBase;

beforeEach(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-hot-"));
  kb = new KnowledgeBase(root);
  clearHotMemory();
  clearQueryCache();
});

afterEach(async () => {
  await fs.rm(root, { recursive: true, force: true });
  delete process.env.HOT_MEMORY;
  delete process.env.HOT_MEMORY_TTL;
  delete process.env.HOT_MEMORY_MAX_OUTPUT_TOKENS;
  vi.restoreAllMocks();
});

describe("hotLookup", () => {
  it("returns null without calling the model when the hot set is empty", async () => {
    const generate = vi.fn();
    expect(await hotLookup(kb, "anything?", {}, generate)).toBeNull();
    expect(generate).not.toHaveBeenCalled();
  });

  it("answers from recently written concepts (read fresh) and falls through on UNKNOWN", async () => {
    await kb.writeConcept("/facts/deploy.md", { type: "Fact", title: "Deploy day" }, "We deploy on Fridays.", "add");
    recordHotWrite("/facts/deploy.md");

    const confident = vi.fn(async () => ({
      text: "We deploy on Fridays. (from /facts/deploy.md)",
      finishReason: "stop" as const,
    }));
    const answer = await hotLookup(kb, "when do we deploy?", {}, confident);
    expect(answer).toContain("Fridays");
    const prompt = confident.mock.calls[0][1] as string;
    expect(prompt).toContain("/facts/deploy.md");
    expect(prompt).toContain("We deploy on Fridays.");

    const unsure = vi.fn(async () => ({ text: "UNKNOWN", finishReason: "stop" as const }));
    expect(await hotLookup(kb, "what is the capital of France?", {}, unsure)).toBeNull();
  });

  it("purges hot Q&As on writes and drops deleted concepts", async () => {
    recordHotQuery("q1", "a1");
    recordHotWrite("/facts/x.md"); // any write invalidates prior answers
    const generate = vi.fn(async () => ({ text: "should not matter", finishReason: "stop" as const }));
    // /facts/x.md doesn't exist on disk → dropped at read; Q&As purged → empty set → null.
    expect(await hotLookup(kb, "q1", {}, generate)).toBeNull();
    expect(generate).not.toHaveBeenCalled();

    recordHotDelete("/facts/x.md");
    expect(await hotLookup(kb, "q1", {}, generate)).toBeNull();
  });

  it("expires entries after HOT_MEMORY_TTL and respects HOT_MEMORY=false", async () => {
    vi.useFakeTimers();
    try {
      process.env.HOT_MEMORY_TTL = "1m";
      await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "alpha", "add");
      recordHotWrite("/facts/a.md");
      vi.advanceTimersByTime(61_000);
      const generate = vi.fn();
      expect(await hotLookup(kb, "a?", {}, generate)).toBeNull();
      expect(generate).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }

    process.env.HOT_MEMORY = "false";
    await kb.writeConcept("/facts/b.md", { type: "Fact", title: "B" }, "beta", "add");
    recordHotWrite("/facts/b.md");
    const generate = vi.fn();
    expect(await hotLookup(kb, "b?", {}, generate)).toBeNull();
    expect(generate).not.toHaveBeenCalled();
  });

  // The bug the recall layer had and this one still did, only worse: hot runs
  // first, so a reply cut off by the output cap answered the query, and
  // query-cache.ts cached the fragment for 24 h without writing a trace. With
  // no cap sent at all the only bound was the server's own limit.
  it("declines a truncated answer and names the cap", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day" },
      "We deploy on Fridays, after the review meeting.",
      "add"
    );
    recordHotWrite("/facts/deploy.md");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "We deploy on Fridays, after the review meeting and then the",
      finishReason: "length" as const,
    }));

    expect(await hotLookup(kb, "when do we deploy?", {}, generate)).toBeNull();
    expect(generate).toHaveBeenCalledTimes(1);
    const line = String(logged.mock.calls[0][0]);
    expect(line).toContain("[understory]");
    expect(line).toContain("2048");
    expect(line).toContain("HOT_MEMORY_MAX_OUTPUT_TOKENS");
  });

  // Same rule as the recall layer: "other" is where a provider that never
  // reports a finish reason lands, so it answers — with one log line.
  it("answers an unrecognised finish reason and says so once", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "alpha", "add");
    recordHotWrite("/facts/a.md");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "alpha (from /facts/a.md)",
      finishReason: "other" as const,
    }));

    expect(await hotLookup(kb, "what is a?", {}, generate)).toContain("alpha");
    expect(logged).toHaveBeenCalledTimes(1);
    expect(String(logged.mock.calls[0][0])).toContain("[understory]");
  });

  it("sends an explicit output cap, overridable per layer", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "alpha", "add");
    recordHotWrite("/facts/a.md");

    const unconfigured = vi.fn<HotGenerate>(async () => ({
      text: "alpha",
      finishReason: "stop",
    }));
    await hotLookup(kb, "a?", {}, unconfigured);
    expect(unconfigured.mock.calls[0][3]).toMatchObject({ maxOutputTokens: 2048 });

    process.env.HOT_MEMORY_MAX_OUTPUT_TOKENS = "640";
    const configured = vi.fn<HotGenerate>(async () => ({
      text: "alpha",
      finishReason: "stop",
    }));
    await hotLookup(kb, "a?", {}, configured);
    expect(configured.mock.calls[0][3]).toMatchObject({ maxOutputTokens: 640 });

    // A cap of 0 is a request no completion survives, not a cap.
    process.env.HOT_MEMORY_MAX_OUTPUT_TOKENS = "0";
    const zero = vi.fn<HotGenerate>(async () => ({ text: "alpha", finishReason: "stop" }));
    await hotLookup(kb, "a?", {}, zero);
    expect(zero.mock.calls[0][3]).toMatchObject({ maxOutputTokens: 2048 });
  });

  it("logs nothing on a generation that completed", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "alpha", "add");
    recordHotWrite("/facts/a.md");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({ text: "alpha", finishReason: "stop" as const }));

    expect(await hotLookup(kb, "a?", {}, generate)).toBe("alpha");
    expect(logged).not.toHaveBeenCalled();
  });
});

describe("runQueryCached layering", () => {
  it("hot answers short-circuit the deep agent and land in the exact cache", async () => {
    const deep = vi.fn(async (): Promise<QueryResult> => ({ answer: "deep", steps: 5, traceId: "t" }));
    const hot = vi.fn(async () => "hot answer");

    const first = await runQueryCached(kb, "q?", {}, deep, hot);
    expect(first.source).toBe("hot");
    expect(first.answer).toBe("hot answer");
    expect(deep).not.toHaveBeenCalled();

    // Identical repeat: exact cache now answers, hot not consulted again.
    const second = await runQueryCached(kb, "q?", {}, deep, hot);
    expect(second.source).toBe("cache");
    expect(hot).toHaveBeenCalledTimes(1);
  });

  it("deep answers feed the hot working set", async () => {
    const deep = vi.fn(async (): Promise<QueryResult> => ({ answer: "42", steps: 3, traceId: "t" }));
    await runQueryCached(kb, "meaning of life?", {}, deep, async () => null);

    // The recorded Q&A is now available to a real hot lookup.
    const generate = vi.fn(async () => ({
      text: "42 (from previous answer)",
      finishReason: "stop" as const,
    }));
    const answer = await hotLookup(kb, "what was the meaning of life again?", {}, generate);
    expect(answer).toContain("42");
    const prompt = generate.mock.calls[0][1] as string;
    expect(prompt).toContain("meaning of life?");
  });

  // Hot answers first, so a truncated hot reply used to short-circuit the two
  // layers below it that are built to handle exactly that. It must fall through
  // instead, and above all must not be what lands in the exact cache.
  it("a truncated hot answer falls through to the next layers", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day" },
      "We deploy on Fridays, after the review meeting.",
      "add"
    );
    recordHotWrite("/facts/deploy.md");
    const deep = vi.fn(async (): Promise<QueryResult> => ({
      answer: "Fridays, after the review meeting, then the migration runs.",
      steps: 7,
      traceId: "t",
    }));
    const recall = vi.fn(async () => ({ answer: null, paths: [] as string[] }));
    vi.spyOn(console, "error").mockImplementation(() => {});
    const truncated = vi.fn(async () => ({
      text: "We deploy on Fridays, after the review meeting and then the",
      finishReason: "length" as const,
    }));
    const hot = (k: KnowledgeBase, q: string, o: AgentOptions = {}) =>
      hotLookup(k, q, o, truncated);
    const question = "when do we deploy?";

    const result = await runQueryCached(kb, question, {}, deep, hot, recall);

    expect(result.source).toBe("deep");
    expect(result.answer).toContain("migration runs");
    expect(truncated).toHaveBeenCalledTimes(1);
    expect(deep).toHaveBeenCalledTimes(1);

    // The repeat is served from the deep answer, not from the fragment.
    const again = await runQueryCached(kb, question, {}, deep, hot, recall);
    expect(again.cached).toBe(true);
    expect(again.answer).toBe(result.answer);
    expect(truncated).toHaveBeenCalledTimes(1);
  });
});
