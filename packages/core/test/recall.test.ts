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
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  for (const k of ["RECALL", "RECALL_SEEDS", "RECALL_CANDIDATES", "RECALL_MIN_SCORE",
    "RECALL_MAX_OUTPUT_TOKENS", "LLM_THINKING_BUDGET"]) {
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
    const generate = vi.fn(async () => ({
      text: "Fridays, after the review. Sources: /facts/deploy.md",
      finishReason: "stop" as const,
    }));

    const result = await runRecall(kb, "when do we deploy cadence?", {}, generate);

    expect(result.answer).toContain("Fridays");
    expect(result.paths).toContain("/facts/deploy.md");
    expect(generate).toHaveBeenCalledTimes(1);
    const prompt = generate.mock.calls[0][1] as string;
    expect(prompt).toContain("CONCEPT /facts/deploy.md");
    expect(generate.mock.calls[0][3]).toMatchObject({ maxOutputTokens: 2048 });
  });

  it("strips the verdict line from a complete answer", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays, after the review meeting.",
      "add"
    );
    const generate = vi.fn(async () => ({
      text: "SUFFICIENT\nWe deploy on Fridays, after the review meeting.\nSources: /facts/deploy.md",
      finishReason: "stop" as const,
    }));

    const result = await runRecall(kb, "when do we deploy cadence?", {}, generate);

    expect(result.answer).toBe(
      "We deploy on Fridays, after the review meeting.\nSources: /facts/deploy.md"
    );
  });

  // The bug this layer used to have: on llama.cpp thinking tokens are billed
  // against max_tokens, so the cap was hit mid-sentence and the fragment was
  // returned as a success, cached for 24 h, and never escalated. A decline
  // writes no trace, so the warning is the only signal it happened.
  it("declines a truncated answer and warns, keeping the candidates", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays, after the review meeting, and the migration runs overnight.",
      "add"
    );
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "SUFFICIENT\nWe deploy on Fridays, after the review meeting and then the",
      finishReason: "length" as const,
    }));

    const result = await runRecall(kb, "when do we deploy cadence?", {}, generate);

    expect(result.answer).toBeNull();
    expect(result.paths).toContain("/facts/deploy.md"); // still handed to the deep run
    expect(logged).toHaveBeenCalledTimes(1);
    const line = String(logged.mock.calls[0][0]);
    expect(line).toContain("2048");
    expect(line).toContain("RECALL_MAX_OUTPUT_TOKENS");
    expect(line).toContain("when do we deploy cadence?");
  });

  it("names the configured cap in the truncation warning", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    vi.stubEnv("RECALL_MAX_OUTPUT_TOKENS", "640");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "SUFFICIENT\nWe deploy on Fridays, after the review meeting and then the",
      finishReason: "length" as const,
    }));

    expect((await runRecall(kb, "deploy cadence day?", {}, generate)).answer).toBeNull();
    expect(generate).toHaveBeenCalledTimes(1);
    expect(String(logged.mock.calls[0][0])).toContain("640");
  });

  it("declines when the model reports the excerpts are not enough", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const generate = vi.fn(async () => ({ text: "UNKNOWN", finishReason: "stop" as const }));
    const result = await runRecall(kb, "deploy day cadence", {}, generate);
    expect(result.answer).toBeNull();
    expect(result.paths).toContain("/facts/deploy.md"); // still handed to the deep run
  });

  it("does not spend a generation when the literal match is too weak", async () => {
    await kb.writeConcept("/facts/odd.md", { type: "Fact", title: "Odd note" }, "A zebra once passed by.", "add");
    const generate = vi.fn(async () => ({ text: "should not run", finishReason: "stop" as const }));
    // One incidental body mention scores far below the confidence gate.
    const result = await runRecall(kb, "zebra", {}, generate);
    expect(result.answer).toBeNull();
    expect(result.paths).toEqual([]);
    expect(generate).not.toHaveBeenCalled();
  });

  it("does not trust absent filenames or common path fragments", async () => {
    for (const directory of ["repositories", "gotchas", "notes"]) {
      for (const name of ["alpha", "beta", "gamma", "delta"]) {
        await kb.writeConcept(
          `/${directory}/${name}.md`,
          { type: "Note", title: `Unrelated ${name}` },
          "Routine material.",
          "add"
        );
      }
    }
    const generate = vi.fn(async () => ({ text: "should not run", finishReason: "stop" as const }));

    const absent = await runRecall(kb, "Where is completely-absent.md?", {}, generate);
    const common = await runRecall(kb, "repositories/md", {}, generate);
    const otherCommon = await runRecall(kb, "gotchas/md", {}, generate);

    expect(absent).toEqual({ answer: null, paths: [] });
    expect(common).toEqual({ answer: null, paths: [] });
    expect(otherCommon).toEqual({ answer: null, paths: [] });
    expect(generate).not.toHaveBeenCalled();
  });

  it("keeps both Sniff installation concepts in recall candidates", async () => {
    await kb.writeConcept(
      "/gotchas/ptr-ms-analysis-pipx-installation.md",
      { type: "Gotcha", title: "PTR-MS/Sniff pipx installation" },
      "Install Sniff with the documented branch workflow.",
      "add"
    );
    await kb.writeConcept(
      "/decisions/ptr-ms-analysis-work-on-main.md",
      { type: "Gotcha", title: "PTR-MS analysis work on main" },
      "Sniff installation and work on main are documented here.",
      "add"
    );
    await kb.writeConcept(
      "/notes/unrelated-install.md",
      { type: "Note", title: "General installation" },
      "This does not mention the project.",
      "add"
    );

    const generate = vi.fn(async () => ({ text: "SUFFICIENT\nThe workflow is documented.", finishReason: "stop" as const }));
    const result = await runRecall(kb, "How do I install Sniff?", {}, generate);

    expect(result.paths.slice(0, 3)).toEqual(
      expect.arrayContaining([
        "/gotchas/ptr-ms-analysis-pipx-installation.md",
        "/decisions/ptr-ms-analysis-work-on-main.md",
      ])
    );
    expect(generate).toHaveBeenCalledTimes(1);
  });

  it("diversifies compound intents within the default candidate budget", async () => {
    await kb.writeConcept(
      "/gotchas/ptr-ms-analysis-pipx-installation.md",
      { type: "Gotcha", title: "PTR-MS/Sniff pipx installation" },
      "Install the editable checkout with pipx so the Sniff launcher uses the current package; the branch/install workflow is documented here.",
      "add"
    );
    await kb.writeConcept(
      "/decisions/ptr-ms-analysis-work-on-main.md",
      { type: "Gotcha", title: "PTR-MS analysis work on main" },
      "Work directly on main when testing Sniff changes; the branch convention is documented here.",
      "add"
    );

    await kb.writeConcept(
      "/notes/generic-installer.md",
      { type: "Note", title: "Generic application installer" },
      "General installer documentation explains how to install, open, and test a current desktop application from a repository.",
      "add"
    );
    await kb.writeConcept(
      "/notes/unrelated-branch.md",
      { type: "Note", title: "Unrelated branch conventions" },
      "General branch documentation explains how to test current changes and keep a repository on its active branch.",
      "add"
    );

    await Promise.all(
      Array.from({ length: 120 }, async (_, index) => {
        const directory = ["repositories", "gotchas", "notes"][index % 3];
        const filename = `${directory}/distractor-${index}.md`;
        const absolute = path.join(root, filename);
        await fs.mkdir(path.dirname(absolute), { recursive: true });
        await fs.writeFile(
          absolute,
          `---\ntype: Note\ntitle: Desktop workflow ${index}\n---\n` +
            "General application installation, testing, and branch conventions.\n"
        );
      })
    );

    const question =
      "How is the current Sniff desktop application installed locally from the " +
      "ptr-ms/sniff repository so Dan can test changes by opening Sniff, and what " +
      "branch/install conventions have been used?";
    const searched = vi.spyOn(kb, "search");
    const generate = vi.fn(async () => ({ text: "SUFFICIENT\nThe workflow is documented.", finishReason: "stop" as const }));

    const result = await runRecall(kb, question, {}, generate);

    expect(result.paths.length).toBeLessThanOrEqual(6);
    expect(result.paths.slice(0, 6)).toEqual(
      expect.arrayContaining([
        "/gotchas/ptr-ms-analysis-pipx-installation.md",
        "/decisions/ptr-ms-analysis-work-on-main.md",
      ])
    );
    expect(searched.mock.calls.length).toBeLessThanOrEqual(5);
    expect(searched.mock.calls.map(([query]) => query)).toEqual(
      expect.arrayContaining(["ptr-ms install", "ptr-ms branch"])
    );
    const prompt = generate.mock.calls[0][1] as string;
    expect(prompt).toContain("CONCEPT /gotchas/ptr-ms-analysis-pipx-installation.md");
    expect(prompt).toContain("CONCEPT /decisions/ptr-ms-analysis-work-on-main.md");
  });

  it("does not expand simple or empty queries", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const searched = vi.spyOn(kb, "search");
    const generate = vi.fn(async () => ({ text: "Fridays.", finishReason: "stop" as const }));

    await runRecall(kb, "deploy cadence day?", {}, generate);
    await runRecall(kb, "", {}, generate);

    expect(searched).toHaveBeenCalledTimes(2);
  });

  it("keeps path-only expansion hits behind the confidence gate", async () => {
    await kb.writeConcept(
      "/facts/acme-policy.md",
      { type: "Fact", title: "Acme installation and branch policy" },
      "Acme installation and branch policy for the current project.",
      "add"
    );
    await kb.writeConcept(
      "/projects/acme/release.md",
      { type: "Note", title: "Acme release" },
      "Routine release notes.",
      "add"
    );
    vi.stubEnv("RECALL_SEEDS", "1");
    const searched = vi.spyOn(kb, "search");
    const generate = vi.fn(async () => ({ text: "UNKNOWN", finishReason: "stop" as const }));

    const result = await runRecall(
      kb,
      "How do I install and branch from acme/release?",
      {},
      generate
    );

    expect(result.answer).toBeNull();
    expect(result.paths).toContain("/facts/acme-policy.md");
    expect(result.paths).not.toContain("/projects/acme/release.md");
    expect(generate).toHaveBeenCalledTimes(1);
    expect(searched.mock.calls.length).toBeGreaterThan(1);
    expect(searched.mock.calls.length).toBeLessThanOrEqual(5);
  });

  it("keeps the logo concept selected after intent expansion", async () => {
    await kb.writeConcept(
      "/notes/euroeval-visual-identity.md",
      { type: "Note", title: "EuroEval visual identity" },
      "The official EuroEval logo artwork is gfx/euroeval.png.",
      "add"
    );
    await Promise.all(
      Array.from({ length: 80 }, async (_, index) => {
        const filename = `notes/logo-distractor-${index}.md`;
        const absolute = path.join(root, filename);
        await fs.mkdir(path.dirname(absolute), { recursive: true });
        await fs.writeFile(
          absolute,
          `---\ntype: Note\ntitle: General project record ${index}\n---\n` +
            "Generic project release and documentation record.\n"
        );
      })
    );
    const generate = vi.fn(async () => ({ text: "SUFFICIENT\nThe logo is documented.", finishReason: "stop" as const }));

    const result = await runRecall(kb, "Where is the EuroEval logo artwork?", {}, generate);

    expect(result.paths.slice(0, 6)).toContain("/notes/euroeval-visual-identity.md");
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
    const generate = vi.fn(async () => ({
      text: "It walks one hop of the link graph. Sources: /facts/b.md",
      finishReason: "stop" as const,
    }));

    const result = await runRecall(kb, "how does retrieval work?", {}, generate);

    expect(result.paths).toContain("/facts/b.md");
    expect((generate.mock.calls[0][1] as string)).toContain("Recall widens by walking");
  });

  it("is disabled by RECALL=false", async () => {
    await kb.writeConcept("/facts/deploy.md", { type: "Fact", title: "Deploy" }, "Fridays.", "add");
    process.env.RECALL = "false";
    const generate = vi.fn(async () => ({ text: "should not run", finishReason: "stop" as const }));
    expect((await runRecall(kb, "deploy?", {}, generate)).answer).toBeNull();
    expect(generate).not.toHaveBeenCalled();
  });

  it("asks for the 2048-token default cap when the env var is unset", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const generate = vi.fn(async () => ({ text: "Fridays.", finishReason: "stop" as const }));

    await runRecall(kb, "deploy cadence day?", {}, generate);

    expect(generate).toHaveBeenCalledTimes(1);
    expect(generate.mock.calls[0][3].maxOutputTokens).toBe(2048);
  });

  // The silent-success sibling of the truncation bug: "other" is the bucket a
  // provider that never reports a finish reason lands on, so declining here
  // would turn recall off for that whole deployment without a word. It answers,
  // and the log line is what stops it from being silent.
  it("answers an unrecognised finish reason and says so once", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "Fridays, after the review. Sources: /facts/deploy.md",
      finishReason: "other" as const,
    }));

    const result = await runRecall(kb, "when do we deploy cadence?", {}, generate);

    expect(result.answer).toContain("Fridays");
    expect(logged).toHaveBeenCalledTimes(1);
    expect(String(logged.mock.calls[0][0])).toContain("[understory]");
  });

  // The shape where reasoning ate the whole cap: nothing at all came back
  // before the stop. Still a decline, and never an empty answer.
  it("declines an empty completion cut off by the cap", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({ text: "", finishReason: "length" as const }));

    const result = await runRecall(kb, "deploy cadence day?", {}, generate);

    expect(result.answer).toBeNull();
    expect(result.paths).toContain("/facts/deploy.md");
    expect(generate).toHaveBeenCalledTimes(1);
    expect(String(logged.mock.calls[0][0])).toContain("RECALL_MAX_OUTPUT_TOKENS");
  });

  it("logs nothing on a generation that completed", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({ text: "Fridays.", finishReason: "stop" as const }));

    expect((await runRecall(kb, "deploy cadence day?", {}, generate)).answer).toBe("Fridays.");
    expect(logged).not.toHaveBeenCalled();
  });

  // intEnv accepts 0, and max_tokens 0 is a request every completion dies on:
  // one doomed generation per query, i.e. the layer off with extra latency.
  it("falls back to the default cap when RECALL_MAX_OUTPUT_TOKENS is not positive", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    for (const raw of ["0", "-1"]) {
      vi.stubEnv("RECALL_MAX_OUTPUT_TOKENS", raw);
      const generate = vi.fn(async () => ({ text: "Fridays.", finishReason: "stop" as const }));

      expect((await runRecall(kb, "deploy cadence day?", {}, generate)).answer).toBe("Fridays.");
      expect(generate.mock.calls[0][3].maxOutputTokens).toBe(2048);
    }
  });
  it("honours RECALL_MAX_OUTPUT_TOKENS", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays.",
      "add"
    );
    vi.stubEnv("RECALL_MAX_OUTPUT_TOKENS", "640");
    const generate = vi.fn(async () => ({ text: "Fridays.", finishReason: "stop" as const }));

    await runRecall(kb, "deploy cadence day?", {}, generate);

    expect(generate).toHaveBeenCalledTimes(1);
    expect(generate.mock.calls[0][3].maxOutputTokens).toBe(640);
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
    const generate = vi.fn(async () => ({
      text: "Fridays. Sources: /facts/deploy.md",
      finishReason: "stop" as const,
    }));

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

  it("answers from the deep agent when recall is truncated, and caches that", async () => {
    await kb.writeConcept(
      "/facts/deploy.md",
      { type: "Fact", title: "Deploy day", description: "weekly deploy cadence" },
      "We deploy on Fridays, after the review meeting.",
      "add"
    );
    const runner = deep("Fridays, after the review meeting, then the migration runs.");
    const logged = vi.spyOn(console, "error").mockImplementation(() => {});
    const generate = vi.fn(async () => ({
      text: "SUFFICIENT\nWe deploy on Fridays, after the review meeting and then the",
      finishReason: "length" as const,
    }));
    const question = "deploy cadence day?";

    const result = await runQueryCached(kb, question, {}, runner, noHot, (k, q, o) =>
      runRecall(k, q, o, generate)
    );

    // The fragment is neither the answer nor cached: the deep run answers.
    expect(result.source).toBe("deep");
    expect(result.answer).toContain("migration runs");
    expect(runner).toHaveBeenCalledTimes(1);
    expect(runner.mock.calls[0][1]).toContain("/facts/deploy.md");

    const again = await runQueryCached(kb, question, {}, runner, noHot, (k, q, o) =>
      runRecall(k, q, o, generate)
    );
    expect(again.cached).toBe(true);
    expect(again.answer).toBe(result.answer);
  });
});

describe("read_concepts", () => {
  it("reads several concepts in one call", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "body a", "add");
    await kb.writeConcept("/facts/b.md", { type: "Fact", title: "B" }, "body b", "add");

    const tools = buildReadTools(kb);
    const out = await tools.read_concepts!.execute!(
      { paths: ["facts/a.md", "//facts/b.md", "/facts/gone.md"] },
      { toolCallId: "c", messages: [] }
    );

    expect((out as { read: Array<{ path: string }> }).read.map((page) => page.path)).toEqual([
      "/facts/a.md",
      "/facts/b.md",
    ]);
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
