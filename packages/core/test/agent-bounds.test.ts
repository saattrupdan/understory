import { describe, expect, it, afterEach, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { KnowledgeBase } from "../src/okf/index.js";
import { buildSystemPrompt } from "../src/agent/system-prompt.js";
import { prepareFinalSynthesisStep } from "../src/agent/agent.js";
import { buildReadTools, buildWriteTools, formatTree } from "../src/agent/tools.js";
import { AgentRunContext } from "../src/agent/run-context.js";
import {
  DEFAULT_AGENT_MAX_DOCUMENT_CHARS,
  DEFAULT_AGENT_MAX_INPUT_CHARS,
  DEFAULT_AGENT_MAX_STEPS,
  DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
  DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
  EXHAUSTION_SERIALISED_LENGTH,
  MIN_AGENT_MAX_STEPS,
  resolveAgentLimits,
} from "../src/agent/limits.js";
import type { TreeNode } from "../src/okf/types.js";

const toolContext = { toolCallId: "test", messages: [] };

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("agent context bounds", () => {
  it("uses safe defaults and rejects invalid environment values", () => {
    expect(resolveAgentLimits({})).toEqual({
      maxSteps: DEFAULT_AGENT_MAX_STEPS,
      maxDocumentChars: DEFAULT_AGENT_MAX_DOCUMENT_CHARS,
      maxToolResultChars: DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
      maxSystemContextChars: DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
      maxInputChars: DEFAULT_AGENT_MAX_INPUT_CHARS,
    });
    expect(
      resolveAgentLimits({
        AGENT_MAX_STEPS: "0",
        AGENT_MAX_DOCUMENT_CHARS: "-1",
        AGENT_MAX_TOOL_RESULT_CHARS: "not-a-number",
      })
    ).toEqual({
      maxSteps: DEFAULT_AGENT_MAX_STEPS,
      maxDocumentChars: DEFAULT_AGENT_MAX_DOCUMENT_CHARS,
      maxToolResultChars: DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
      maxSystemContextChars: DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
      maxInputChars: DEFAULT_AGENT_MAX_INPUT_CHARS,
    });
    expect(resolveAgentLimits({ AGENT_MAX_STEPS: "1" }).maxSteps).toBe(MIN_AGENT_MAX_STEPS);
    expect(
      resolveAgentLimits({
        AGENT_MAX_STEPS: "3",
        AGENT_MAX_DOCUMENT_CHARS: "400",
        AGENT_MAX_TOOL_RESULT_CHARS: "900",
        AGENT_MAX_SYSTEM_CONTEXT_CHARS: "700",
      })
    ).toEqual({
      maxSteps: 3,
      maxDocumentChars: 400,
      maxToolResultChars: 900,
      maxSystemContextChars: 700,
      maxInputChars: DEFAULT_AGENT_MAX_INPUT_CHARS,
    });
  });

  it("reserves the final step for synthesis", () => {
    const prepare = prepareFinalSynthesisStep(2);
    expect(prepare({ stepNumber: 0 })).toBeUndefined();
    expect(prepare({ stepNumber: 1 })).toEqual({ activeTools: [] });
  });

  it("pages a concept and reports truncation metadata", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/long.md", { type: "Fact", title: "Long" }, "0123456789", "add");
      vi.stubEnv("AGENT_MAX_DOCUMENT_CHARS", "4");
      const tools = buildReadTools(kb);

      const first = (await tools.read_concept!.execute!(
        { path: "/facts/long.md", offset: 4 },
        toolContext
      )) as { body: string; offset: number; total_chars: number; truncated: boolean; next_offset: number | null };
      expect(first).toMatchObject({
        body: "4567",
        offset: 4,
        total_chars: 11,
        truncated: true,
        next_offset: 8,
      });

      const last = (await tools.read_concept!.execute!(
        { path: "/facts/long.md", offset: first.next_offset ?? 0 },
        toolContext
      )) as { body: string; truncated: boolean; next_offset: number | null };
      expect(last).toMatchObject({ body: "89\n", truncated: false, next_offset: null });
      await expect(
        tools.read_concept!.execute!({ path: "/facts/long.md", offset: 12 }, toolContext)
      ).resolves.toMatchObject({
        path: "/facts/long.md",
        read: false,
        error: { code: "invalid_offset" },
      });
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("never returns undefined after exhausting a paged run", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const body = "0123456789".repeat(2_600) + "\n";
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/very-long.md", { type: "Fact" }, body, "add");
      const state = new AgentRunContext({
        maxSteps: 8,
        maxDocumentChars: 12_000,
        maxToolResultChars: DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
        maxSystemContextChars: DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
      });
      const tools = buildReadTools(kb, undefined, state);
      let offset = 0;
      let exhausted = false;
      for (let call = 0; call < 5; call += 1) {
        const page = await tools.read_concept!.execute!({ path: "/facts/very-long.md", offset }, toolContext);
        expect(page).not.toBeUndefined();
        if (typeof page === "string") {
          exhausted = true;
          break;
        }
        expect((page as ReadPage).offset).toBe(offset);
        if ((page as ReadPage).next_offset === null) break;
        offset = (page as ReadPage).next_offset!;
      }
      expect(exhausted).toBe(true);
      expect(offset).toBeGreaterThan(0);
      expect(offset).toBeLessThan(body.length + 1);

      const continuation = buildReadTools(
        kb,
        undefined,
        new AgentRunContext({
          maxSteps: 8,
          maxDocumentChars: 12_000,
          maxToolResultChars: DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
          maxSystemContextChars: DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
        })
      );
      const next = await continuation.read_concept!.execute!({
        path: "/facts/very-long.md",
        offset,
      }, toolContext) as ReadPage;
      expect(next.body).toBe(body.slice(offset, offset + next.body.length));
      expect(next.offset).toBe(offset);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("returns partial search results and visible list truncation markers", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      for (let index = 0; index < 30; index += 1) {
        await kb.writeConcept(
          `/facts/${index}.md`,
          { type: "Fact", title: `Fact ${index}` },
          `shared keyword ${index}`,
          "add"
        );
      }
      const limits = {
        maxSteps: 8,
        maxDocumentChars: 1_000,
        maxToolResultChars: 500,
        maxSystemContextChars: 240,
      };
      const search = await buildReadTools(kb, undefined, new AgentRunContext(limits))
        .search_knowledge!.execute!({ query: "shared" }, toolContext) as {
          truncated: boolean;
          hits: unknown[];
        };
      expect(search.truncated).toBe(true);
      expect(search.hits.length).toBeGreaterThan(0);

      const listing = await buildReadTools(kb, undefined, new AgentRunContext(limits))
        .list_directory!.execute!({}, toolContext) as string;
      expect(listing).toContain("... [truncated; total_chars=");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("keeps populated system context explicit at degenerate configured sizes", () => {
    const state = new AgentRunContext({
      maxSteps: 8,
      maxDocumentChars: 1,
      maxToolResultChars: 1,
      maxSystemContextChars: 1,
    });
    const types = state.systemTypes(["type-" + "x".repeat(10_000)]);
    const tree = state.systemTree("node ".repeat(10_000));

    expect(types.join(", ")).toContain("system types truncated");
    expect(tree).toContain("system tree truncated");
    expect(types.join(", ").length + tree.length).toBeLessThanOrEqual(240);
  });

  it("caps aggregate multi-read output as a serialised payload", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "abcdefghij", "add");
      await kb.writeConcept("/facts/b.md", { type: "Fact", title: "B" }, "klmnopqrst", "add");
      vi.stubEnv("AGENT_MAX_DOCUMENT_CHARS", "8");
      vi.stubEnv("AGENT_MAX_TOOL_RESULT_CHARS", "500");
      const tools = buildReadTools(kb);
      const out = (await tools.read_concepts!.execute!(
        { paths: ["/facts/a.md", "/facts/b.md"] },
        toolContext
      )) as { read: Array<{ frontmatter: { title?: string }; body: string }>; truncated: boolean; returned_body_chars: number };

      expect(JSON.stringify(out).length).toBeLessThanOrEqual(500);
      expect(out).toHaveProperty("missing");
      expect(out).toHaveProperty("omitted");
      expect(out).toHaveProperty("truncated");
      expect(out).toHaveProperty("continuation");
      expect(out.omitted).toContain("/facts/b.md");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("keeps concurrent tool payloads within one shared budget", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      for (let index = 0; index < 20; index += 1) {
        await kb.writeConcept(
          `/facts/${index}.md`,
          { type: "Fact", title: `A ${index}`, description: "x".repeat(400) },
          "body " + "y".repeat(400),
          "add"
        );
      }
      const state = new AgentRunContext({
        maxSteps: 8,
        maxDocumentChars: 1_000,
        maxToolResultChars: 500,
        maxSystemContextChars: 24_000,
      });
      const tools = buildReadTools(kb, undefined, state);
      const results = await Promise.all([
        tools.read_concept!.execute!({ path: "/facts/0.md" }, toolContext),
        tools.search_knowledge!.execute!({ query: "Fact" }, toolContext),
        tools.list_directory!.execute!({}, toolContext),
        tools.lint_knowledge!.execute!({}, toolContext),
      ]);
      const payloadChars = results.reduce(
        (total, value) => total + (JSON.stringify(value)?.length ?? 0),
        0
      );
      // Exhausted parallel calls each still return the explicit notice; those
      // repeated control messages are the documented small accounting overrun.
      expect(payloadChars).toBeLessThanOrEqual(500 + 3 * EXHAUSTION_SERIALISED_LENGTH);
      expect(JSON.stringify(results[0])?.length ?? 0).toBeLessThanOrEqual(500);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("requires a complete current body for replace_body and rejects overwrite writes", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/a.md", { type: "Fact" }, "abcdefghij", "add");
      const state = new AgentRunContext({
        maxSteps: 8,
        maxDocumentChars: 4,
        maxToolResultChars: 10_000,
        maxSystemContextChars: 24_000,
      });
      const reads = buildReadTools(kb, undefined, state);
      const writes = buildWriteTools(kb, new Set(), undefined, state);
      await expect(
        writes.patch_concept!.execute!({
          path: "/facts/a.md",
          replace_body: "new",
          log_summary: "Updated [a](/facts/a.md).",
        }, toolContext)
      ).rejects.toThrow("complete, unchanged read");
      const first = await reads.read_concept!.execute!({ path: "/facts/a.md" }, toolContext) as ReadPage;
      expect(first.truncated).toBe(true);
      await expect(
        writes.patch_concept!.execute!({
          path: "/facts/a.md",
          replace_body: "new",
          log_summary: "Updated [a](/facts/a.md).",
        }, toolContext)
      ).rejects.toThrow("complete, unchanged read");
      await reads.read_concept!.execute!({ path: "/facts/a.md", offset: first.next_offset ?? 0 }, toolContext);
      await reads.read_concept!.execute!({ path: "/facts/a.md", offset: 8 }, toolContext);
      await writes.patch_concept!.execute!({
        path: "/facts/a.md",
        replace_body: "replaced",
        log_summary: "Updated [a](/facts/a.md).",
      }, toolContext);
      await kb.writeConcept("/facts/a.md", { type: "Fact" }, "external", "external update");
      await expect(
        writes.patch_concept!.execute!({
          path: "/facts/a.md",
          replace_body: "stale",
          log_summary: "Updated [a](/facts/a.md).",
        }, toolContext)
      ).rejects.toThrow("complete, unchanged read");
      await expect(
        writes.write_concept!.execute!({
          path: "/facts/a.md",
          frontmatter: { type: "Fact" },
          body: "overwrite",
          log_summary: "Updated [a](/facts/a.md).",
        }, toolContext)
      ).rejects.toThrow("already exists");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("bounds system context separately and keeps truncation markers", () => {
    const state = new AgentRunContext({
      maxSteps: 8,
      maxDocumentChars: 12_000,
      maxToolResultChars: 100_000,
      maxSystemContextChars: 240,
    });
    const types = state.systemTypes(["type-" + "x".repeat(500)]);
    const tree = state.systemTree("t".repeat(500));

    expect(tree).toContain("system tree truncated");
    expect(types.join(", ")).toContain("system types truncated");
    expect(tree.length + types.join(", ").length).toBeLessThanOrEqual(240);
    expect(state.remaining).toBe(100_000);
  });

  it("allows a default first page after system context", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/large.md", { type: "Fact" }, "x".repeat(12_000), "add");
      const state = new AgentRunContext({
        maxSteps: DEFAULT_AGENT_MAX_STEPS,
        maxDocumentChars: DEFAULT_AGENT_MAX_DOCUMENT_CHARS,
        maxToolResultChars: DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
        maxSystemContextChars: DEFAULT_AGENT_MAX_SYSTEM_CONTEXT_CHARS,
      });
      state.systemTree("tree ".repeat(3_000));
      state.systemTypes(["Fact", "Decision"]);
      const tools = buildReadTools(kb, undefined, state);
      const page = await tools.read_concept!.execute!({ path: "/facts/large.md" }, toolContext);

      expect(page).toMatchObject({ path: "/facts/large.md", offset: 0, total_chars: 12_001 });
      expect((page as { body: string }).body).toHaveLength(12_000);
      expect(JSON.stringify(page).length).toBeLessThanOrEqual(DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("retains page metadata when frontmatter is too large", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept(
        "/facts/large-frontmatter.md",
        { type: "Fact", title: "x".repeat(5_000) },
        "body content",
        "add"
      );
      const state = new AgentRunContext({
        maxSteps: 8,
        maxDocumentChars: 1_000,
        maxToolResultChars: 1_000,
        maxSystemContextChars: 24_000,
      });
      const tools = buildReadTools(kb, undefined, state);
      const page = await tools.read_concept!.execute!(
        { path: "/facts/large-frontmatter.md" },
        toolContext
      ) as {
        frontmatter_truncated: boolean;
        body: string;
        offset: number;
        total_chars: number;
        truncated: boolean;
        next_offset: number | null;
      };

      expect(page.frontmatter_truncated).toBe(true);
      expect(page.body.length).toBeGreaterThan(3);
      expect(page).toMatchObject({ offset: 0, total_chars: 13, truncated: false, next_offset: null });
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("keeps a useful body page when frontmatter exceeds the result budget", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept(
        "/facts/huge-frontmatter.md",
        { type: "Fact", title: "x".repeat(30_000) },
        "body ".repeat(4_000),
        "add"
      );
      const tools = buildReadTools(kb);
      const page = await tools.read_concept!.execute!(
        { path: "/facts/huge-frontmatter.md" },
        toolContext
      ) as ReadPage;

      expect(page.frontmatter_truncated).toBe(true);
      expect(page.body.length).toBeGreaterThan(4_000);
      expect(page.next_offset).toBe(page.body.length);
      expect(JSON.stringify(page).length).toBeLessThanOrEqual(DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("meters compact missing and invalid-offset read results", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/read.md", { type: "Fact" }, "body", "add");
      const state = new AgentRunContext({
        maxSteps: 8,
        maxDocumentChars: 1_000,
        maxToolResultChars: 500,
        maxSystemContextChars: 24_000,
      });
      const tools = buildReadTools(kb, undefined, state);
      expect(
        tools.read_concept!.inputSchema.safeParse({
          path: "/facts/" + "x".repeat(600) + ".md",
        }).success
      ).toBe(false);
      const before = state.remaining;
      const missing = await tools.read_concept!.execute!({ path: "/facts/missing.md" }, toolContext);
      expect(missing).toMatchObject({ read: false, error: { code: "not_found" } });
      expect(state.remaining).toBeLessThan(before);
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("uses compact trees in prompts and directory reads", async () => {
    const tree: TreeNode = {
      name: "/",
      path: "/",
      kind: "directory",
      children: [
        {
          name: "fact.md",
          path: "/fact.md",
          kind: "concept",
          type: "Fact",
          title: "A title",
          description: "A description that belongs outside the compact tree",
        },
      ],
    };
    expect(formatTree(tree, 0, false)).toBe("/\n  fact.md  [Fact]");
    expect(buildSystemPrompt({ existingTypes: [], treeSummary: formatTree(tree, 0, false), mode: "query" })).not.toContain(
      "description that belongs outside"
    );

    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept(
        "/fact.md",
        { type: "Fact", title: "A title", description: "A description" },
        "body",
        "add"
      );
      const tools = buildReadTools(kb);
      const listing = (await tools.list_directory!.execute!({}, toolContext)) as string;
      expect(listing).toContain("fact.md  [Fact]");
      expect(listing).not.toContain("A description");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("rejects giant write fields and meters cumulative writes without blocking normal writes", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      const state = new AgentRunContext({
        maxSteps: 8,
        maxDocumentChars: 1_000,
        maxToolResultChars: 10_000,
        maxSystemContextChars: 24_000,
        maxInputChars: 180,
      });
      const writes = buildWriteTools(kb, new Set(), undefined, state);
      const giant = "x".repeat(1_000_000);
      expect(
        writes.write_concept!.inputSchema.safeParse({
          path: "/facts/giant.md",
          frontmatter: { type: "Fact", nested: { payload: giant } },
          body: "small",
          log_summary: "Added giant.",
        }).success
      ).toBe(false);
      await expect(
        writes.write_concept!.execute!({
          path: "/facts/giant.md",
          frontmatter: { type: "Fact", nested: { payload: giant } },
          body: "small",
          log_summary: "Added giant.",
        }, toolContext)
      ).rejects.toThrow("AGENT_MAX_INPUT_CHARS");

      await writes.write_concept!.execute!({
        path: "/facts/a.md",
        frontmatter: { type: "Fact" },
        body: "a".repeat(20),
        log_summary: "Added a.",
      }, toolContext);
      await expect(
        writes.write_concept!.execute!({
          path: "/facts/b.md",
          frontmatter: { type: "Fact" },
          body: "b".repeat(20),
          log_summary: "Added b.",
        }, toolContext)
      ).rejects.toThrow("AGENT_MAX_INPUT_CHARS");
      await expect(kb.readConcept("/facts/a.md")).resolves.toBeDefined();
      await expect(kb.readConcept("/facts/b.md")).rejects.toThrow();
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });
});
