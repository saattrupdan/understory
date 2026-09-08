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
  DEFAULT_AGENT_MAX_STEPS,
  DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
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
    });
    expect(resolveAgentLimits({ AGENT_MAX_STEPS: "1" }).maxSteps).toBe(MIN_AGENT_MAX_STEPS);
    expect(
      resolveAgentLimits({
        AGENT_MAX_STEPS: "3",
        AGENT_MAX_DOCUMENT_CHARS: "400",
        AGENT_MAX_TOOL_RESULT_CHARS: "900",
      })
    ).toEqual({ maxSteps: 3, maxDocumentChars: 400, maxToolResultChars: 900 });
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
      ).rejects.toThrow("Invalid offset");
    } finally {
      await fs.rm(root, { recursive: true, force: true });
    }
  });

  it("caps aggregate multi-read output as a serialised payload", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-bounds-"));
    try {
      const kb = new KnowledgeBase(root);
      await kb.writeConcept("/facts/a.md", { type: "Fact", title: "A" }, "abcdefghij", "add");
      await kb.writeConcept("/facts/b.md", { type: "Fact", title: "B" }, "klmnopqrst", "add");
      vi.stubEnv("AGENT_MAX_DOCUMENT_CHARS", "8");
      vi.stubEnv("AGENT_MAX_TOOL_RESULT_CHARS", "12");
      const tools = buildReadTools(kb);
      const out = (await tools.read_concepts!.execute!(
        { paths: ["/facts/a.md", "/facts/b.md"] },
        toolContext
      )) as { read: Array<{ frontmatter: { title?: string }; body: string }>; truncated: boolean; returned_body_chars: number };

      expect(JSON.stringify(out).length).toBeLessThanOrEqual(12);
      expect(out.read).toEqual([]);
      expect(out.truncated).toBeUndefined();
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
      expect(payloadChars).toBeLessThanOrEqual(500);
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
});
