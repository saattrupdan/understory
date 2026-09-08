import { describe, expect, it, afterEach, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { KnowledgeBase } from "../src/okf/index.js";
import { buildSystemPrompt } from "../src/agent/system-prompt.js";
import { buildReadTools, formatTree } from "../src/agent/tools.js";
import {
  DEFAULT_AGENT_MAX_DOCUMENT_CHARS,
  DEFAULT_AGENT_MAX_STEPS,
  DEFAULT_AGENT_MAX_TOOL_RESULT_CHARS,
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
    expect(
      resolveAgentLimits({
        AGENT_MAX_STEPS: "3",
        AGENT_MAX_DOCUMENT_CHARS: "400",
        AGENT_MAX_TOOL_RESULT_CHARS: "900",
      })
    ).toEqual({ maxSteps: 3, maxDocumentChars: 400, maxToolResultChars: 900 });
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

  it("caps aggregate multi-read output while retaining frontmatter", async () => {
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

      expect(out.read).toHaveLength(2);
      expect(out.read.map((c) => c.frontmatter.title)).toEqual(["A", "B"]);
      expect(out.read.map((c) => c.body)).toEqual(["abcdefgh", "klmn"]);
      expect(out).toMatchObject({ truncated: true, returned_body_chars: 12 });
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
