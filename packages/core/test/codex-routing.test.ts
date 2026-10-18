import { afterEach, describe, expect, it, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

const { startThread, run } = vi.hoisted(() => ({ startThread: vi.fn(), run: vi.fn() }));
vi.mock("@openai/codex-sdk", () => ({ Codex: class { startThread = startThread; } }));
import { runMutation, runQuery, streamChat } from "../src/agent/agent.js";
import { KnowledgeBase } from "../src/okf/index.js";

let root: string;
afterEach(async () => {
  delete process.env.LLM_BACKEND;
  delete process.env.AGENT_MAX_STEPS;
  if (root) await fs.rm(root, { recursive: true, force: true });
  startThread.mockReset();
  run.mockReset();
});

describe("Codex backend routing", () => {
  it("routes an explicit codex query through the SDK action loop", async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "codex-routing-"));
    process.env.LLM_BACKEND = "codex";
    run.mockResolvedValue({ finalResponse: '{"kind":"final","text":"answer"}' });
    startThread.mockReturnValue({ run });
    const result = await runQuery(new KnowledgeBase(root), "question");
    expect(result.answer).toBe("answer");
    expect(startThread).toHaveBeenCalledWith(expect.objectContaining({ sandboxMode: "read-only", approvalPolicy: "never" }));
    expect(run).toHaveBeenCalledTimes(1);
  });

  it("reports codex mutations as partial when a later model step fails", async () => {
    root = await fs.mkdtemp(path.join(os.tmpdir(), "codex-partial-"));
    process.env.LLM_BACKEND = "codex";
    run.mockResolvedValueOnce({ finalResponse: JSON.stringify({ kind: "action", tool: "write_concept", arguments: { path: "/facts/a.md", frontmatter: { type: "Fact", title: "A" }, body: "alpha", log_summary: "Added A." } }) }).mockRejectedValueOnce(new Error("model step failed"));
    startThread.mockReturnValue({ run });
    const result = await runMutation(new KnowledgeBase(root), "Record A");
    expect(result).toMatchObject({ ok: false, status: "partial", filesChanged: ["/facts/a.md"], error: "model step failed" });
  });

  it("rejects unsupported chat instead of falling back to an API", async () => {
    process.env.LLM_BACKEND = "codex";
    await expect(streamChat(new KnowledgeBase("/unused"), [{ role: "user", content: "hi" }])).rejects.toThrow("does not support web chat");
    expect(startThread).not.toHaveBeenCalled();
  });
});
