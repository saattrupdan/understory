import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { KnowledgeBase } from "@understory/core";
import { buildMcpServer } from "../src/mcp/server.js";

let root: string | undefined;

afterEach(async () => {
  vi.unstubAllEnvs();
  if (root) await fs.rm(root, { recursive: true, force: true });
  root = undefined;
});

type RegisteredTool = {
  inputSchema: { safeParse(value: unknown): { success: boolean } };
  handler: (value: unknown) => Promise<unknown>;
};

async function tools(): Promise<Record<string, RegisteredTool>> {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-mcp-bounds-"));
  vi.stubEnv("AGENT_MAX_INPUT_CHARS", "32");
  const server = await buildMcpServer(new KnowledgeBase(root));
  return (server as unknown as { _registeredTools: Record<string, RegisteredTool> })._registeredTools;
}

describe("MCP input bounds", () => {
  it("rejects million-character caller arguments in schemas before handlers", async () => {
    const registered = await tools();
    const giant = "x".repeat(1_000_000);
    expect(registered.memory_query.inputSchema.safeParse({ question: giant }).success).toBe(false);
    expect(registered.memory_add.inputSchema.safeParse({ content: giant }).success).toBe(false);
    expect(registered.memory_update.inputSchema.safeParse({ instruction: giant }).success).toBe(false);
  });
});
