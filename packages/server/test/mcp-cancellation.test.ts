import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const runMutationMock = vi.hoisted(() => vi.fn());
const runQueryCachedMock = vi.hoisted(() => vi.fn());
vi.mock("@understory/core", async () => {
  const actual = await vi.importActual<typeof import("@understory/core")>("@understory/core");
  return { ...actual, runMutation: runMutationMock, runQueryCached: runQueryCachedMock };
});

import { KnowledgeBase } from "@understory/core";
import { buildMcpServer } from "../src/mcp/server.js";

let root: string | undefined;

afterEach(async () => {
  runMutationMock.mockReset();
  runQueryCachedMock.mockReset();
  if (root) await fs.rm(root, { recursive: true, force: true });
  root = undefined;
});

type RegisteredTool = {
  handler: (value: unknown, extra: { signal: AbortSignal }) => Promise<unknown>;
};

async function registeredTools(): Promise<Record<string, RegisteredTool>> {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-mcp-cancellation-"));
  const server = await buildMcpServer(new KnowledgeBase(root));
  return (server as unknown as { _registeredTools: Record<string, RegisteredTool> })
    ._registeredTools;
}

describe("MCP cancellation", () => {
  it("forwards the request signal to memory_query", async () => {
    runQueryCachedMock.mockResolvedValue({ answer: "answer", source: "deep" });
    const registered = await registeredTools();
    const controller = new AbortController();

    await registered.memory_query.handler(
      { question: "what is known?" },
      { signal: controller.signal }
    );

    expect(runQueryCachedMock).toHaveBeenCalledWith(
      expect.anything(),
      "what is known?",
      { signal: controller.signal }
    );
  });

  it("forwards the request signal to mutations", async () => {
    runMutationMock.mockResolvedValue({
      ok: true,
      result: { summary: "done", filesChanged: [], steps: 1, traceId: "trace" },
    });
    const registered = await registeredTools();
    const controller = new AbortController();

    await registered.memory_update.handler(
      { instruction: "change it" },
      { signal: controller.signal }
    );

    expect(runMutationMock).toHaveBeenCalledWith(
      expect.anything(),
      "change it",
      { signal: controller.signal }
    );
  });
});
