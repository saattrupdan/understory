import http from "node:http";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import express from "express";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { afterEach, describe, expect, it, vi } from "vitest";

const runMutationMock = vi.hoisted(() => vi.fn());
const runQueryCachedMock = vi.hoisted(() => vi.fn());
vi.mock("@understory/core", async () => {
  const actual = await vi.importActual<typeof import("@understory/core")>("@understory/core");
  return { ...actual, runMutation: runMutationMock, runQueryCached: runQueryCachedMock };
});

import { KnowledgeBase } from "@understory/core";
import { buildMcpServer } from "../src/mcp/server.js";
import { mcpRouter } from "../src/mcp/http.js";

let root: string | undefined;
let httpServer: http.Server | undefined;
let client: Client | undefined;

afterEach(async () => {
  runMutationMock.mockReset();
  runQueryCachedMock.mockReset();
  await client?.close();
  client = undefined;
  if (httpServer) {
    await new Promise<void>((resolve, reject) =>
      httpServer!.close((error) => (error ? reject(error) : resolve()))
    );
    httpServer = undefined;
  }
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

  it("routes an SDK cancellation notification to the original HTTP handler", async () => {
    let started!: () => void;
    let aborted!: () => void;
    const handlerStarted = new Promise<void>((resolve) => {
      started = resolve;
    });
    const handlerAborted = new Promise<void>((resolve) => {
      aborted = resolve;
    });
    runQueryCachedMock.mockImplementationOnce(
      (_kb: unknown, _question: string, options: { signal: AbortSignal }) => {
        started();
        return new Promise((_resolve, reject) => {
          options.signal.addEventListener(
            "abort",
            () => {
              aborted();
              reject(options.signal.reason);
            },
            { once: true }
          );
        });
      }
    );

    root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-mcp-http-cancellation-"));
    const app = express();
    app.use(express.json());
    app.use(mcpRouter(new KnowledgeBase(root)));
    httpServer = http.createServer(app);
    await new Promise<void>((resolve) => httpServer!.listen(0, "127.0.0.1", resolve));
    const address = httpServer.address();
    if (!address || typeof address === "string") throw new Error("test server did not start");

    const transport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${address.port}`));
    client = new Client({ name: "cancellation-test", version: "1.0.0" }, { capabilities: {} });
    await client.connect(transport);

    const controller = new AbortController();
    const call = client.callTool(
      { name: "memory_query", arguments: { question: "wait" } },
      undefined,
      { signal: controller.signal }
    );
    const callRejected = expect(call).rejects.toThrow();
    await handlerStarted;
    controller.abort(new DOMException("client cancelled", "AbortError"));
    await handlerAborted;
    await callRejected;
  });
});
