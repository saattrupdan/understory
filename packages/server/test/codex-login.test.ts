import { afterEach, describe, expect, it, vi } from "vitest";
import { EventEmitter } from "node:events";
import { promises as fs } from "node:fs";
import type { Server } from "node:http";
import os from "node:os";
import path from "node:path";
import { PassThrough } from "node:stream";
import { KnowledgeBase } from "@understory/core";
import { buildApp } from "../src/app.js";
import { CodexBackend } from "../src/codex-login.js";
import { singleMemoryHost } from "../src/host.js";

let server: Server | undefined;
let root: string | undefined;
afterEach(async () => {
  if (server) await new Promise<void>(resolve => server!.close(() => resolve()));
  if (root) await fs.rm(root, { recursive: true, force: true });
  server = undefined;
  root = undefined;
});

async function serve(backend: CodexBackend, allowed = true, authToken: string | null = "private-token") {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "codex-login-test-"));
  const app = buildApp(singleMemoryHost(new KnowledgeBase(root), authToken ?? undefined), {
    codexBackend: backend,
    codexLoginAllowed: allowed,
  });
  server = app.listen(0);
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("server did not bind");
  return `http://127.0.0.1:${address.port}/api/codex/login`;
}

const authorized = { authorization: "Bearer private-token" };

type FakeChild = EventEmitter & {
  stdin: PassThrough;
  stdout: PassThrough;
  killed: boolean;
  exitCode: null;
  kill(): boolean;
};

function appServer(state: { loggedIn: boolean; startError?: boolean; delayStart?: boolean }) {
  const children: FakeChild[] = [];
  const spawnProcess = vi.fn((_executable: string, _args: string[], _options: { env?: NodeJS.ProcessEnv }) => {
    const child = Object.assign(new EventEmitter(), {
      stdin: new PassThrough(), stdout: new PassThrough(),
      killed: false, exitCode: null as null,
      kill() { child.killed = true; return true; },
    }) as FakeChild;
    children.push(child);
    let input = "";
    child.stdin.on("data", (chunk: Buffer) => {
      input += chunk.toString("utf8");
      let newline: number;
      while ((newline = input.indexOf("\n")) !== -1) {
        const line = input.slice(0, newline);
        input = input.slice(newline + 1);
        const request = JSON.parse(line) as { id?: number; method: string };
        if (request.id === undefined) continue;
        const send = (data: unknown) => child.stdout.write(`${JSON.stringify(data)}\n`);
        if (request.method === "initialize") send({ id: request.id, result: {} });
        if (request.method === "account/read") send({ id: request.id, result: {
          account: state.loggedIn ? { type: "chatgpt", planType: "plus" } : null,
          requiresOpenaiAuth: true,
        } });
        if (request.method === "account/login/start") {
          if (state.delayStart) continue;
          if (state.startError) send({ id: request.id, error: { message: "private failure details" } });
          else send({ id: request.id, result: {
            type: "chatgptDeviceCode", loginId: "attempt-1",
            verificationUrl: "https://auth.openai.com/codex/device", userCode: "ABCD-1234",
          } });
        }
      }
    });
    return child;
  });
  function complete(success: boolean, loginId = "attempt-1") {
    state.loggedIn = success && loginId === "attempt-1";
    children.at(-1)!.stdout.write(`${JSON.stringify({ method: "account/login/completed", params: {
      loginId, success, error: success ? null : "rejected",
    } })}\n`);
  }
  return { children, spawnProcess, complete };
}

describe("Codex login routes", () => {
  it("requires a bearer token and rejects disabled deployments before spawning", async () => {
    const backend = new CodexBackend(() => { throw new Error("must not spawn"); });
    const url = await serve(backend, false);
    expect((await fetch(url, { method: "POST" })).status).toBe(401);
    const denied = await fetch(url, { method: "POST", headers: authorized });
    expect(denied.status).toBe(403);
  });

  it("rejects submitted credentials rather than accepting or echoing secrets", async () => {
    const backend = new CodexBackend(() => { throw new Error("must not spawn"); });
    const url = await serve(backend);
    const response = await fetch(url, { method: "POST", headers: { ...authorized, "content-type": "application/json" }, body: JSON.stringify({ token: "secret" }) });
    expect(response.status).toBe(400);
    expect(await response.text()).not.toContain("secret");
  });

  it("starts a device flow, switches on matching success, and restores a persisted login", async () => {
    const state = { loggedIn: false };
    const fake = appServer(state);
    const backend = new CodexBackend(fake.spawnProcess as never);
    const url = await serve(backend);
    const start = await fetch(url, { method: "POST", headers: authorized });
    expect(start.status).toBe(200);
    expect(await start.json()).toMatchObject({ verificationUrl: "https://auth.openai.com/codex/device", userCode: "ABCD-1234" });
    expect(backend.backend()).toBe("api");
    expect((await fetch(url, { method: "POST", headers: authorized })).status).toBe(409);
    fake.complete(true);
    await vi.waitFor(() => expect(backend.backend()).toBe("codex"));
    const status = await fetch(url, { headers: authorized });
    expect(await status.json()).toMatchObject({ loggedIn: true, backend: "codex", pending: false });
    const restarted = new CodexBackend(fake.spawnProcess as never);
    await restarted.restore();
    expect(restarted.backend()).toBe("codex");
    expect(fake.spawnProcess.mock.calls.every((args) => args[0] === process.execPath)).toBe(true);
    expect(fake.children.every((child) => child.killed)).toBe(true);
  });

  it("does not switch on rejected or mismatched completion", async () => {
    const state = { loggedIn: false };
    const fake = appServer(state);
    const backend = new CodexBackend(fake.spawnProcess as never);
    const url = await serve(backend);
    expect((await fetch(url, { method: "POST", headers: authorized })).status).toBe(200);
    fake.complete(false);
    await vi.waitFor(async () => {
      const status = await fetch(url, { headers: authorized });
      expect(await status.json()).toMatchObject({ backend: "api", pending: false });
    });
    expect((await fetch(url, { method: "POST", headers: authorized })).status).toBe(200);
    fake.complete(true, "different-attempt");
    await vi.waitFor(async () => {
      const status = await fetch(url, { headers: authorized });
      expect(await status.json()).toMatchObject({ backend: "api", pending: false });
    });
  });

  it("cancels the original login process and allows retry", async () => {
    const fake = appServer({ loggedIn: false });
    const backend = new CodexBackend(fake.spawnProcess as never);
    const url = await serve(backend);
    expect((await fetch(url, { method: "POST", headers: authorized })).status).toBe(200);
    const loginChild = fake.children.at(-1)!;
    expect((await fetch(url, { method: "DELETE", headers: authorized })).status).toBe(200);
    expect(loginChild.killed).toBe(true);
    expect(backend.backend()).toBe("api");
    expect((await fetch(url, { method: "POST", headers: authorized })).status).toBe(200);
    expect((await fetch(url, { method: "DELETE", headers: authorized })).status).toBe(200);
  });

  it("cleans up a failed start without leaking its error or blocking retries", async () => {
    const fake = appServer({ loggedIn: false, startError: true });
    const backend = new CodexBackend(fake.spawnProcess as never);
    const url = await serve(backend);
    const result = await fetch(url, { method: "POST", headers: authorized });
    expect(result.status).toBe(502);
    expect(await result.text()).not.toContain("private failure details");
    expect(backend.backend()).toBe("api");
    expect(fake.children.at(-1)!.killed).toBe(true);
    expect((await fetch(url, { method: "POST", headers: authorized })).status).toBe(502);
  });

  it("cancels a disconnected startup but not a login after delivering its code", async () => {
    const state = { loggedIn: false, delayStart: true };
    const fake = appServer(state);
    const backend = new CodexBackend(fake.spawnProcess as never);
    const controller = new AbortController();
    const starting = backend.loginStart(controller.signal);
    await vi.waitFor(() => expect(fake.children.length).toBe(1));
    controller.abort();
    await expect(starting).rejects.toThrow();
    expect(fake.children[0]!.killed).toBe(true);
    state.delayStart = false;
    const afterChallenge = new AbortController();
    expect(await backend.loginStart(afterChallenge.signal)).toMatchObject({ userCode: "ABCD-1234" });
    afterChallenge.abort();
    expect(fake.children.at(-1)!.killed).toBe(false);
    fake.complete(true);
    await vi.waitFor(() => expect(backend.backend()).toBe("codex"));
  });

  it("exposes the MCP login action only on the authenticated single-user server", async () => {
    const backend = new CodexBackend(() => { throw new Error("listing must not spawn"); });
    const url = await serve(backend);
    const mcpUrl = url.replace("/api/codex/login", "/mcp");
    const list = async (headers: Record<string, string>) => fetch(mcpUrl, {
      method: "POST", headers: { ...headers, accept: "application/json, text/event-stream", "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }),
    });
    expect((await list({})).status).toBe(401);
    const response = await list(authorized);
    expect(response.status).toBe(200);
    const body = await response.json() as { result?: { tools?: Array<{ name: string }> } };
    expect(body.result?.tools?.map(tool => tool.name)).toContain("memory_codex_login");
  });

  it("allows the login tool on an explicitly local, token-free single-user server", async () => {
    const backend = new CodexBackend(() => { throw new Error("listing must not spawn"); });
    const url = await serve(backend, true, null);
    const response = await fetch(url.replace("/api/codex/login", "/mcp"), {
      method: "POST", headers: { accept: "application/json, text/event-stream", "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }),
    });
    expect(response.status).toBe(200);
    const body = await response.json() as { result?: { tools?: Array<{ name: string }> } };
    expect(body.result?.tools?.map(tool => tool.name)).toContain("memory_codex_login");
  });

  it("hides the MCP login action when single-user login is disabled", async () => {
    const backend = new CodexBackend(() => { throw new Error("listing must not spawn"); });
    const url = await serve(backend, false);
    const response = await fetch(url.replace("/api/codex/login", "/mcp"), {
      method: "POST", headers: { ...authorized, accept: "application/json, text/event-stream", "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "tools/list", params: {} }),
    });
    expect(response.status).toBe(200);
    const body = await response.json() as { result?: { tools?: Array<{ name: string }> } };
    expect(body.result?.tools?.map(tool => tool.name)).not.toContain("memory_codex_login");
  });

  it("starts login through MCP and switches the shared backend on completion", async () => {
    const fake = appServer({ loggedIn: false });
    const backend = new CodexBackend(fake.spawnProcess as never);
    const url = await serve(backend);
    const response = await fetch(url.replace("/api/codex/login", "/mcp"), {
      method: "POST", headers: { ...authorized, accept: "application/json, text/event-stream", "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/call", params: {
        name: "memory_codex_login", arguments: { action: "start" },
      } }),
    });
    expect(response.status).toBe(200);
    const body = await response.json() as { result?: { content?: Array<{ text: string }> } };
    expect(JSON.parse(body.result?.content?.[0]?.text ?? "null")).toEqual({
      verificationUrl: "https://auth.openai.com/codex/device", userCode: "ABCD-1234",
    });
    expect(backend.backend()).toBe("api");
    fake.complete(true);
    await vi.waitFor(() => expect(backend.backend()).toBe("codex"));
  });
});
