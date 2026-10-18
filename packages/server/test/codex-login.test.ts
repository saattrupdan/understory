import { afterEach, describe, expect, it } from "vitest";
import express from "express";
import type { Server } from "node:http";
import { CodexBackend } from "../src/codex-login.js";

let server: Server | undefined;
afterEach(async () => { if (server) await new Promise<void>(resolve => server!.close(() => resolve())); server = undefined; });
async function serve(backend: CodexBackend, allowed: boolean) {
  const app = express(); app.use(express.json()); app.use("/api", backend.router(allowed));
  server = app.listen(0);
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("server did not bind");
  return `http://127.0.0.1:${address.port}`;
}

describe("Codex login routes", () => {
  it("denies profile/open deployments before starting a child process", async () => {
    const backend = new CodexBackend(() => { throw new Error("must not spawn"); });
    const base = await serve(backend, false);
    const response = await fetch(`${base}/api/codex/login`, { method: "POST" });
    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({ error: "Codex login requires authenticated single-user mode." });
  });

  it("rejects credentials and bodies rather than accepting secrets", async () => {
    const backend = new CodexBackend(() => { throw new Error("must not spawn"); });
    const base = await serve(backend, true);
    const response = await fetch(`${base}/api/codex/login`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ token: "secret" }) });
    expect(response.status).toBe(400);
    expect(await response.text()).not.toContain("secret");
  });

  it("starts with API selected", () => expect(new CodexBackend().backend()).toBe("api"));
});
