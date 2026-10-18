import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import type { Router } from "express";
import express from "express";

const MAX_LINE = 64 * 1024;
const LOGIN_TIMEOUT_MS = 120_000;

type Json = Record<string, unknown>;
type Spawn = typeof spawn;

/** A process-scoped selector; each request captures its current backend. */
export class CodexBackend {
  private selected: "api" | "codex" = "api";
  private pending = false;
  constructor(private readonly spawnProcess: Spawn = spawn) {}

  backend(): "api" | "codex" { return this.selected; }

  router(allowed: boolean): Router {
    const router = express.Router();
    router.get("/codex/login", async (_req, res) => {
      if (!allowed) return res.status(403).json({ error: "Codex login is available only in authenticated single-user mode." });
      try {
        const account = await this.accountRead();
        res.json({ backend: this.selected, loggedIn: account?.authMode === "chatgpt", pending: this.pending });
      } catch {
        res.status(503).json({ error: "Codex account status unavailable." });
      }
    });
    router.post("/codex/login", async (req, res) => {
      if (!allowed) return res.status(403).json({ error: "Codex login is available only in authenticated single-user mode." });
      if (req.body && Object.keys(req.body).length) return res.status(400).json({ error: "Login takes no credentials or request body." });
      if (this.pending) return res.status(409).json({ error: "A Codex login is already in progress." });
      this.pending = true;
      try {
        const device = await this.deviceLogin();
        res.json({ verificationUrl: device.verificationUrl, userCode: device.userCode, expiresAt: device.expiresAt });
        void device.completed.then((account) => {
          if (account?.authMode !== "chatgpt") throw new Error("Codex login did not authenticate a ChatGPT account.");
          this.selected = "codex";
        }).catch(() => {}).finally(() => { this.pending = false; });
      } catch (err) {
        this.pending = false;
        if (!res.headersSent) res.status(502).json({ error: err instanceof Error ? err.message : "Codex login failed." });
      }
    });
    return router;
  }

  private async accountRead(): Promise<Json | null> {
    const rpc = await this.start();
    try { return await rpc.call("account/read", { refreshToken: false }) as Json; }
    finally { rpc.close(); }
  }

  private async deviceLogin(signal: AbortSignal): Promise<{ verificationUrl: string; userCode: string; expiresAt?: string; completed: Promise<Json | null> }> {
    const rpc = await this.start(signal);
    const completed = rpc.notification("account/login/completed");
    try {
      const result = await rpc.call("account/login/start", { type: "chatgptDeviceCode" }) as Json;
      const login = result.login as Json | undefined;
      const url = String(login?.verificationUrl ?? result.verificationUrl ?? "");
      const code = String(login?.userCode ?? result.userCode ?? "");
      if (!url || !code) throw new Error("Codex did not return device authorization instructions.");
      return { verificationUrl: url, userCode: code, expiresAt: typeof login?.expiresAt === "string" ? login.expiresAt : undefined,
        completed: completed.then(async (event) => {
          if (event && event.loginId && result.loginId && event.loginId !== result.loginId) throw new Error("Codex login completion did not match the active login.");
          const account = await rpc.call("account/read", { refreshToken: false }) as Json;
          rpc.close();
          return account;
        }) };
    } catch (err) { rpc.close(); throw err; }
  }

  private async start(signal?: AbortSignal): Promise<Rpc> {
    const child = this.spawnProcess("codex", ["app-server"], { stdio: ["pipe", "pipe", "ignore"], env: process.env });
    const rpc = new Rpc(child, signal);
    try {
      await rpc.call("initialize", { clientInfo: { name: "understory", title: "Understory", version: "0.1.0" }, capabilities: {} });
      rpc.notify("initialized", {});
      return rpc;
    } catch (err) { rpc.close(); throw err; }
  }
}

class Rpc {
  private id = 0;
  private buffer = "";
  private output = 0;
  private readonly replies = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>();
  private readonly notifications = new Map<string, { resolve(value: Json): void; reject(error: Error): void }>();
  private readonly timer: NodeJS.Timeout;
  constructor(private readonly child: ChildProcessWithoutNullStreams, signal?: AbortSignal) {
    this.timer = setTimeout(() => this.fail(new Error("Codex login timed out.")), LOGIN_TIMEOUT_MS);
    child.stdout.on("data", (chunk: Buffer) => {
      this.output += chunk.length;
      if (this.output > 1024 * 1024) return this.fail(new Error("Codex app-server output limit exceeded."));
      this.buffer += chunk.toString("utf8");
      if (this.buffer.length > MAX_LINE && !this.buffer.includes("\n")) return this.fail(new Error("Codex app-server response too large."));
      let idx: number;
      while ((idx = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, idx); this.buffer = this.buffer.slice(idx + 1);
        try { this.message(JSON.parse(line) as Json); } catch { this.fail(new Error("Invalid Codex app-server response.")); }
      }
    });
    child.once("error", () => this.fail(new Error("Could not start Codex app-server.")));
    child.once("exit", () => this.fail(new Error("Codex app-server exited unexpectedly.")));
    signal?.addEventListener("abort", () => this.fail(new Error("Codex login cancelled.")), { once: true });
  }
  call(method: string, params: Json): Promise<unknown> {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.replies.set(id, { resolve, reject });
      this.write({ jsonrpc: "2.0", id, method, params });
    });
  }
  notify(method: string, params: Json): void { this.write({ jsonrpc: "2.0", method, params }); }
  notification(method: string): Promise<Json> { return new Promise((resolve, reject) => this.notifications.set(method, { resolve, reject })); }
  close(): void { clearTimeout(this.timer); if (!this.child.killed) this.child.kill("SIGTERM"); }
  private write(value: Json): void { this.child.stdin.write(`${JSON.stringify(value)}\n`); }
  private message(message: Json): void {
    if (typeof message.id === "number") {
      const pending = this.replies.get(message.id); if (!pending) return;
      this.replies.delete(message.id);
      if (message.error) pending.reject(new Error("Codex app-server request failed.")); else pending.resolve(message.result);
    } else if (typeof message.method === "string") {
      this.notifications.get(message.method)?.resolve(message.params as Json ?? {});
    }
  }
  private fail(error: Error): void {
    for (const pending of this.replies.values()) pending.reject(error);
    this.replies.clear();
    for (const waiter of this.notifications.values()) waiter.reject(error);
    this.notifications.clear();
    this.close();
  }
}
