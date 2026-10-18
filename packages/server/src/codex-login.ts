import { spawn, type ChildProcess } from "node:child_process";
import { createRequire } from "node:module";
import { mkdir, chmod } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import type { Router } from "express";
import express from "express";

const MAX_LINE = 64 * 1024;
const LOGIN_TIMEOUT_MS = 15 * 60_000;
type Json = Record<string, unknown>;
type Spawn = typeof spawn;
const require = createRequire(import.meta.url);

/** Process-scoped backend selector. */
export class CodexBackend {
  private selected: "api" | "codex" = "api";
  private pending?: { controller: AbortController };
  constructor(private readonly spawnProcess: Spawn = spawn) {}
  backend(): "api" | "codex" { return this.selected; }
  async restore(): Promise<void> {
    try {
      const result = await this.accountRead();
      if (result?.account && isJson(result.account) && result.account.type === "chatgpt") this.selected = "codex";
    } catch { /* app-server may be unavailable; API remains selected */ }
  }

  async loginStatus(): Promise<{ backend: "api" | "codex"; loggedIn: boolean; pending: boolean }> {
    const account = await this.accountRead();
    const loggedIn = Boolean(account?.account && isJson(account.account) && account.account.type === "chatgpt");
    if (loggedIn) this.selected = "codex";
    return { backend: this.selected, loggedIn, pending: Boolean(this.pending) };
  }

  async loginStart(): Promise<{ verificationUrl: string; userCode: string; expiresAt?: string }> {
    if (this.pending) throw new Error("A Codex login is already in progress.");
    const pending = { controller: new AbortController() };
    this.pending = pending;
    try {
      const session = await this.deviceLogin(pending.controller.signal);
      void session.completed.then((account) => {
        if (!account?.account || !isJson(account.account) || account.account.type !== "chatgpt") throw new Error("Login did not return a ChatGPT account.");
        this.selected = "codex";
      }).catch(() => {}).finally(() => { if (this.pending === pending) this.pending = undefined; });
      return { verificationUrl: session.verificationUrl, userCode: session.userCode, ...(session.expiresAt ? { expiresAt: session.expiresAt } : {}) };
    } catch (error) {
      if (this.pending === pending) this.pending = undefined;
      throw error;
    }
  }

  loginCancel(): boolean {
    const pending = this.pending;
    if (!pending) return false;
    pending.controller.abort();
    this.pending = undefined;
    return true;
  }

  router(allowed: boolean): Router {
    const router = express.Router();
    const denied = (_req: express.Request, res: express.Response) => res.status(403).json({ error: "Codex login requires authenticated single-user mode." });
    router.get("/codex/login", async (req, res) => {
      if (!allowed) return denied(req, res);
      try { res.json(await this.loginStatus()); }
      catch { res.status(503).json({ error: "Codex account status unavailable." }); }
    });
    router.post("/codex/login", async (req, res) => {
      if (!allowed) return denied(req, res);
      if (req.body && Object.keys(req.body).length) return res.status(400).json({ error: "Login takes no credentials or request body." });
      if (this.pending) return res.status(409).json({ error: "A Codex login is already in progress." });
      const pending = { controller: new AbortController() };
      this.pending = pending;
      const onClose = () => { if (!res.writableEnded) pending.controller.abort(); };
      res.once("close", onClose);
      try {
        const session = await this.deviceLogin(pending.controller.signal);
        if (pending.controller.signal.aborted) throw new Error("Login request disconnected.");
        res.json({ verificationUrl: session.verificationUrl, userCode: session.userCode, expiresAt: session.expiresAt });
        res.off("close", onClose);
        void session.completed.then((account) => {
          if (!account?.account || !isJson(account.account) || account.account.type !== "chatgpt") throw new Error("Login did not return a ChatGPT account.");
          this.selected = "codex";
        }).catch(() => {}).finally(() => { if (this.pending === pending) this.pending = undefined; });
      } catch {
        res.off("close", onClose);
        if (this.pending === pending) this.pending = undefined;
        if (!res.headersSent && !res.destroyed) res.status(502).json({ error: "Could not start Codex device login." });
      }
    });
    router.delete("/codex/login", async (req, res) => {
      if (!allowed) return denied(req, res);
      if (!this.loginCancel()) return res.status(409).json({ error: "No Codex login is in progress." });
      res.json({ cancelled: true });
    });
    return router;
  }

  private async accountRead(): Promise<Json | null> {
    const rpc = await this.start();
    try { const result = await rpc.call("account/read", { refreshToken: false }); return isJson(result) ? result : null; }
    finally { rpc.close(); }
  }
  private async deviceLogin(signal: AbortSignal): Promise<{ verificationUrl: string; userCode: string; expiresAt?: string; loginId: string; completed: Promise<Json | null> }> {
    const rpc = await this.start(signal);
    const completed = rpc.notification("account/login/completed");
    // A failed start can reject the waiter before the completion handler exists.
    void completed.catch(() => {});
    try {
      const result = await rpc.call("account/login/start", { type: "chatgptDeviceCode" });
      if (!isJson(result)) throw new Error("bad login response");
      const login = isJson(result.login) ? result.login : result;
      const url = typeof login.verificationUrl === "string" ? login.verificationUrl : "";
      const code = typeof login.userCode === "string" ? login.userCode : "";
      const loginId = typeof login.loginId === "string" ? login.loginId : "";
      if (!url || !code || !loginId) throw new Error("incomplete login response");
      const verification = new URL(url);
      if (verification.protocol !== "https:" || verification.hostname !== "auth.openai.com" ||
          verification.pathname !== "/codex/device" || verification.username || verification.password ||
          !/^[A-Z0-9-]{4,64}$/.test(code) || loginId.length > 256) {
        throw new Error("unexpected device authorization instructions");
      }
      const finished = completed.then(async event => {
        if (event.loginId !== loginId || event.success !== true) throw new Error("device login failed");
        const account = await rpc.call("account/read", { refreshToken: false });
        return isJson(account) ? account : null;
      }).finally(() => rpc.close());
      return { verificationUrl: verification.toString(), userCode: code, loginId, expiresAt: typeof login.expiresAt === "string" ? login.expiresAt : undefined, completed: finished };
    } catch (error) { rpc.close(); throw error; }
  }
  private async start(signal?: AbortSignal): Promise<Rpc> {
    const codexBin = require.resolve("@openai/codex/bin/codex.js");
    const env: NodeJS.ProcessEnv = {};
    for (const key of ["PATH", "HOME", "TMPDIR", "TMP", "TEMP", "SystemRoot"]) {
      if (process.env[key]) env[key] = process.env[key];
    }
    env.CODEX_HOME = process.env.CODEX_HOME || path.join(os.homedir(), ".codex");
    if (process.env.CODEX_HOME) {
      await mkdir(env.CODEX_HOME, { recursive: true, mode: 0o700 });
      await chmod(env.CODEX_HOME, 0o700);
    }
    const child = this.spawnProcess(process.execPath, [codexBin, "app-server"], { stdio: ["pipe", "pipe", "ignore"], env });
    const rpc = new Rpc(child, signal);
    try {
      await rpc.call("initialize", { clientInfo: { name: "understory", title: "Understory", version: "0.1.0" }, capabilities: {} });
      rpc.notify("initialized", {});
      return rpc;
    } catch (error) { rpc.close(); throw error; }
  }
}
function isJson(value: unknown): value is Json { return value !== null && typeof value === "object" && !Array.isArray(value); }

class Rpc {
  private id = 0; private buffer = ""; private output = 0; private closed = false;
  private readonly replies = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>();
  private readonly notifications = new Map<string, { resolve(value: Json): void; reject(error: Error): void }>();
  private readonly timer: NodeJS.Timeout;
  constructor(private readonly child: ChildProcess, signal?: AbortSignal) {
    this.timer = setTimeout(() => this.fail(new Error("timeout")), LOGIN_TIMEOUT_MS);
    child.stdout?.on("data", (chunk: Buffer) => {
      this.output += chunk.length;
      if (this.output > 1024 * 1024) return this.fail(new Error("output limit"));
      this.buffer += chunk.toString("utf8");
      if (this.buffer.length > MAX_LINE && !this.buffer.includes("\n")) return this.fail(new Error("line limit"));
      let idx: number;
      while ((idx = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, idx); this.buffer = this.buffer.slice(idx + 1);
        try { this.message(JSON.parse(line) as Json); } catch { this.fail(new Error("invalid response")); }
      }
    });
    child.once("error", () => this.fail(new Error("spawn failure")));
    child.once("exit", () => this.fail(new Error("process exited")));
    if (signal?.aborted) this.fail(new Error("cancelled"));
    else signal?.addEventListener("abort", () => this.fail(new Error("cancelled")), { once: true });
  }
  call(method: string, params: Json): Promise<unknown> {
    const id = ++this.id;
    return new Promise((resolve, reject) => { this.replies.set(id, { resolve, reject }); this.write({ jsonrpc: "2.0", id, method, params }); });
  }
  notify(method: string, params: Json): void { this.write({ jsonrpc: "2.0", method, params }); }
  notification(method: string): Promise<Json> { return new Promise((resolve, reject) => this.notifications.set(method, { resolve, reject })); }
  close(reason: Error = new Error("Codex app-server closed.")): void {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.timer);
    for (const item of this.replies.values()) item.reject(reason);
    this.replies.clear();
    for (const item of this.notifications.values()) item.reject(reason);
    this.notifications.clear();
    if (this.child.exitCode === null && !this.child.killed) this.child.kill("SIGTERM");
  }
  private write(value: Json): void { if (this.closed || !this.child.stdin?.writable) throw new Error("closed"); this.child.stdin.write(`${JSON.stringify(value)}\n`, error => { if (error) this.fail(new Error("write failed")); }); }
  private message(message: Json): void {
    if (typeof message.id === "number") {
      const pending = this.replies.get(message.id); if (!pending) return;
      this.replies.delete(message.id);
      if (message.error) pending.reject(new Error("Codex app-server request failed.")); else pending.resolve(message.result);
    } else if (typeof message.method === "string") {
      const pending = this.notifications.get(message.method); if (!pending) return;
      this.notifications.delete(message.method); pending.resolve(isJson(message.params) ? message.params : {});
    }
  }
  private fail(error: Error): void { this.close(error); }
}
