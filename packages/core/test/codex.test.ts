import { describe, expect, it, vi } from "vitest";
import { runCodexActions } from "../src/agent/codex.js";

const schema = { safeParse: (value: unknown) => typeof value === "object" && value !== null && "value" in value
  ? { success: true, data: value } : { success: false, error: { message: "value required" } } };
function setup(responses: string[]) {
  const run = vi.fn(async () => ({ finalResponse: responses.shift()! }));
  const startThread = vi.fn((options: object) => ({ run }));
  const codexFactory = vi.fn(() => ({ startThread }));
  return { run, startThread, codexFactory };
}
const tool = { description: "read", inputSchema: schema, execute: vi.fn(async ({ value }: { value: string }) => ({ value })) };

describe("Codex action loop", () => {
  it("uses only the supplied host tools and starts in an isolated read-only temp directory", async () => {
    const mock = setup(['{"kind":"action","tool":"read","arguments":{"value":"x"}}', '{"kind":"final","text":"done"}']);
    const result = await runCodexActions("question", { read: tool }, 3, undefined, { codexFactory: mock.codexFactory as never, env: { HOME: "/home/test", PATH: "/bin", SECRET: "hidden" } });
    expect(result).toEqual({ kind: "final", text: "done", steps: 2 });
    expect(tool.execute).toHaveBeenCalledWith({ value: "x" });
    expect(mock.startThread).toHaveBeenCalledWith(expect.objectContaining({ skipGitRepoCheck: true, sandboxMode: "read-only", approvalPolicy: "never", networkAccessEnabled: false, webSearchMode: "disabled" }));
    const cwd = mock.startThread.mock.calls[0]![0].workingDirectory;
    expect(cwd).not.toContain("/home/test");
    expect(mock.run.mock.calls[0]![0]).not.toContain("/home/test");
    expect(mock.codexFactory.mock.calls[0]![0].env).toEqual({ HOME: "/home/test", PATH: "/bin" });
  });

  it("rejects unsupported tool names and invalid arguments without execution", async () => {
    const badTool = setup(['{"kind":"action","tool":"delete_everything","arguments":{}}']);
    await expect(runCodexActions("q", { read: tool }, 2, undefined, { codexFactory: badTool.codexFactory as never })).rejects.toThrow("unsupported action");
    tool.execute.mockClear();
    const badArgs = setup(['{"kind":"action","tool":"read","arguments":{}}']);
    await expect(runCodexActions("q", { read: tool }, 2, undefined, { codexFactory: badArgs.codexFactory as never })).rejects.toThrow("Invalid arguments");
    expect(tool.execute).not.toHaveBeenCalled();
  });

  it("caps action output and stops at the configured step count", async () => {
    const capTool = { ...tool, execute: vi.fn(async () => "x".repeat(30_000)) };
    const capped = setup(['{"kind":"action","tool":"read","arguments":{"value":"x"}}', '{"kind":"final","text":"ok"}']);
    await runCodexActions("q", { read: capTool }, 2, undefined, { codexFactory: capped.codexFactory as never });
    expect(capped.run.mock.calls[1]![0]).toContain("[truncated]");
    const limited = setup(['{"kind":"action","tool":"read","arguments":{"value":"x"}}']);
    expect(await runCodexActions("q", { read: tool }, 1, undefined, { codexFactory: limited.codexFactory as never })).toEqual({ kind: "step-limit", steps: 1 });
    expect(limited.run).toHaveBeenCalledTimes(1);
  });

  it("passes a deadline signal through to the SDK", async () => {
    const timeout = AbortSignal.timeout(1_000);
    const mock = setup(['{"kind":"final","text":"done"}']);
    await runCodexActions("q", { read: tool }, 2, timeout, { codexFactory: mock.codexFactory as never });
    expect(mock.run.mock.calls[0]![1].signal).toBe(timeout);
  });

  it("passes cancellation through to the SDK and preserves already-run host mutations", async () => {
    const controller = new AbortController();
    const partialFiles = new Set<string>();
    const mutatingTool = { ...tool, execute: vi.fn(async () => { partialFiles.add("/created.md"); return { written: true }; }) };
    const mock = setup(['{"kind":"action","tool":"write","arguments":{"value":"x"}}']);
    await expect(runCodexActions("q", { write: mutatingTool }, 2, controller.signal, { codexFactory: mock.codexFactory as never })).rejects.toThrow();
    expect(partialFiles.has("/created.md")).toBe(true);
    expect(mock.run.mock.calls[0]![1].signal).toBe(controller.signal);
    controller.abort(new Error("cancelled"));
    const aborted = setup([]);
    await expect(runCodexActions("q", { read: tool }, 2, controller.signal, { codexFactory: aborted.codexFactory as never })).rejects.toThrow("cancelled");
    expect(aborted.run).not.toHaveBeenCalled();
  });
});
