import { describe, expect, it, vi } from "vitest";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { AgentRunContext } from "../src/agent/run-context.js";
import { buildWriteTools } from "../src/agent/tools.js";
import { KnowledgeBase } from "../src/okf/index.js";

const toolContext = { toolCallId: "cancellation-test", messages: [] };

describe("mutation cancellation", () => {
  it("does not start a cancelled write waiting behind another mutation", async () => {
    const root = await fs.mkdtemp(path.join(os.tmpdir(), "okf-cancellation-test-"));
    let first: Promise<unknown> | undefined;
    let releaseFirst!: () => void;
    try {
      const kb = new KnowledgeBase(root);
      const firstStarted = new Promise<void>((resolve) => {
        const originalWrite = kb.bundle.writeConcept.bind(kb.bundle);
        let calls = 0;
        vi.spyOn(kb.bundle, "writeConcept").mockImplementation(async (...args) => {
          if (calls++ === 0) {
            resolve();
            await new Promise<void>((release) => {
              releaseFirst = release;
            });
          }
          return originalWrite(...args);
        });
      });

      first = kb.writeConcept("/first.md", { type: "Fact" }, "first", "first");
      await firstStarted;

      const controller = new AbortController();
      const writes = buildWriteTools(kb, new Set(), undefined, AgentRunContext.unbounded(controller.signal));
      const cancellation = new DOMException("cancelled", "AbortError");
      const second = writes.write_concept.execute!({
        path: "/second.md",
        frontmatter: { type: "Fact" },
        body: "second",
        log_summary: "second",
      }, toolContext);
      const third = kb.writeConcept("/third.md", { type: "Fact" }, "third", "third");

      controller.abort(cancellation);
      releaseFirst();

      await expect(first).resolves.toBeDefined();
      await expect(second).rejects.toBe(cancellation);
      await expect(third).resolves.toBeDefined();
      await expect(kb.readConcept("/second.md")).rejects.toThrow("not found");
      await expect(kb.readConcept("/third.md")).resolves.toBeDefined();
    } finally {
      releaseFirst?.();
      await first?.catch(() => undefined);
      await fs.rm(root, { recursive: true, force: true });
      vi.restoreAllMocks();
    }
  });
});
