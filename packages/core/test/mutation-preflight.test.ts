import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const generateTextMock = vi.hoisted(() => vi.fn());
vi.mock("ai", async () => ({
  ...(await vi.importActual<typeof import("ai")>("ai")),
  generateText: generateTextMock,
}));
vi.mock("../src/providers/index.js", async () => ({
  ...(await vi.importActual<typeof import("../src/providers/index.js")>("../src/providers/index.js")),
  createModel: vi.fn(async () => ({
    specificationVersion: "v1",
    provider: "test-provider",
    modelId: "test-model",
    doGenerate: async () => ({ content: [{ type: "text", text: "done" }], usage: { inputTokens: 1, outputTokens: 1 } }),
  })),
}));

import { runMutation } from "../src/agent/agent.js";
import { KnowledgeBase } from "../src/okf/index.js";

let root: string;
let kb: KnowledgeBase;
const toolContext = { toolCallId: "preflight-test", messages: [] };
const oldBody = "Monthly charges are scheduled; ad-hoc charges come from support tooling.";
const newBody = "Monthly charges are scheduled; support tooling creates ad-hoc charges only after approval.";

type MutationRequest = {
  prompt: string;
  tools: {
    patch_concept: {
      execute: (input: unknown, context: typeof toolContext) => Promise<unknown>;
    };
  };
};

beforeEach(async () => {
  root = await fs.mkdtemp(path.join(os.tmpdir(), "ustory-preflight-"));
  kb = new KnowledgeBase(root);
  vi.stubEnv("LLM_API_BASE_URL", "http://localhost:1/v1");
  vi.stubEnv("LLM_API_FORMAT", "openai");
  vi.stubEnv("LLM_MODEL", "test-model");
  vi.stubEnv("MUTATION_PREFLIGHT", "true");
});

afterEach(async () => {
  await fs.rm(root, { recursive: true, force: true });
  generateTextMock.mockReset();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

async function billingFixture() {
  await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, oldBody, "add");
  await kb.writeConcept("/playbooks/oncall-billing.md", { type: "Playbook", title: "Billing On-call" }, "Check billing API errors when charges fail.", "add");
  await kb.writeConcept("/tables/customers.md", { type: "Table", title: "Customers" }, "Customers have billing charges.", "add");
}

const updateInstruction = "Update the Billing API concept: support tooling creates ad-hoc charges only after approval. Replace the old unconditional claim.";

function patchRequest(request: MutationRequest, body = newBody) {
  return request.tools.patch_concept.execute({
    path: "/apis/billing-api.md",
    replace_body: body,
    log_summary: "Updated [Billing API](/apis/billing-api.md) approval rule.",
  }, toolContext);
}

describe("mutation owner preflight", () => {
  it("pre-reads a dominant owner and authorizes an unchanged complete-body replacement", async () => {
    await billingFixture();
    generateTextMock.mockImplementationOnce(async (request: MutationRequest) => {
      expect(request.prompt).toContain("DETERMINISTIC PREFLIGHT");
      expect(request.prompt).toContain(oldBody);
      await patchRequest(request);
      return { text: "Updated Billing API approval rule.", steps: [{ toolCalls: [] }] };
    });

    const result = await runMutation(kb, updateInstruction, { preflightInput: updateInstruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body.trim()).toBe(newBody);
  });

  it("does not authorize a replacement when preflight is disabled", async () => {
    await billingFixture();
    vi.stubEnv("MUTATION_PREFLIGHT", "false");
    generateTextMock.mockImplementationOnce(async (request: MutationRequest) => {
      expect(request.prompt).not.toContain("DETERMINISTIC PREFLIGHT");
      await expect(patchRequest(request)).rejects.toThrow("complete, unchanged read");
      return { text: "No write occurred.", steps: [{ toolCalls: [] }] };
    });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body.trim()).toBe(oldBody);
  });

  it("uses raw add content rather than wrapper instructions for the search", async () => {
    await billingFixture();
    const search = vi.spyOn(kb, "search");
    generateTextMock.mockResolvedValueOnce({ text: "Already present.", steps: [{ toolCalls: [] }] });
    await runMutation(kb, "Persist the following knowledge into the knowledge base.\nKNOWLEDGE: billing", {
      preflightInput: "Billing API support tooling creates ad-hoc charges.",
    });
    expect(search).toHaveBeenCalledTimes(1);
    expect(search.mock.calls[0][0]).toContain("billing api support tooling");
    expect(search.mock.calls[0][0]).not.toContain("persist");
  });

  it("leaves close competing owners to the ordinary read path", async () => {
    await kb.writeConcept("/facts/a.md", { type: "Fact", title: "Deploy schedule" }, "Current deployment day is Monday.", "add");
    await kb.writeConcept("/facts/b.md", { type: "Fact", title: "Deploy schedule" }, "Current deployment day is Friday.", "add");
    generateTextMock.mockImplementationOnce(async (request: MutationRequest) => {
      expect(request.prompt).not.toContain("DETERMINISTIC PREFLIGHT");
      return { text: "Need to inspect both facts.", steps: [{ toolCalls: [] }] };
    });
    await runMutation(kb, "Update current deployment schedule day.");
  });

  it("does not authorize oversized evidence or a stale pre-read", async () => {
    await billingFixture();
    await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, `${oldBody}${" more details.".repeat(1200)}`, "expand");
    generateTextMock.mockImplementationOnce(async (request: MutationRequest) => {
      expect(request.prompt).not.toContain("DETERMINISTIC PREFLIGHT");
      await expect(patchRequest(request)).rejects.toThrow("complete, unchanged read");
      return { text: "Need a paged read.", steps: [{ toolCalls: [] }] };
    });
    await runMutation(kb, updateInstruction);

    await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, oldBody, "reset");
    generateTextMock.mockImplementationOnce(async (request: MutationRequest) => {
      expect(request.prompt).toContain("DETERMINISTIC PREFLIGHT");
      await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, "Externally updated body.", "external");
      await expect(patchRequest(request)).rejects.toThrow("complete, unchanged read");
      return { text: "Stale read; no write.", steps: [{ toolCalls: [] }] };
    });
    await runMutation(kb, updateInstruction);
    expect((await kb.readConcept("/apis/billing-api.md")).body.trim()).toBe("Externally updated body.");
  });

  it("propagates cancellation rather than falling through to a write", async () => {
    const controller = new AbortController();
    vi.spyOn(kb, "search").mockImplementationOnce(async () => {
      controller.abort(new DOMException("cancelled", "AbortError"));
      return [];
    });
    await expect(runMutation(kb, "Update billing policy.", { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect(generateTextMock).not.toHaveBeenCalled();
  });
});
