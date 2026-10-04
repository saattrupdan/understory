import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const generateTextMock = vi.hoisted(() => vi.fn());
const generateObjectMock = vi.hoisted(() => vi.fn());
vi.mock("ai", async () => ({
  ...(await vi.importActual<typeof import("ai")>("ai")),
  generateText: generateTextMock,
  generateObject: generateObjectMock,
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
  generateObjectMock.mockReset();
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
  const staged = () => vi.stubEnv("MUTATION_STAGED", "true");
  const baseProposal = {
    path: "/apis/billing-api.md", old_text: "", new_text: "", body: "",
    frontmatter: { type: "", title: "", description: "" }, claim: "", reason: "",
  };

  it("requires a verified claim matching the requested knowledge for a no-op", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "noop", claim: oldBody.slice(0, 37),
    } });
    const unrelated = await runMutation(kb, "Remember that QA enables Beta on Tuesdays.");
    expect(unrelated).toMatchObject({ ok: false, status: "failed" });
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "noop", claim: oldBody,
    } });
    const duplicate = await runMutation(kb, `Remember that ${oldBody}`);
    expect(duplicate).toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("rejects a no-op if the observed body changes before final verification", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, "External revision.", "external");
      return { object: { ...baseProposal, action: "noop", claim: oldBody } };
    });
    expect(await runMutation(kb, `Remember that ${oldBody}`)).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body.trim()).toBe("External revision.");
  });

  it("appends an independently approved fact without changing prior claims", async () => {
    await billingFixture(); staged();
    const newFact = "The Billing API logs a request ID for each charge created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "append", new_text: newFact,
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "No conflicting statement in the owner." } });
    const result = await runMutation(kb, `Remember that ${newFact} This is a detail of the existing Billing API.`);
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    const body = (await kb.readConcept("/apis/billing-api.md")).body;
    expect(body).toContain(oldBody);
    expect(body).toContain(newFact);
  });

  it("rejects a contradictory append even if the model's verifier would approve it", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "append", new_text: "Billing API support tooling creates ad-hoc charges only after an approval check.",
    } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("defers append when an independent check finds a conflict", async () => {
    await billingFixture(); staged();
    const newFact = "The Billing API logs a request ID for each charge created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "append", new_text: newFact,
    } }).mockResolvedValueOnce({ object: { safe: false, reason: "Conflicting owner assertion." } });
    const result = await runMutation(kb, `Remember that ${newFact} This is a detail of the existing Billing API.`);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body).not.toContain(newFact);
  });

  it("creates a distinct concept without modifying a merely mentioned unrelated owner", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "create", path: "/policies/qa-beta-toggle.md",
      frontmatter: { type: "Policy", title: "QA Beta Feature Toggle", description: "QA Beta schedule" },
      body: "QA enables the Beta feature toggle on Tuesdays. It is unrelated to the Billing API.",
    } });
    const result = await runMutation(kb, "Record a distinct policy: QA enables the Beta feature toggle on Tuesdays. It is unrelated to the Billing API.");
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/policies/qa-beta-toggle.md"] } });
    expect((await kb.readConcept("/policies/qa-beta-toggle.md")).body).toContain("Tuesdays");
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("rejects creating a duplicate claim under a different title", async () => {
    await billingFixture(); staged();
    await kb.writeConcept("/policies/rollout-schedule.md", { type: "Policy", title: "Rollout Schedule" }, "QA enables the Beta feature toggle on Tuesdays.", "add");
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "create", path: "/policies/qa-beta-toggle.md",
      frontmatter: { type: "Policy", title: "QA Beta Feature Toggle", description: "QA Beta schedule" },
      body: "QA enables the Beta feature toggle on Tuesdays. It is unrelated to the Billing API.",
    } });
    expect(await runMutation(kb, "Record a distinct policy: QA enables the Beta feature toggle on Tuesdays. It is unrelated to the Billing API.")).toMatchObject({ ok: false, status: "failed" });
    await expect(kb.readConcept("/policies/qa-beta-toggle.md")).rejects.toThrow("not found");
  });

  it("does not claim a failed exclusive create wrote an already-existing identical body", async () => {
    await billingFixture(); staged();
    const body = "QA enables the Beta feature toggle on Tuesdays. It is unrelated to the Billing API.";
    await kb.writeConcept("/policies/qa-beta-toggle.md", { type: "Policy", title: "Misc" }, body, "external");
    vi.spyOn(kb, "search").mockResolvedValue([]);
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "create", path: "/policies/qa-beta-toggle.md",
      frontmatter: { type: "Policy", title: "QA Beta Feature Toggle", description: "QA Beta schedule" }, body,
    } });
    const result = await runMutation(kb, "Record a distinct policy: QA enables the Beta feature toggle on Tuesdays. It is unrelated to the Billing API.");
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/policies/qa-beta-toggle.md")).body).toContain(body);
  });

  it("reports a partial write when post-write indexing fails", async () => {
    await billingFixture(); staged();
    const original = kb.patchConcept.bind(kb);
    vi.spyOn(kb, "patchConcept").mockImplementation(async (...args) => {
      await original(...args);
      throw new Error("indexing failed after write");
    });
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "replace", old_text: oldBody,
      new_text: "Monthly charges are scheduled; support tooling creates ad-hoc charges only after approval.",
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "Correction supported." } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "partial", filesChanged: ["/apis/billing-api.md"] });
  });

  it("defers a replacement rejected by the independent consistency check", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "replace", old_text: oldBody,
      new_text: "Monthly charges are scheduled; support tooling creates ad-hoc charges only after approval.",
    } }).mockResolvedValueOnce({ object: { safe: false, reason: "Contradicts a different assertion." } });
    expect(await runMutation(kb, updateInstruction)).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("rejects an unrelated second edit even if the model would approve it", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "replace", old_text: oldBody,
      new_text: "Monthly charges are NOT scheduled; support tooling creates ad-hoc charges only after approval.",
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "Model overlooked the extra negation." } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("rejects an unsupported addition inside the changed clause", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "replace", old_text: oldBody,
      new_text: "Monthly charges are scheduled; support tooling creates ad-hoc charges only after approval and records customer SSNs.",
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "Model overlooked the added assertion." } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("does not accept a negated request as an unrelated exact no-op", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "noop", claim: oldBody } });
    const result = await runMutation(kb, "Remember that monthly charges are not scheduled and ad-hoc charges come from support tooling.");
    expect(result).toMatchObject({ ok: false, status: "failed" });
  });

  it("keeps complete staged evidence independent of the generic tool-result cap", async () => {
    await billingFixture(); staged();
    vi.stubEnv("AGENT_MAX_TOOL_RESULT_CHARS", "260");
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(oldBody);
      return { object: { ...baseProposal, action: "defer", reason: "No safe edit." } };
    });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("rejects a stale exact replacement after the owner changed", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, "External revision.", "external");
      return { object: { ...baseProposal, action: "replace", old_text: oldBody, new_text: newBody } };
    }).mockResolvedValueOnce({ object: { safe: true, reason: "Proposed correction is supported." } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body.trim()).toBe("External revision.");
  });

  it("defers an oversized owner before asking the model", async () => {
    await billingFixture(); staged();
    await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, `${oldBody}${" more details".repeat(5_500)}`, "expand");
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("does not write after invalid structured output or cancellation", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockRejectedValueOnce(new Error("No object generated"));
    expect(await runMutation(kb, updateInstruction)).toMatchObject({ ok: false, status: "failed" });
    const controller = new AbortController();
    vi.spyOn(kb, "search").mockImplementationOnce(async () => {
      controller.abort(new DOMException("cancelled", "AbortError")); return [];
    });
    await expect(runMutation(kb, updateInstruction, { signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("checks the complete body of a large hinted owner before appending", async () => {
    await billingFixture(); staged();
    const tail = "Last section: billing request IDs remain meaningful for support.";
    await kb.patchConcept("/apis/billing-api.md", {
      replaceBody: `${oldBody}\n\n${"Background billing details. ".repeat(750)}\n\n${tail}\n`,
    }, "expand");
    const fact = "Billing API records an audit request ID for each support-created charge.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(tail);
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, `Remember that ${fact}`, { ownerHint: "/apis/billing-api.md", preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    const body = (await kb.readConcept("/apis/billing-api.md")).body;
    expect(body).toContain(tail);
    expect(body).toContain(fact);
  });

  it("reads an existing owner hint outside ranked search results but never a missing path", async () => {
    await billingFixture(); staged();
    const searchHit = (path: string, score: number) => ({
      path, type: "Fact", title: path, score, confidence: 50,
      confidenceQualified: true, matchedGroups: 2, contentGroups: 2,
      distinctiveGroups: 2, exactCompoundGroups: 0,
    });
    vi.spyOn(kb, "search").mockResolvedValue([
      searchHit("/apis/billing-api.md", 100),
      searchHit("/playbooks/oncall-billing.md", 99),
    ]);
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("OWNER /playbooks/oncall-billing.md");
      return { object: { safe: false, reason: "No safe edit." } };
    });
    expect(await runMutation(kb, "Remember a billing incident response fact.", { ownerHint: "/playbooks/oncall-billing.md", preflightInput: "Remember a billing incident response fact.", directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("OWNER /tables/customers.md");
      return { object: { safe: false, reason: "Owner does not fit." } };
    });
    expect(await runMutation(kb, "Remember a Customers billing incident response fact.", { ownerHint: "/tables/customers.md", preflightInput: "Remember a Customers billing incident response fact.", directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    generateObjectMock.mockResolvedValueOnce({ object: { path: "" } })
      .mockResolvedValueOnce({ object: { ...baseProposal, action: "defer", reason: "No owner." } });
    expect(await runMutation(kb, "Remember a billing incident response fact.", { ownerHint: "/other/unsearched.md", preflightInput: "Remember a billing incident response fact.", directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("appends to a relevant existing hinted owner even when broad search ranks another", async () => {
    await billingFixture(); staged();
    const newFact = "Billing on-call responders check the charge request ID before escalation.";
    const search = kb.search.bind(kb);
    vi.spyOn(kb, "search").mockImplementation(async (query, options) =>
      (await search(query, options)).filter((hit) => hit.path !== "/playbooks/oncall-billing.md"));
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true, reason: "On-call fact belongs in this runbook." } });
    const result = await runMutation(kb, `Remember that ${newFact}`, { ownerHint: "/playbooks/oncall-billing.md", preflightInput: newFact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/playbooks/oncall-billing.md"] } });
    expect((await kb.readConcept("/playbooks/oncall-billing.md")).body).toContain(newFact);
  });

  it("selects an unhinted owner without rewriting the exact caller fact", async () => {
    await billingFixture(); staged();
    const fact = "Billing API logs a request ID for each charge created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: "/apis/billing-api.md" } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, `Persist this knowledge: ${fact}`, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("does not confuse incidental update language in an add with correction intent", async () => {
    await billingFixture(); staged();
    const fact = "Billing API reporting update uses request IDs rather than log offsets for support-created charges.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, `Persist this knowledge: ${fact}`, {
      ownerHint: "/apis/billing-api.md", preflightInput: fact, directAdd: true,
    });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("keeps explicit memory_add corrections out of the exact-append path", async () => {
    await billingFixture(); staged();
    const fact = "Correct Billing API: replace the unconditional support-charge statement with approval required.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("Propose exactly one safe knowledge-base mutation");
      return { object: { ...baseProposal, action: "defer", reason: "Requires exact correction." } };
    });
    expect(await runMutation(kb, fact, { ownerHint: "/apis/billing-api.md", preflightInput: fact, directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(oldBody);
  });

  it("matches a hyphenated entity to its owner slug on an unhinted add", async () => {
    staged();
    const owner = "/repos/pi-agent/understory-async-write-feasibility.md";
    await kb.writeConcept(owner, { type: "Concept", title: "Understory Async Write Feasibility" },
      "Pi Memory-Async has a persistent write queue and a paused worker.", "fixture");
    const fact = "Pi Memory-Async canary-once pauses the worker after one MCP attempt.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: owner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, `Persist this knowledge: ${fact}`, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("verifies an identical hinted fact without a duplicate write", async () => {
    await billingFixture(); staged();
    const fact = "Billing API logs request IDs for support-created charges.";
    await kb.patchConcept("/apis/billing-api.md", { replaceBody: `${oldBody}\n\n${fact}\n` }, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { supported: true } });
    const result = await runMutation(kb, `Remember that ${fact}`, { ownerHint: "/apis/billing-api.md", preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept("/apis/billing-api.md")).body.split(fact)).toHaveLength(2);
  });

  it("does not mistake a retracted substring for an established no-op", async () => {
    await billingFixture(); staged();
    const fact = "Billing API logs request IDs for each support-created charge.";
    await kb.patchConcept("/apis/billing-api.md", {
      replaceBody: `${fact}\n\nThat statement is obsolete: request IDs are no longer logged.`,
    }, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { supported: false } });
    const result = await runMutation(kb, `Remember that ${fact}`, { ownerHint: "/apis/billing-api.md", preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain("no longer logged");
  });

  it("rejects a wrong hinted owner when independent evidence disapproves", async () => {
    await billingFixture(); staged();
    const fact = "Billing API logs a request ID for each charge created by support tooling.";
    const result = await runMutation(kb, `Remember that ${fact}`, { ownerHint: "/tables/customers.md", preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
    expect((await kb.readConcept("/tables/customers.md")).body).not.toContain(fact);
  });

  it("applies a caller-quoted correction with one verified model call", async () => {
    await billingFixture(); staged();
    const instruction = "Correct `/apis/billing-api.md`: replace the exact phrase `ad-hoc charges come from support tooling.` with `ad-hoc charges come from support tooling only after approval.` Preserve monthly charges.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(oldBody);
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    const body = (await kb.readConcept("/apis/billing-api.md")).body;
    expect(body).toContain("Monthly charges are scheduled");
    expect(body).toContain("only after approval");
  });

  it("rejects a quoted correction when the owner changes before the write", async () => {
    await billingFixture(); staged();
    const instruction = "Correct `/apis/billing-api.md`: replace the exact phrase `ad-hoc charges come from support tooling.` with `ad-hoc charges come from support tooling only after approval.`";
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept("/apis/billing-api.md", { type: "API Endpoint", title: "Billing API" }, "External revision.", "external");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, instruction, { preflightInput: instruction })).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body.trim()).toBe("External revision.");
  });

  it("stages an exact correction and removes the old claim", async () => {
    await billingFixture();
    vi.stubEnv("MUTATION_STAGED", "true");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: "/apis/billing-api.md",
      old_text: "ad-hoc charges come from support tooling.",
      new_text: "support tooling creates ad-hoc charges only after approval.",
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "Correction supported." } });
    const result = await runMutation(kb, updateInstruction);
    expect(result.ok).toBe(true);
    const concept = await kb.readConcept("/apis/billing-api.md");
    expect(concept.body).not.toContain("ad-hoc charges come from support tooling.");
    expect(concept.body).toContain("only after approval");
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });
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
