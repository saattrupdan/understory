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

  it("retracts only one exact, complete paragraph from the explicitly named owner", async () => {
    staged();
    const paragraph = "Claes Jev paragraph states one sentence. It contains a second exact sentence.";
    const retained = "The synthetic dataset generator writes deterministic records.";
    await kb.writeConcept("/projects/fs-fotovalidering/synthetic-dataset-generation.md", {
      type: "Project", title: "Synthetic Dataset Generation",
    }, `${retained}\n\n${paragraph}\n\nAnother unrelated paragraph remains.`, "add");
    const result = await runMutation(kb, `Retract exact paragraph from \`/projects/fs-fotovalidering/synthetic-dataset-generation.md\`:\n<<<\n${paragraph}\n>>>`);
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/projects/fs-fotovalidering/synthetic-dataset-generation.md"] } });
    expect((await kb.readConcept("/projects/fs-fotovalidering/synthetic-dataset-generation.md")).body)
      .toBe(`${retained}\n\nAnother unrelated paragraph remains.\n`);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("retracts an exact final paragraph with the normal body trailing newline", async () => {
    staged();
    const ownerPath = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    const paragraph = "Claes Jev diagnostics were filed under the wrong owner. This sentence is part of the same paragraph.";
    await kb.writeConcept(ownerPath, { type: "Project", title: "Synthetic Dataset Generation" },
      `Unrelated existing fact.\n\n${paragraph}\n`, "seed");
    const result = await runMutation(kb, `Retract exact paragraph from \`${ownerPath}\`:\n<<<\n${paragraph}\n>>>`);
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [ownerPath] } });
    expect((await kb.readConcept(ownerPath)).body).toBe("Unrelated existing fact.\n");
  });

  it("refuses stale, ambiguous, nonexact, and mismatched-path retractions", async () => {
    staged();
    const paragraph = "This accidentally filed paragraph has two exact sentences. It should be removed verbatim.";
    const ownerPath = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    const instruction = `Retract exact paragraph from \`${ownerPath}\`:\n<<<\n${paragraph}\n>>>`;
    await kb.writeConcept(ownerPath, { type: "Project", title: "Synthetic Dataset Generation" }, `Before.\n\n${paragraph}\n\nAfter.`, "add");
    const originalRead = kb.readConcept.bind(kb);
    let reads = 0;
    vi.spyOn(kb, "readConcept").mockImplementation(async (pathName) => {
      reads += 1;
      if (reads === 2) {
        await kb.writeConcept(ownerPath, { type: "Project", title: "Synthetic Dataset Generation" }, "Concurrent external revision.", "external");
      }
      return originalRead(pathName);
    });
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    vi.restoreAllMocks();

    await kb.writeConcept(ownerPath, { type: "Project", title: "Synthetic Dataset Generation" }, `Before.\n\n${paragraph}\n\n${paragraph}\n\nAfter.`, "external");
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    await kb.writeConcept(ownerPath, { type: "Project", title: "Synthetic Dataset Generation" }, `Before.\n\n${paragraph} Extra words.\n\nAfter.`, "external");
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    await kb.writeConcept("/projects/other.md", { type: "Project", title: "Other" }, paragraph, "add");
    expect(await runMutation(kb, instruction, { ownerHint: "/projects/other.md" })).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

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

  it("stores an exact verified fact in a deterministic continuation when the owner is full", async () => {
    await billingFixture(); staged();
    const ownerPath = "/apis/billing-api.md";
    const fact = "Billing API records an immutable support-charge audit reference for every transaction.";
    const ownerBody = `${oldBody}\n\n${"Existing verified billing details. ".repeat(2_100)}`.slice(0, 63_990);
    await kb.patchConcept(ownerPath, { replaceBody: ownerBody }, "fill owner");
    const savedOwnerBody = (await kb.readConcept(ownerPath)).body;
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    const options = { ownerHint: ownerPath, preflightInput: fact, directAdd: true };
    const result = await runMutation(kb, `Remember that ${fact}`, options);
    expect(result).toMatchObject({ ok: true });
    const changed = (result as { result?: { filesChanged?: string[] } }).result?.filesChanged ?? [];
    expect(changed).toHaveLength(1);
    expect(changed[0]).toMatch(/^\/apis\/billing-api-continuation-[a-f0-9]{16}\.md$/);
    const continuation = await kb.readConcept(changed[0]);
    expect(continuation.body).toContain(fact);
    expect(continuation.body).toContain(`[Billing API](${ownerPath})`);
    expect(continuation.frontmatter.title).toContain("Billing API");
    expect((await kb.readConcept(ownerPath)).body).toBe(savedOwnerBody);

    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    const replay = await runMutation(kb, `Remember that ${fact}`, options);
    expect(replay).toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await fs.readdir(path.join(root, "apis"))).filter((name) => name.includes("continuation"))).toHaveLength(1);
  });

  it("does not append a differently named project's generic dataset fact to the selected owner", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    await kb.writeConcept(owner, { type: "Project", title: "Synthetic Dataset Generation" }, "Synthetic datasets are generated for FS Fotovalidering.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: owner, type: "Project", title: "Synthetic Dataset Generation",
      score: 10, confidence: 10, confidenceQualified: false,
    }]);
    const fact = "Claes Jev diagnostics use the dataset validation pipeline for their independent project.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: owner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("creates an exact fact at a distinct missing hinted path despite generic unrelated search hits", async () => {
    staged();
    const unrelated = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    await kb.writeConcept(unrelated, { type: "Project", title: "Synthetic Dataset Generation" },
      "Synthetic datasets are generated for FS Fotovalidering.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: unrelated, type: "Project", title: "Synthetic Dataset Generation",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const hintedPath = "/projects/claes-jev-diagnostics.md";
    const fact = "Claes Jev diagnostics use a dedicated calibration process for independent validation.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("FULL EXISTING SEARCH EVIDENCE");
      expect(request.prompt).toContain("Synthetic datasets are generated");
      expect(request.prompt).toContain(fact);
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, fact, { ownerHint: hintedPath, preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [hintedPath] } });
    expect((await kb.readConcept(hintedPath)).body).toBe(`${fact}\n`);
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
  });

  it("creates a verified owner for an unhinted named application without using an unrelated rate-limit hit", async () => {
    staged();
    const unrelated = "/repos/pi-agent/rate-limit-retry.md";
    await kb.writeConcept(unrelated, { type: "Extension", title: "Pi Agent Rate Limit Retry" },
      "Pi retries provider HTTP 429 responses with exponential backoff.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: unrelated, type: "Extension", title: "Pi Agent Rate Limit Retry",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "Job-bot account rate limits deployed on main at bf272c6 (Oct 2026): PostgreSQL rate_limit_reservations keyed by hash of verified Auth0 sub.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: true } })
      .mockImplementationOnce(async (request: { prompt: string }) => {
        expect(request.prompt).toContain("/repos/job-bot.md");
        expect(request.prompt).toContain("Pi retries provider HTTP 429");
        return { object: { safe: true } };
      });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/repos/job-bot.md"] } });
    expect((await kb.readConcept("/repos/job-bot.md")).body).toBe(`${fact}\n`);
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
  });

  it("does not infer a repository from a hyphenated policy belonging to another application", async () => {
    staged();
    const unrelated = "/repos/pi-agent/rate-limit-retry.md";
    await kb.writeConcept(unrelated, { type: "Extension", title: "Rate Limit Retry" },
      "Pi retries provider HTTP 429 responses.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: unrelated, type: "Extension", title: "Rate Limit Retry",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "Rate-limit policy for Billing API deployed on main at bf272c6: reservations enforce per-account quotas.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: false } })
      .mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(await kb.bundle.exists("/repos/rate-limit.md")).toBe(false);
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
  });

  it("rejects an unhinted application creation if independent evidence fits an existing owner", async () => {
    staged();
    const existing = "/servers/job-bot.md";
    await kb.writeConcept(existing, { type: "Server", title: "Job-bot" },
      "Job-bot hosts the account rate-limit database.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: existing, type: "Server", title: "Job-bot", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const fact = "Job-bot account rate limits deployed on main at bf272c6: PostgreSQL reservations enforce per-account quotas.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: true } })
      .mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(await kb.bundle.exists("/repos/job-bot.md")).toBe(false);
    expect((await kb.readConcept(existing)).body).not.toContain(fact);
  });

  it("uses an existing unhinted application owner outside search ranking with independent verification", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Job-bot" },
      "Job-bot builds job applications.", "seed");
    await kb.writeConcept("/repos/pi-agent/rate-limit-retry.md", { type: "Extension", title: "Rate Limit Retry" },
      "Pi retries provider HTTP 429 responses.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: "/repos/pi-agent/rate-limit-retry.md", type: "Extension", title: "Rate Limit Retry",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "Job-bot account rate limits deployed on main at bf272c6: PostgreSQL reservations enforce per-account quotas.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: true } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("rejects a missing hinted path unrelated to the exact fact", async () => {
    staged();
    await kb.writeConcept("/projects/general-datasets.md", { type: "Project", title: "General Datasets" },
      "General dataset notes.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: "/projects/general-datasets.md", type: "Project", title: "General Datasets",
      score: 10, confidence: 10, confidenceQualified: false,
    }]);
    const fact = "Claes Jev diagnostics use a dedicated calibration process for independent validation.";
    expect(await runMutation(kb, fact, {
      ownerHint: "/projects/unrelated-release-notes.md", preflightInput: fact, directAdd: true,
    })).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
    expect(await kb.bundle.exists("/projects/unrelated-release-notes.md")).toBe(false);
  });

  it("does not create a missing hinted concept when the independent check detects a duplicate or unsafe fact", async () => {
    staged();
    const existing = "/projects/claes-jev-history.md";
    await kb.writeConcept(existing, { type: "Project", title: "Claes Jev History" },
      "Claes Jev diagnostics use a dedicated calibration process for independent validation.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: existing, type: "Project", title: "Claes Jev History", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const hintedPath = "/projects/claes-jev-diagnostics.md";
    const fact = "Claes Jev diagnostics use a dedicated calibration process for independent validation.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { ownerHint: hintedPath, preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(await kb.bundle.exists(hintedPath)).toBe(false);
    expect((await kb.readConcept(existing)).body.match(/dedicated calibration/g)).toHaveLength(1);
  });

  it("does not create a missing hinted concept when the fact fails the independent safety check", async () => {
    staged();
    const existing = "/projects/reference.md";
    await kb.writeConcept(existing, { type: "Project", title: "Reference" }, "Reference context.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: existing, type: "Project", title: "Reference", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const hintedPath = "/projects/claes-jev-diagnostics.md";
    const fact = "Claes Jev diagnostics use a dedicated calibration process for independent validation.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { ownerHint: hintedPath, preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(await kb.bundle.exists(hintedPath)).toBe(false);
  });

  it("checks every complete competing body separately when new-concept evidence exceeds the prompt budget", async () => {
    staged();
    const paths = ["/projects/unrelated-a.md", "/projects/unrelated-b.md"];
    for (const [index, ownerPath] of paths.entries()) {
      await kb.writeConcept(ownerPath, { type: "Project", title: `Unrelated ${index}` },
        `${"Other project details. ".repeat(2_300)}\nEND-OF-OWNER-${index}`, "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue(paths.map((ownerPath, index) => ({
      path: ownerPath, type: "Project", title: `Unrelated ${index}`,
      score: 10 - index, confidence: 10 - index, confidenceQualified: false,
    })));
    const fact = "Claes Jev diagnostics use a dedicated calibration process for independent validation.";
    const prompts: string[] = [];
    generateObjectMock.mockImplementation(async (request: { prompt: string }) => {
      prompts.push(request.prompt);
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, fact, {
      ownerHint: "/projects/claes-jev-diagnostics.md", preflightInput: fact, directAdd: true,
    });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/projects/claes-jev-diagnostics.md"] } });
    expect(prompts).toHaveLength(2);
    expect(prompts[0]).toContain("END-OF-OWNER-0");
    expect(prompts[1]).toContain("END-OF-OWNER-1");
  });

  it("does not overwrite a missing hinted path created concurrently after safety approval", async () => {
    staged();
    const existing = "/projects/reference.md";
    await kb.writeConcept(existing, { type: "Project", title: "Reference" }, "Reference context.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: existing, type: "Project", title: "Reference", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const hintedPath = "/projects/claes-jev-diagnostics.md";
    const fact = "Claes Jev diagnostics use a dedicated calibration process for independent validation.";
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept(hintedPath, { type: "Project", title: "Concurrent" }, "Concurrent content.", "race");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { ownerHint: hintedPath, preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(hintedPath)).body).toBe("Concurrent content.\n");
  });

  it("allows a fact explicitly naming the selected project despite generic dataset terms", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    await kb.writeConcept(owner, { type: "Project", title: "Synthetic Dataset Generation" }, "Synthetic datasets are generated for FS Fotovalidering.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: owner, type: "Project", title: "Synthetic Dataset Generation",
      score: 10, confidence: 10, confidenceQualified: false,
    }]);
    const fact = "Synthetic Dataset Generation diagnostics use the dataset validation pipeline.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: owner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("routes an unrelated unhinted continuation to its relevant parent", async () => {
    await billingFixture(); staged();
    const wrongPath = "/apis/billing-api-continuation-abc123abc123abcd.md";
    await kb.writeConcept(wrongPath, { type: "Concept", title: "Billing API Legacy — Continuation abc123" },
      "The legacy billing project was retired.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: wrongPath, type: "Concept", title: "Billing API Legacy — Continuation abc123",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "Billing API logs a request ID for each new support charge.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: wrongPath } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
    expect((await kb.readConcept(wrongPath)).body).not.toContain(fact);
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(fact);
  });

  it("appends to a selected continuation that already records the fact's item ID", async () => {
    await billingFixture(); staged();
    const continuation = "/apis/billing-api-continuation-abc123abc123abcd.md";
    await kb.writeConcept(continuation, { type: "Concept", title: "Billing API — Continuation abc123" },
      "The support test for billing_case9 was recorded here.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: continuation, type: "Concept", title: "Billing API — Continuation abc123",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "Billing API billing_case9 records an additional verified support observation.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: continuation } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: true, result: { filesChanged: [continuation] },
    });
    expect((await kb.readConcept(continuation)).body).toContain(fact);
  });

  it("does not create a continuation after the owner changes during verification", async () => {
    await billingFixture(); staged();
    const ownerPath = "/apis/billing-api.md";
    const fact = "Billing API records an immutable support-charge audit reference for every transaction.";
    const ownerBody = `${oldBody}\n\n${"Existing verified billing details. ".repeat(2_100)}`.slice(0, 63_990);
    await kb.patchConcept(ownerPath, { replaceBody: ownerBody }, "fill owner");
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept(ownerPath, { type: "API Endpoint", title: "Billing API" }, "External revision", "external");
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, `Remember that ${fact}`, {
      ownerHint: ownerPath, preflightInput: fact, directAdd: true,
    });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await fs.readdir(path.join(root, "apis"))).some((name) => name.includes("continuation"))).toBe(false);
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
    expect(await runMutation(kb, "Remember a billing incident response fact.", { ownerHint: "/other/unsearched.md", preflightInput: "Remember a billing incident response fact.", directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
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

  it("routes the failed Holi possessive-framing fact to its named owner without a hint", async () => {
    staged();
    const owner = "/repos/holi.md";
    await kb.writeConcept(owner, { type: "Project", title: "Holi" },
      "Holi maintains the hereinmygarage vault.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: owner, type: "Project", title: "Holi", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const fact = "For Holi's hereinmygarage vault, Dan prefers the existing `report` PDF template to `syv-report` and asked to remove `syv-report`. On 15 Oct 2026 the `syv-report` template and its dedicated `_brand/fonts/syv-report` fonts were removed from the Holi-managed vault, committed/pushed as 0ef1810, and the separate gitsky checkout was fast-forwarded. When suggesting or generating report PDFs for this vault, use `report`.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: owner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("does not treat a single-word author as a conflicting named owner", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering" },
      "FS Fotovalidering has an annotation review interface.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: owner, type: "Project", title: "FS Fotovalidering", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const fact = "Dan prefers the FS Fotovalidering annotation buttons to remain visible during review.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: owner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("rejects a different named subject even when the selected owner is otherwise relevant", async () => {
    staged();
    const owner = "/repos/holi.md";
    await kb.writeConcept(owner, { type: "Project", title: "Holi" },
      "Holi maintains the hereinmygarage vault.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: owner, type: "Project", title: "Holi", score: 10, confidence: 10,
      confidenceQualified: false,
    }]);
    const fact = "For Claes Jev's hereinmygarage vault, the backup rotation retains verified snapshots for 30 days.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: owner } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
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

  it("selects one owner from bounded metadata when competing complete bodies exceed the prompt budget", async () => {
    staged();
    const api = "/apis/billing-api.md";
    const playbook = "/playbooks/oncall-billing.md";
    const table = "/tables/customers.md";
    const tail = "Final Billing API constraint: support charges require approval.";
    await kb.writeConcept(api, { type: "API Endpoint", title: "Billing API", description: "Creates customer charges" },
      `${"Background billing API context. ".repeat(1_150)}\n${tail}`, "seed");
    await kb.writeConcept(playbook, { type: "Playbook", title: "Billing On-call", description: "Handles charge failures" },
      "On-call incident context. ".repeat(1_400), "seed");
    await kb.writeConcept(table, { type: "Table", title: "Customers", description: "Customer records" },
      "Customer table context. ".repeat(1_400), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([api, playbook, table].map((conceptPath) => ({
      path: conceptPath, type: "Concept", title: conceptPath === api ? "Billing API" : conceptPath === playbook ? "Billing On-call" : "Customers",
      description: conceptPath === api ? "Creates customer charges" : conceptPath === playbook
        ? ({ unexpected: "non-string metadata" } as unknown as string) : "Other billing context",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Billing API logs request IDs for charges created by support tooling; Billing On-call is a competing owner for this request.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("CANDIDATE METADATA");
      expect(request.prompt).toContain("Billing On-call");
      expect(request.prompt).not.toContain("non-string metadata");
      expect(request.prompt).not.toContain(tail);
      return { object: { path: api } };
    }).mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(tail);
      expect(request.prompt).toContain("Billing On-call");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: true, result: { filesChanged: [api] },
    });
    expect((await kb.readConcept(api)).body).toContain(tail);
    expect((await kb.readConcept(api)).body).toContain(fact);
    expect((await kb.readConcept(playbook)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("defers empty or wrong aggregate-budget selections without a write", async () => {
    staged();
    const api = "/apis/billing-api.md";
    const playbook = "/playbooks/oncall-billing.md";
    await kb.writeConcept(api, { type: "API Endpoint", title: "Billing API" }, "Billing API context. ".repeat(2_300), "seed");
    await kb.writeConcept(playbook, { type: "Playbook", title: "Billing On-call" }, "On-call context. ".repeat(2_700), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([api, playbook].map((conceptPath) => ({
      path: conceptPath, type: "Concept", title: conceptPath === api ? "Billing API" : "Billing On-call",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Billing logs request IDs for charges created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: "" } })
      .mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
    generateObjectMock.mockResolvedValueOnce({ object: { path: playbook } })
      .mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
    expect((await kb.readConcept(api)).body).not.toContain(fact);
    expect((await kb.readConcept(playbook)).body).not.toContain(fact);
  });

  it("chooses a relevant owner when the metadata selector abstains on an equal fit", async () => {
    staged();
    const api = "/apis/billing-api.md";
    const playbook = "/playbooks/oncall-billing.md";
    await kb.writeConcept(api, { type: "API Endpoint", title: "Billing API" }, "Billing API context. ".repeat(2_300), "seed");
    await kb.writeConcept(playbook, { type: "Playbook", title: "Billing On-call" }, "On-call context. ".repeat(2_700), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([api, playbook].map((conceptPath) => ({
      path: conceptPath, type: "Concept", title: conceptPath === api ? "Billing API" : "Billing On-call",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Billing logs request IDs for charges created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: "" } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: true, result: { filesChanged: [api] },
    });
    expect((await kb.readConcept(api)).body).toContain(fact);
    expect((await kb.readConcept(playbook)).body).not.toContain(fact);
  });

  it.each([3, 4])("re-reads and accepts the selected metadata hit at rank %s", async (rank) => {
    staged();
    const paths = Array.from({ length: 5 }, (_, i) => `/projects/candidate-${i + 1}.md`);
    const titles = ["Project Alpha", "Project Beta", "Project Gamma", "FS Fotovalidering Godkendt ID9", "FS Fotovalidering Backup ID8"];
    for (let i = 0; i < paths.length; i++) {
      const body = i === rank ? "The authoritative SHA is old-value-id9." : `Candidate ${i + 1} context. `.repeat(2_100);
      await kb.writeConcept(paths[i], { type: "Project", title: titles[i] }, body, "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue(paths.map((conceptPath, i) => ({
      path: conceptPath, type: "Project", title: titles[i], score: 10, confidence: 10, confidenceQualified: false,
    })));
    const selectedPath = paths[rank];
    const fact = `Correct FS Fotovalidering ${rank === 3 ? "godkendt_id9" : "backup_id8"}: replace old-value-id9 with new-value-id9.`;
    generateObjectMock.mockResolvedValueOnce({ object: { path: selectedPath } })
      .mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: selectedPath,
        old_text: "old-value-id9", new_text: "new-value-id9" } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Exact local correction." } });
    const result = await runMutation(kb, fact, { preflightInput: fact });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [selectedPath] } });
    expect((await kb.readConcept(selectedPath)).body).toContain("new-value-id9");
  });

  it("uses a unique exact old identifier as title-less update evidence", async () => {
    staged();
    const selected = "/projects/authoritative-record.md";
    const competitors = ["/projects/other-record.md", "/projects/archive-record.md"];
    await kb.writeConcept(selected, { type: "Project" }, "The authoritative SHA is old-value-id9.", "seed");
    for (const competitor of competitors) {
      await kb.writeConcept(competitor, { type: "Project", title: "Other Record" }, "Other project details. ".repeat(2_500), "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue([selected, ...competitors].map((conceptPath) => ({
      path: conceptPath, type: "Project", title: conceptPath === selected ? "" : "Other Record",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const instruction = "Correct the authoritative record: replace old-value-id9 with new-value-id9.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: selected } })
      .mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: selected,
        old_text: "old-value-id9", new_text: "new-value-id9" } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Exact claim corrected." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction })).toMatchObject({
      ok: true, result: { filesChanged: [selected] },
    });
  });

  it.each([false, true])("checks all five complete owners for a title-less identifier (shared=%s)", async (shared) => {
    staged();
    const paths = Array.from({ length: 5 }, (_, i) => `/projects/record-${i}.md`);
    for (let i = 0; i < paths.length; i++) {
      const body = i === 3 ? "The authoritative old-value-id9 is recorded here." :
        i === 4 && shared ? "Archive: old-value-id9 was recorded here." :
        `Other project context ${i}. `.repeat(i < 3 ? 1_800 : 5);
      await kb.writeConcept(paths[i], { type: "Project", title: `Record ${i}` }, body, "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue(paths.map((conceptPath, i) => ({
      path: conceptPath, type: "Project", title: `Record ${i}`,
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const instruction = "Correct the authoritative claim: replace old-value-id9 with new-value-id9.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: paths[3] } });
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: paths[3],
      old_text: "old-value-id9", new_text: "new-value-id9" } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Exact correction." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [paths[3]] } });
    expect((await kb.readConcept(paths[3])).body).toContain("new-value-id9");
  });

  it.each(["unique", "shared", "negated"])('grounds an exact add in a unique existing item ID (%s)', async (caseName) => {
    staged();
    const paths = Array.from({ length: 5 }, (_, i) => `/projects/record-${i}.md`);
    for (let i = 0; i < paths.length; i++) {
      const body = i === 3 ? 'Owner baseline mentions godkendt_id9.' :
        i === 4 && caseName === 'shared' ? 'Another baseline mentions godkendt_id9.' :
        `Other validation details ${i}. `.repeat(i < 3 ? 1_700 : 2);
      await kb.writeConcept(paths[i], { type: 'Project', title: `Record ${i}` }, body, 'seed');
    }
    vi.spyOn(kb, 'search').mockResolvedValue(paths.map((conceptPath, i) => ({
      path: conceptPath, type: 'Project', title: `Record ${i}`,
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = caseName === 'negated'
      ? 'FS Fotovalidering records an unrelated observation, not about godkendt_id9.'
      : 'FS Fotovalidering godkendt_id9 measured head height at 79.37 percent.';
    generateObjectMock.mockResolvedValueOnce({ object: { path: paths[3] } });
    if (caseName !== 'negated') generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    if (caseName !== 'negated') {
      expect(result).toMatchObject({ ok: true, result: { filesChanged: [paths[3]] } });
      expect((await kb.readConcept(paths[3])).body).toContain(fact);
    } else {
      expect(result).toMatchObject({ ok: false, status: 'failed' });
      expect((await kb.readConcept(paths[3])).body).not.toContain(fact);
      expect(generateObjectMock).toHaveBeenCalledTimes(1);
    }
  });

  it("does not authorize a title-less owner using an incidental unique date or negated name", async () => {
    staged();
    const selected = "/projects/archive-record.md";
    const other = "/projects/current-record.md";
    await kb.writeConcept(selected, { type: "Project", title: "Archive Record" },
      "Archive dated 2026-10-14 contains old-value-id9.", "seed");
    await kb.writeConcept(other, { type: "Project", title: "Current Record" },
      "Current record also contains old-value-id9.", "seed");
    const filler = "/projects/unrelated-record.md";
    await kb.writeConcept(filler, { type: "Project", title: "Unrelated Record" }, "Unrelated details. ".repeat(5_000), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([selected, other, filler].map((conceptPath) => ({
      path: conceptPath, type: "Project", title: conceptPath === selected ? "Archive Record" : "Current Record",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const instruction = "Correct the dated 2026-10-14 record, not Archive: replace old-value-id9 with new-value-id9.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: selected } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction })).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(selected)).body).toContain("old-value-id9");
    expect((await kb.readConcept(other)).body).toContain("old-value-id9");
  });

  it("skips an explicitly negated owner on metadata abstention", async () => {
    staged();
    const api = "/apis/billing-api.md";
    const playbook = "/playbooks/oncall-billing.md";
    await kb.writeConcept(api, { type: "API Endpoint", title: "Billing API" }, "Billing API context. ".repeat(2_300), "seed");
    await kb.writeConcept(playbook, { type: "Playbook", title: "Billing On-call" }, "On-call context. ".repeat(2_700), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([api, playbook].map((conceptPath) => ({
      path: conceptPath, type: "Concept", title: conceptPath === api ? "Billing API" : "Billing On-call",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Billing responders on call handle incidents, not Billing API.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: "" } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: true, result: { filesChanged: [playbook] },
    });
    expect((await kb.readConcept(api)).body).not.toContain(fact);
    expect((await kb.readConcept(playbook)).body).toContain(fact);
  });

  it("rejects a generic owner when a specific competing topic is named", async () => {
    staged();
    const general = "/projects/fs-fotovalidering-general.md";
    const specific = "/projects/fs-fotovalidering-tono-onot.md";
    await kb.writeConcept(general, { type: "Project", title: "FS Fotovalidering General" }, "General context. ".repeat(3_000), "seed");
    await kb.writeConcept(specific, { type: "Project", title: "FS Fotovalidering Tono Onot" }, "Tono Onot context. ".repeat(3_000), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([general, specific].map((conceptPath) => ({
      path: conceptPath, type: "Project", title: conceptPath === general ? "FS Fotovalidering General" : "FS Fotovalidering Tono Onot",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "FS Fotovalidering experiment 69 in Tono Onot measured head height.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: general } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(general)).body).not.toContain(fact);
    expect((await kb.readConcept(specific)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("accepts one of two equally plausible metadata owners", async () => {
    staged();
    const firstOwner = "/projects/fs-fotovalidering-tests.md";
    const secondOwner = "/projects/fs-fotovalidering-review.md";
    await kb.writeConcept(firstOwner, { type: "Project", title: "FS Fotovalidering Tests" }, "Project context. ".repeat(2_700), "seed");
    await kb.writeConcept(secondOwner, { type: "Project", title: "FS Fotovalidering Review" }, "Review context. ".repeat(2_700), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([firstOwner, secondOwner].map((conceptPath) => ({
      path: conceptPath, type: "Project",
      title: conceptPath === firstOwner ? "FS Fotovalidering Tests" : "FS Fotovalidering Review",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "FS Fotovalidering experiment 69 records a project observation.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: firstOwner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [firstOwner] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
    expect((await kb.readConcept(firstOwner)).body).toContain(fact);
    expect((await kb.readConcept(secondOwner)).body).not.toContain(fact);
  });

  it("rejects every token of a negated multiword owner mention despite verifier approval", async () => {
    staged();
    const wrongOwner = "/projects/fs-fotovalidering-tono-onot.md";
    const rightOwner = "/projects/fs-fotovalidering-general.md";
    await kb.writeConcept(wrongOwner, { type: "Project", title: "FS Fotovalidering Tono Onot" }, "Project context. ".repeat(2_700), "seed");
    await kb.writeConcept(rightOwner, { type: "Project", title: "FS Fotovalidering General" }, "General project context. ".repeat(2_700), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([wrongOwner, rightOwner].map((conceptPath) => ({
      path: conceptPath, type: "Project",
      title: conceptPath === wrongOwner ? "FS Fotovalidering Tono Onot" : "FS Fotovalidering General",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "FS Fotovalidering experiment 69 records an observation, not about Tono Onot.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: wrongOwner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(wrongOwner)).body).not.toContain(fact);
    expect((await kb.readConcept(rightOwner)).body).not.toContain(fact);
  });

  it("includes all five retained metadata hits when checking selected-owner discrimination", async () => {
    staged();
    const paths = [
      "/projects/tono-onot.md",
      "/projects/project-beta.md",
      "/projects/project-gamma.md",
      "/projects/project-onot-archive.md",
      "/projects/project-epsilon.md",
    ];
    const titles = ["Tono Onot", "Project Beta", "Project Gamma", "Project Onot Archive", "Project Epsilon"];
    for (let i = 0; i < paths.length; i++) {
      await kb.writeConcept(paths[i], { type: "Project", title: titles[i] }, `Project ${i} context. `.repeat(2_700), "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue(paths.map((conceptPath, i) => ({
      path: conceptPath, type: "Project", title: titles[i],
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Onot archive experiment 69 records a camera calibration observation for this project.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: paths[0] } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    for (const conceptPath of paths) expect((await kb.readConcept(conceptPath)).body).not.toContain(fact);
  });

  it("allows a positively named domain-specific owner through the independent check", async () => {
    staged();
    const wrongOwner = "/projects/fs-fotovalidering-general.md";
    const rightOwner = "/projects/fs-fotovalidering-tono-onot.md";
    await kb.writeConcept(wrongOwner, { type: "Project", title: "FS Fotovalidering General" }, "General project context. ".repeat(2_000), "seed");
    await kb.writeConcept(rightOwner, { type: "Project", title: "FS Fotovalidering Tono Onot" }, "Tono Onot project context. ".repeat(2_000), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([wrongOwner, rightOwner].map((conceptPath) => ({
      path: conceptPath, type: "Project",
      title: conceptPath === wrongOwner ? "FS Fotovalidering General" : "FS Fotovalidering Tono Onot",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Tono Onot experiment 69 records a camera calibration observation for this project.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: rightOwner } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [rightOwner] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
    expect((await kb.readConcept(rightOwner)).body).toContain(fact);
    expect((await kb.readConcept(wrongOwner)).body).not.toContain(fact);
  });

  it("does not use description-only incidental mentions as metadata owner anchors", async () => {
    staged();
    const first = "/projects/project-alpha.md";
    const second = "/projects/project-beta.md";
    await kb.writeConcept(first, { type: "Project", title: "Project Alpha", description: "Status overview for the camera calibration project." }, "Alpha context. ".repeat(3_500), "seed");
    await kb.writeConcept(second, { type: "Project", title: "Project Beta", description: "Beta project context." }, "Beta context. ".repeat(3_500), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([
      { path: first, type: "Project", title: "Project Alpha", description: "Status overview for the camera calibration project.", score: 10, confidence: 10, confidenceQualified: false },
      { path: second, type: "Project", title: "Project Beta", description: "Beta project context.", score: 10, confidence: 10, confidenceQualified: false },
    ]);
    const fact = "The camera calibration status overview records the lens alignment result.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: first } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: false, status: "failed", error: expect.stringContaining("metadata-selected owner lacks a positive discriminative anchor"),
    });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept(first)).body).not.toContain(fact);
  });

  it("defers an oversized selected owner without sending a truncated body for verification", async () => {
    staged();
    const api = "/apis/billing-api.md";
    const playbook = "/playbooks/oncall-billing.md";
    await kb.writeConcept(api, { type: "API Endpoint", title: "Billing API" }, "Billing API details. ".repeat(3_500), "seed");
    await kb.writeConcept(playbook, { type: "Playbook", title: "Billing On-call" }, "On-call context. ".repeat(1_800), "seed");
    vi.spyOn(kb, "search").mockResolvedValue([api, playbook].map((conceptPath) => ({
      path: conceptPath, type: "Concept", title: conceptPath === api ? "Billing API" : "Billing On-call",
      score: 10, confidence: 10, confidenceQualified: false,
    })));
    const fact = "Billing logs request IDs for charges created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: api } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: false, status: "failed", error: expect.stringContaining("owner body exceeds evidence limit"),
    });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept(api)).body).not.toContain(fact);
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

  it("prioritizes a distinctive late identifier in the bounded mutation search", async () => {
    await billingFixture(); staged();
    const search = vi.spyOn(kb, "search");
    const fact = "FS Fotovalidering godkendt_id9 head-height detail is approved for review.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    await runMutation(kb, `Remember the detailed project status and operational follow-up: ${fact}`, {
      ownerHint: "/apis/billing-api.md", preflightInput: fact, directAdd: true,
    });
    expect(search.mock.calls[0][0]).toContain("godkendt_id9");
  });

  it("keeps a late named owner in the bounded search when the new identifier is absent", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering" }, "Photo validation baseline.", "seed");
    const search = vi.spyOn(kb, "search");
    const fact = "Remember the detailed operational project status and evaluation follow-up: FS Fotovalidering godkendt_id9 measured 79.37 percent head height.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "defer", reason: "Test only." } });
    await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(search.mock.calls[0][0]).toContain("fotovalidering");
    expect(search.mock.results[0].value).toBeDefined();
    expect((await kb.search(search.mock.calls[0][0])).some((hit) => hit.path === owner)).toBe(true);
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

  it("applies a caller-quoted correction with independent verification and compare-and-swap", async () => {
    await billingFixture(); staged();
    const instruction = "Correct `/apis/billing-api.md`: replace the exact phrase `ad-hoc charges come from support tooling.` with `ad-hoc charges come from support tooling only after approval.` Preserve monthly charges.";
    const patchSpy = vi.spyOn(kb, "patchConcept");
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(oldBody);
      expect(request.prompt).toContain("ALTERNATIVE SEARCH HITS");
      expect(request.prompt).toContain("OLD TEXT:");
      expect(request.prompt).toContain("NEW TEXT:");
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect(patchSpy).toHaveBeenCalledWith(
      "/apis/billing-api.md",
      { replaceBody: expect.stringContaining("ad-hoc charges come from support tooling only after approval.") },
      "Staged mutation: exact quoted correction",
      expect.stringMatching(/^[a-f0-9]{64}$/),
      undefined,
      expect.any(Function),
    );
    const body = (await kb.readConcept("/apis/billing-api.md")).body;
    expect(body).toContain("Monthly charges are scheduled");
    expect(body).toContain("only after approval");
  });

  it("accepts a quoted correction that changes sentence punctuation locally", async () => {
    await billingFixture(); staged();
    const instruction = "Correct `/apis/billing-api.md`: replace the exact phrase `ad-hoc charges come from support tooling.` with `ad-hoc charges come from support tooling. Approval is required.` Preserve monthly charges.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain("support tooling. Approval is required.");
  });

  it("rejects a quoted correction that edits separated clauses", async () => {
    await billingFixture(); staged();
    const instruction = "Correct `/apis/billing-api.md`: replace the exact phrase `Monthly charges are scheduled; ad-hoc charges come from support tooling.` with `Monthly charges are approved; ad-hoc charges require approval.`";
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
    expect((await kb.readConcept("/apis/billing-api.md")).body).toBe(`${oldBody}\n`);
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
  it("accepts a generated replacement that changes sentence punctuation locally", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: "/apis/billing-api.md",
      old_text: "ad-hoc charges come from support tooling.",
      new_text: "support tooling creates ad-hoc charges. Approval is required.",
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "Correction supported." } });
    const result = await runMutation(kb, "Update the Billing API: support tooling creates ad-hoc charges. Approval is required.");
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain("ad-hoc charges. Approval is required.");
  });

  it("rejects generated replacements that edit separated clauses", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: "/apis/billing-api.md",
      old_text: oldBody,
      new_text: "Monthly charges are approved; ad-hoc charges require approval.",
    } });
    const result = await runMutation(kb, "Update the Billing API: monthly charges and ad-hoc charges require approval.");
    expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("one supported local change") });
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
    expect((await kb.readConcept("/apis/billing-api.md")).body).toBe(`${oldBody}\n`);
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
