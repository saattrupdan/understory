import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const generateTextMock = vi.hoisted(() => vi.fn());
const generateObjectMock = vi.hoisted(() => vi.fn());
const streamObjectMock = vi.hoisted(() => vi.fn());
vi.mock("ai", async () => ({
  ...(await vi.importActual<typeof import("ai")>("ai")),
  generateText: generateTextMock,
  generateObject: generateObjectMock,
  streamObject: streamObjectMock,
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

import { NoObjectGeneratedError } from "ai";
import { runMutation } from "../src/agent/agent.js";
import { createModel } from "../src/providers/index.js";
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
  vi.stubEnv("MUTATION_STRUCTURED_STREAM", "false");
});

afterEach(async () => {
  await fs.rm(root, { recursive: true, force: true });
  generateTextMock.mockReset();
  generateObjectMock.mockReset();
  streamObjectMock.mockReset();
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

  const claesSeedlessOwner = "/projects/claes-jev-diagnostics.md";
  const claesSeedlessInstruction = "Update Claes design/preference: Dan explicitly does not want reproducibility seeds set anywhere in remaining repository code for this dataset. He chose to REMOVE all archived pilot workflows (scripts/modules/tests and runnable workflow docs), but preserve historical data artifacts. On feat/laya-taxonomy-sampler at 942642e, remaining task-first Laya sampler/collector and benchmark have no --seed parameters, derived seeds, or seed fields in new manifests; local draws use fresh system entropy. Resume uses source-bound manifest + append-only checkpoint for completed row identities, tasks validated against pool and persisted informal row-index schedule; a seed-bearing legacy manifest is rejected rather than migrated. Generic Jev question-answering and task-first Luna generation/review remain. Historic docs/samples/seed-* data artifacts and old ignored data/ outputs were preserved. make check and 81 tests pass; independent reviewer LGTM. Seedless throughput should be benchmarked anew when needed; previous 32-worker ~5.22 assignments/s measurement was from seeded implementation, though Laya transport/concurrency otherwise unchanged.";
  const claesSeedlessHistory = "Claes Laya throughput knowledge: the task-first pipeline update uses a seed-spread Informal pilot; historical seed20261013 benchmark files measured 5.223 assignments/s with 32 workers. Pilot observations and data artifacts remain.";

  it("supersedes Claes seed guidance without erasing historical artifacts and verifies a repeat", async () => {
    staged();
    await kb.writeConcept(claesSeedlessOwner, { type: "Project", title: "Claes Jev Diagnostics" },
      `${claesSeedlessHistory}\n\nUnrelated Jev answer history remains intact.`, "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(claesSeedlessHistory);
      expect(prompt).toContain("942642e");
      expect(prompt).toContain("supersedes earlier seeded workflow and pilot guidance");
      return { object: { safe: true, reason: "Later status is complete and historical measurements retained." } };
    });
    expect(await runMutation(kb, claesSeedlessInstruction, { preflightInput: claesSeedlessInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [claesSeedlessOwner] } });
    const body = (await kb.readConcept(claesSeedlessOwner)).body;
    expect(body).toContain(`Historical Claes workflow snapshot (superseded by seedless 942642e; earlier pilot and seeded benchmark advice is not current guidance): ${claesSeedlessHistory}`);
    expect(body).not.toMatch(/^Claes Laya throughput knowledge:/m);
    expect(body).toContain("Unrelated Jev answer history remains intact.");
    expect(body).toContain("REMOVE all archived pilot workflows");
    expect(body).toContain("seed-bearing legacy manifest is rejected rather than migrated");
    expect(body).toContain("previous 32-worker ~5.22 assignments/s measurement was from seeded implementation");
    expect(await runMutation(kb, claesSeedlessInstruction, { preflightInput: claesSeedlessInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(claesSeedlessOwner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("defers a Claes seedless update when a competing owner already records the same release", async () => {
    staged();
    await kb.writeConcept(claesSeedlessOwner, { type: "Project", title: "Claes Jev Diagnostics" }, claesSeedlessHistory, "seed");
    await kb.writeConcept("/projects/claes/workflow-copy.md", { type: "Project", title: "Claes Workflow" },
      "At 942642e the seedless workflow rejected legacy manifests.", "seed");
    expect(await runMutation(kb, claesSeedlessInstruction, { preflightInput: claesSeedlessInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing owner") });
    expect((await kb.readConcept(claesSeedlessOwner)).body).not.toContain("REMOVE all archived pilot workflows");
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("defers Claes seedless status when the independent owner check declines", async () => {
    staged();
    await kb.writeConcept(claesSeedlessOwner, { type: "Project", title: "Claes Jev Diagnostics" }, claesSeedlessHistory, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false, reason: "Unresolved contradiction" } });
    expect(await runMutation(kb, claesSeedlessInstruction, { preflightInput: claesSeedlessInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent Claes seedless") });
    expect((await kb.readConcept(claesSeedlessOwner)).body).toContain(claesSeedlessHistory);
  });

  const fs577Owner = "/projects/fs-fotovalidering-continuation-faa257ccc4eab983.md";
  const fs577Old = "FS Fotovalidering baseline 2026-10-14: The source PPTX/XLSX were reorganized on 14 Oct 2026: 23 previously Godkendt-marked photos moved into matching Motiv/Billedekvalitet sections, Godkendt now has exactly three rows and eleven approved photo columns. Baseline slide media hashes were preserved and parser.extract_images found 258 unique photo IDs. The user's specified scope was PPTX and source XLSX only; parsed/ remains stale, 2026-10-14/regenerate_parsed.py was deleted. Important gotcha: src/scripts/rebuild_test_baseline.py and baseline-local regenerate_parsed.py still assume the old 34-column/31-row Godkendt layout, so they cannot regenerate the reorganized baseline without adapting the builder. Do not interpret parsed/ as up-to-date with the new Office sources.";
  const fs577Instruction = "Correct FS Fotovalidering baseline 2026-10-14 status: the repository source-only builder src/scripts/rebuild_test_baseline.py was FIXED and committed on main (HEAD 5779826, feature merged 14 Oct 2026) to accept 257 extracted photos and exactly 10 contiguous explicitly x-marked Godkendt IDs 1–10 in a three-row worksheet (no old godkendt_id21 rejection row). Keep 50 workbook-only exclusions and 24 active classifier labels. The 2026-10-14 parsed/ was regenerated from the six PPTX and single XLSX using `uv run python src/scripts/rebuild_test_baseline.py <source-folder> --output-dir <new-output-dir>` and published after private staging validation: 257 unique photo SHA-256s, 10 accept / 247 reject, 24 labels per record, 50 workbook-only exclusions, source file hashes recorded in parsed/source-hashes.json. This supersedes prior memory stating parsed/ is absent/stale or the builder cannot regenerate it. Baseline images were processed automatically without surfacing content to the agent. Targeted rebuild/parser tests 41 passed; make check passed; frontend tests 258 passed. Full make test under Python 3.12 yielded 491 passed, 1 skipped, 3 existing unrelated API/preprocess content-gate failures; Python 3.14 also lacked native _pillow_heif, so use `uv run --python 3.12` when testing in clean worktrees.";

  it("archives the 258/11 reorganization snapshot before adding the dated 5779826 milestone", async () => {
    staged();
    await kb.writeConcept(fs577Owner, { type: "Project Continuation", title: "FS baseline reorganization" },
      `${fs577Old}\n\nOther owner context remains.`, "seed");
    const later = "/projects/fs-fotovalidering-continuation-later.md";
    const laterStatus = "After 5779826, a later reviewer fixed src/scripts/rebuild_test_baseline.py for owner-corrected Godkendt on a 257-photo parsed/ with 50 workbook-only columns at a081a66.";
    await kb.writeConcept(later, { type: "Project Continuation", title: "Later FS correction" }, laterStatus, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`${later}: ${laterStatus}`);
      expect(prompt).toContain("5779826 state as a dated milestone");
      expect(prompt).toContain(fs577Old);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fs577Instruction, { preflightInput: fs577Instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fs577Owner] } });
    const body = (await kb.readConcept(fs577Owner)).body;
    expect(body).toContain(`Historical FS Fotovalidering baseline reorganization snapshot (before ID 30 removal and source-only rebuild; not current guidance): ${fs577Old}`);
    expect(body).toContain("FS Fotovalidering source-only baseline milestone (14 Oct 2026; 5779826; later owner corrections may change marks):");
    expect(body).toContain("parsed/source-hashes.json");
    expect(body).toContain("Other owner context remains.");
    expect(await runMutation(kb, fs577Instruction, { preflightInput: fs577Instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(fs577Owner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("defers 5779826 status when another FS owner records the same commit", async () => {
    staged();
    await kb.writeConcept(fs577Owner, { type: "Project Continuation", title: "FS reorganization" }, fs577Old, "seed");
    const peer = "/projects/fs-fotovalidering-other.md";
    await kb.writeConcept(peer, { type: "Project", title: "FS other" },
      "5779826 fixed src/scripts/rebuild_test_baseline.py for 257 source photos.", "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("5779826 fixed src/scripts/rebuild_test_baseline.py");
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fs577Instruction, { preflightInput: fs577Instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS 5779826") });
    expect((await kb.readConcept(fs577Owner)).body).toBe(`${fs577Old}\n`);
  });

  it("presents bounded context from a long FS 5779826 peer passage", async () => {
    staged();
    await kb.writeConcept(fs577Owner, { type: "Project Continuation", title: "FS reorganization" }, fs577Old, "seed");
    const peer = "/projects/fs-fotovalidering-long.md";
    await kb.writeConcept(peer, { type: "Project Continuation", title: "FS later correction" },
      `Earlier findings: ${"unrelated classification evidence. ".repeat(140)}Later a081a66 fixed src/scripts/rebuild_test_baseline.py for the owner-corrected 257-photo parsed/ baseline after 5779826.${" Unrelated audit history. ".repeat(100)}`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`${peer}: Earlier findings:`);
      expect(prompt).toContain("Later a081a66 fixed src/scripts/rebuild_test_baseline.py");
      expect(prompt).toContain("[... omitted peer context ...]");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fs577Instruction, { preflightInput: fs577Instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fs577Owner] } });
  });

  it("defers 5779826 status when independent review vetoes or the owner changes", async () => {
    staged();
    await kb.writeConcept(fs577Owner, { type: "Project Continuation", title: "FS reorganization" }, fs577Old, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fs577Instruction, { preflightInput: fs577Instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS 5779826") });
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept(fs577Owner, { type: "Project Continuation", title: "FS reorganization" },
        `${fs577Old}\n\nConcurrent owner update.`, "external");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fs577Instruction, { preflightInput: fs577Instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("Concept changed") });
    expect((await kb.readConcept(fs577Owner)).body).toContain("Concurrent owner update.");
  });

  const annotationHandoffOwner = "/projects/fs-fotovalidering-continuation-257a96e360815542.md";
  const annotationHandoffCompletion = "The derived parsed/ contains 258 test photos: motiv3_id6 and motiv8_id12 have only three positives each. The temporary photo-bearing /tmp/fs-baseline-uncertain-labels.html was deleted after use; downloaded fs-baseline-uncertain-labels.json (43 binary decisions: 1 yes, 42 no) was kept. On 14 Oct 2026 all 43 '?' cells were replaced, yes as x and 42 no as blank; validation found zero remaining '?' across the workbook.";
  const annotationHandoffFact = "FS Fotovalidering baseline annotation handoff (14 Oct 2026): Dan does not want question-mark annotations in the Excel workbook; uncertain category labels must be decided by him from a local image-annotation HTML, not silently converted to blanks/negatives. Current 2026-10-14 source workbook has 43 '?' cells, all on two photos, motiv3_id6 and motiv8_id12 in the Motiv sheet. A private single-file HTML was created and opened at file:///tmp/fs-baseline-uncertain-labels.html with both baseline photos embedded for Dan, 43 binary category prompts, keyboard arrows, local autosave, and JSON export fs-baseline-uncertain-labels.json. Agent did not inspect/render photos; only a synthetic UI fixture was tested. Workbook has not been altered pending Dan's JSON decisions; when received, map yes to x and no to blank, preserving existing positives and the other sheets, then delete temporary photo-bearing HTML once finished.";

  it("records the pending annotation handoff as history beside completed decisions", async () => {
    staged();
    await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS Fotovalidering — baseline decisions" },
      `${annotationHandoffCompletion}\n\nIndependent later note.`, "seed");
    await kb.writeConcept("/projects/fs-fotovalidering-tono-onot.md", { type: "Project", title: "FS TONO/ONOT experiment" },
      "TONO/ONOT compares unrelated passport evaluation categories.", "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(annotationHandoffCompletion);
      expect(prompt).toContain("Before Dan's decisions:");
      expect(prompt).toContain("43 '?' cells were replaced (1 yes as x, 42 no as blank)");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [annotationHandoffOwner] } });
    const body = (await kb.readConcept(annotationHandoffOwner)).body;
    expect(body).toContain(annotationHandoffCompletion);
    expect(body).toContain("Independent later note.");
    expect(body).toContain("Historical FS baseline annotation handoff (14 Oct 2026; decisions completed later that day):");
    expect(body).toContain("Workbook has not been altered pending Dan's JSON decisions");
    expect(body).toContain("zero '?' remained in the source workbook, and the photo-bearing HTML was deleted.");
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(annotationHandoffOwner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
    expect((await kb.readConcept("/projects/fs-fotovalidering-tono-onot.md")).body).not.toContain(annotationHandoffFact);
  });

  it("defers an annotation handoff with a competing detailed owner or missing completion", async () => {
    staged();
    await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS baseline decisions" },
      annotationHandoffCompletion, "seed");
    const peer = "/projects/fs-fotovalidering-other.md";
    await kb.writeConcept(peer, { type: "Project", title: "FS other" },
      "fs-baseline-uncertain-labels.html contains 43 prompts for motiv3_id6 and motiv8_id12.", "seed");
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing detailed owner") });
    await kb.writeConcept(peer, { type: "Project", title: "FS other" }, "Unrelated note.", "replace");
    await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS baseline decisions" },
      "The workbook still has 43 question marks. The HTML remains open.", "replace");
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("completion owner lacks") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("checks a paraphrased competing FS handoff even without the HTML filename", async () => {
    staged();
    await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS baseline decisions" },
      annotationHandoffCompletion, "seed");
    const peer = "/projects/fs-fotovalidering-other.md";
    const competing = "Dan used an image annotation UI for 43 binary category decisions on motiv3_id6 and motiv8_id12; the pending workbook was not changed yet.";
    await kb.writeConcept(peer, { type: "Project", title: "FS other" }, competing, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`${peer}: ${competing}`);
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS annotation handoff") });
    expect((await kb.readConcept(annotationHandoffOwner)).body).toBe(`${annotationHandoffCompletion}\n`);
  });

  it("defers a completed owner that also asserts a current pending workbook", async () => {
    staged();
    await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS baseline decisions" },
      `${annotationHandoffCompletion}\n\nThe workbook still has 43 '?' cells awaiting Dan; the HTML remains open.`, "seed");
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("unqualified pending guidance") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("defers a historical annotation handoff on independent veto or concurrent change", async () => {
    staged();
    await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS baseline decisions" },
      annotationHandoffCompletion, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS annotation handoff") });
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept(annotationHandoffOwner, { type: "Project Continuation", title: "FS baseline decisions" },
        `${annotationHandoffCompletion}\n\nA concurrent note.`, "external");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, annotationHandoffFact, { preflightInput: annotationHandoffFact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("Concept changed") });
    expect((await kb.readConcept(annotationHandoffOwner)).body).toContain("A concurrent note.");
  });

  const fsBaselineOwner = "/projects/fs-fotovalidering-continuation-d2fd99e3dfc67efb.md";
  const fsBaselineOld = "FS Fotovalidering 2026-10-14 baseline deletion warning — Do NOT delete everything except the PPTX and source XLSX. The current private baselines/2026-10-14/regenerate_parsed.py is NOT a clean PPTX+XLSX-only parser: it reads parsed/manifest.jsonl, parsed/labels.xlsx, parsed/images/ and parsed/exclusions.json as prior metadata/images, plus generation-provenance.json for six PPTX SHA pins. The repository parse_test_set.py cannot currently regenerate the corrected 258-photo baseline from those PPTX and XLSX alone. Preserve review exports.";
  const fsBaselineInstruction = "Update the FS Fotovalidering 2026-10-14 baseline deletion/regeneration warning: it was correct for the old generator but is now superseded. A durable project script at src/scripts/rebuild_test_baseline.py regenerates the complete 258-photo current test dataset using ONLY six PPTX and the authoritative XLSX as data inputs; it never reads parsed/, generation-provenance.json or owner-24-review. The baseline-local regenerate_parsed.py was replaced by a wrapper invoking uv run --project <repo> on that script, tested successfully from the baseline directory outside the repo. A clean /tmp input directory containing exactly those seven source files produced all 258 image hashes, 24-category labels (including unknowns), roles, paths, both sheets of labels.xlsx and exclusions matching current published parsed/ exactly; only godkendt_id20 byte_size JSON type is normalised from string to int (numeric value unchanged), so regenerated manifest SHA-256 is ea58999a498cffdc9d311600a3e168991c3e3f743d7ab2cee82e2e75869d77c5 versus currently published 3d27505b085a9506a11a87855fa782031500248ea288d48700da44619976ccb6. Private baseline generation-provenance.json records this check; prior parsed/ was deliberately left intact so the active review page/autosave hash remains unchanged. 33 parser/synthetic tests passed; source-only /tmp copies of private photos and decks were removed after validation. It is now safe to delete/rebuild parsed/ from only the PPTX+XLSX via the repository script, but owner-24-review historical exports, old workbook copies, and audit provenance cannot be recreated from those sources and should be retained if the user wants audit history.";

  it("supersedes only the old FS baseline deletion warning and preserves the audit caveat", async () => {
    staged();
    await kb.writeConcept(fsBaselineOwner, { type: "Project", title: "FS Phase 1 continuation" },
      `Earlier independent review finding.\n\n${fsBaselineOld}\n\nLater independent note.`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fsBaselineOld);
      expect(prompt).toContain("both sheets of labels.xlsx");
      expect(prompt).toContain("owner-24-review historical exports");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsBaselineOwner] } });
    const body = (await kb.readConcept(fsBaselineOwner)).body;
    expect(body).toContain(`Historical FS Fotovalidering 2026-10-14 baseline deletion warning (superseded by source-only rebuild; old generator only): ${fsBaselineOld}`);
    expect(body).not.toMatch(/^FS Fotovalidering 2026-10-14 baseline deletion warning/m);
    expect(body).toContain("Earlier independent review finding.\n\n");
    expect(body).toContain("\n\nLater independent note.");
    expect(body).toContain("ea58999a498cffdc9d311600a3e168991c3e3f743d7ab2cee82e2e75869d77c5 versus currently published 3d27505b085a9506a11a87855fa782031500248ea288d48700da44619976ccb6");
    expect(body).toContain("audit provenance cannot be recreated");
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(fsBaselineOwner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("defers a source-only FS baseline update if the old warning or rebuild is duplicated", async () => {
    staged();
    await kb.writeConcept(fsBaselineOwner, { type: "Project", title: "FS Phase 1 continuation" }, fsBaselineOld, "seed");
    const peer = "/projects/fs-fotovalidering/other-baseline.md";
    await kb.writeConcept(peer, { type: "Project", title: "FS other baseline" }, fsBaselineOld, "seed");
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("duplicated") });
    await kb.writeConcept(peer, { type: "Project", title: "FS other baseline" },
      "rebuild_test_baseline.py regenerated ea58999a498cffdc9d311600a3e168991c3e3f743d7ab2cee82e2e75869d77c5.", "replace");
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing owner") });
    expect((await kb.readConcept(fsBaselineOwner)).body).toBe(`${fsBaselineOld}\n`);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("checks an older source-only script warning from a different FS owner before accepting", async () => {
    staged();
    await kb.writeConcept(fsBaselineOwner, { type: "Project", title: "FS Phase 1 continuation" }, fsBaselineOld, "seed");
    const peer = "/projects/fs-fotovalidering-continuation-faa257ccc4eab983.md";
    const peerWarning = "Important gotcha: src/scripts/rebuild_test_baseline.py and baseline-local regenerate_parsed.py still assume the old Godkendt layout and cannot regenerate the reorganized parsed/ baseline.";
    await kb.writeConcept(peer, { type: "Project", title: "FS earlier script status" }, peerWarning, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`${peer}: ${peerWarning}`);
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS baseline") });
    expect((await kb.readConcept(fsBaselineOwner)).body).toBe(`${fsBaselineOld}\n`);
  });

  it("presents bounded script context from a long FS peer paragraph", async () => {
    staged();
    await kb.writeConcept(fsBaselineOwner, { type: "Project", title: "FS Phase 1 continuation" }, fsBaselineOld, "seed");
    const peer = "/projects/fs-fotovalidering-continuation-long.md";
    await kb.writeConcept(peer, { type: "Project", title: "FS long continuation" },
      `Earlier details: ${"other private review observations. ".repeat(110)}The old src/scripts/rebuild_test_baseline.py cannot regenerate the reorganized parsed/ baseline until the Godkendt fix.${" Unrelated audit data. ".repeat(75)}`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`${peer}: Earlier details:`);
      expect(prompt).toContain("old src/scripts/rebuild_test_baseline.py cannot regenerate");
      expect(prompt).toContain("[... omitted peer context ...]");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsBaselineOwner] } });
  });

  it("defers an FS baseline update when the independent check declines or the owner changes", async () => {
    staged();
    await kb.writeConcept(fsBaselineOwner, { type: "Project", title: "FS Phase 1 continuation" }, fsBaselineOld, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS baseline") });
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept(fsBaselineOwner, { type: "Project", title: "FS Phase 1 continuation" },
        `${fsBaselineOld}\n\nNew concurrent finding.`, "external");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fsBaselineInstruction, { preflightInput: fsBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("Concept changed") });
    expect((await kb.readConcept(fsBaselineOwner)).body).toContain("New concurrent finding.");
  });

  const yuNetOwner = "/projects/fs-fotovalidering/auto-crop-investigation.md";
  const yuNetOld = "FS Fotovalidering YuNet face gate (feat/yunet-face-gate, Oct 2026): /preprocess screens NSFW. /validate checks 0.90. The old cache uses rendered crop JPEG digest.";
  const yuNetInstruction = "Update FS Fotovalidering YuNet face gate architecture (feat/yunet-face-gate, Oct 2026): Upload selection calls /content-screen at 0.90. Cache by original exact bytes and model revision/SHA. Camera /preprocess must NOT fall back to local crop. Tests: 2 backend passed, 3 frontend passed.";

  it("supersedes only the uniquely owned YuNet baseline and verifies a repeat as a no-op", async () => {
    staged();
    const unrelated = "Keep this independent project observation unchanged.";
    await kb.writeConcept(yuNetOwner, { type: "Project", title: "Crop research" },
      `${unrelated}\n\n${yuNetOld}\n\nAnother independent observation.`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(yuNetOld);
      expect(prompt).toContain(yuNetInstruction);
      expect(prompt.length).toBeLessThan(3_000);
      expect(prompt).not.toContain(unrelated);
      return { object: { safe: true }, finishReason: "stop" };
    });
    const first = await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction });
    expect(first).toMatchObject({ ok: true, result: { filesChanged: [yuNetOwner] } });
    const body = (await kb.readConcept(yuNetOwner)).body;
    expect(body).toContain(unrelated);
    expect(body).toContain("Historical YuNet face gate baseline (superseded 13 Oct 2026; not current guidance):");
    expect(body).toContain(yuNetOld);
    expect(body).toContain("original exact bytes and model revision/SHA");
    expect(await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("defers YuNet corrections with duplicate owners, missing anchors, or an existing partial update", async () => {
    staged();
    await kb.writeConcept(yuNetOwner, { type: "Project", title: "Crop research" }, yuNetOld, "seed");
    const peer = "/projects/fs-fotovalidering-continuation-aaaabbbb.md";
    await kb.writeConcept(peer, { type: "Project", title: "Other research" }, yuNetOld, "seed");
    expect(await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    await kb.writeConcept(peer, { type: "Project", title: "Other research" }, "Unrelated facts.", "replace");
    await kb.writeConcept(yuNetOwner, { type: "Project", title: "Crop research" },
      yuNetOld.replace("rendered crop JPEG digest", "some other key"), "replace");
    expect(await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    await kb.writeConcept(yuNetOwner, { type: "Project", title: "Crop research" },
      `${yuNetOld}\n\nFS Fotovalidering YuNet face gate architecture (feat/yunet-face-gate, Oct 2026): ${yuNetInstruction.slice(yuNetInstruction.indexOf(": ") + 2)}`, "replace");
    expect(await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("vetoes a YuNet correction and rejects a concurrent owner change", async () => {
    staged();
    await kb.writeConcept(yuNetOwner, { type: "Project", title: "Crop research" }, yuNetOld, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false }, finishReason: "stop" });
    expect(await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    generateObjectMock.mockImplementationOnce(async () => {
      await kb.writeConcept(yuNetOwner, { type: "Project", title: "Crop research" },
        `${yuNetOld}\n\nA concurrent observation.`, "external");
      return { object: { safe: true }, finishReason: "stop" };
    });
    expect(await runMutation(kb, yuNetInstruction, { preflightInput: yuNetInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(yuNetOwner)).body).toContain("A concurrent observation.");
  });

  it("disables reasoning for independent staged checks even when deep mutations enable it", async () => {
    staged();
    vi.stubEnv("MUTATION_ENABLE_THINKING", "true");
    vi.mocked(createModel).mockClear();
    const owner = "/projects/owner.md";
    const paragraph = "This complete owner paragraph has a distinctive beginning and a distinctive ending.";
    await kb.writeConcept(owner, { type: "Project", title: "Owner" }, `Before.\n\n${paragraph}\n\nAfter.`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb,
      `In \`${owner}\`, remove ONLY the exact paragraph beginning \`This complete owner paragraph\` and ending \`a distinctive ending.\`.`);
    expect(result.ok).toBe(true);
    expect(generateObjectMock).toHaveBeenCalledOnce();
    expect(vi.mocked(createModel).mock.calls[0]?.[0].extraBody).toMatchObject({
      chat_template_kwargs: { enable_thinking: false },
    });
  });

  it("repairs only a current local sweep status in an explicitly named server", async () => {
    staged();
    const owner = "/servers/sparkie.md";
    const stale = "Claes Laya high-concurrency throughput sweep is still deferred.";
    const updated = "Claes Laya high-concurrency throughput sweep completed on 13 Oct 2026 after Sparkie's competing jobbot became idle.";
    const hostDetails = "Sparkie's competing jobbot previously used the GPU.";
    const instruction = `In \`${owner}\`, correct ONLY the obsolete current claim that the Claes Laya high-concurrency throughput sweep is still deferred/not run: the live sweep was completed on 13 Oct 2026 after Sparkie's competing jobbot became idle. Preserve the earlier deferral as dated historical planning if it appears, and preserve every unrelated Sparkie/Qwen/GPU/server assertion verbatim. Do not touch any other concept; if the server concept contains no such current claim, make no write.`;
    await kb.writeConcept(owner, { type: "Server", title: "Sparkie" },
      `Qwen3.8 mmap performance remains independent.\n\n${stale} ${hostDetails}\n\nThe earlier scheduling plan from 2 Oct is historical.`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: stale, new_text: updated } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(updated);
    expect(body).not.toContain(stale);
    expect(body).toContain(hostDetails);
    expect(body).toContain("Qwen3.8 mmap performance remains independent.");
    expect(body).toContain("The earlier scheduling plan from 2 Oct is historical.");
  });

  it("narrows a broad local status proposal to one exact sentence", async () => {
    staged();
    const owner = "/servers/sparkie.md";
    const stale = "The Claes Laya high-concurrency throughput sweep is still deferred until the competing jobbot becomes idle.";
    const corrected = "The Claes Laya high-concurrency throughput sweep completed on 13 Oct 2026 after the competing jobbot became idle.";
    const history = "The Qwen3.8 GPU host uses its own independent mmap roadmap. ".repeat(12);
    const paragraph = `${history}${stale} The Sparkie router remains on port 31000.`;
    const instruction = `In \`${owner}\`, correct ONLY the obsolete current claim that the Claes Laya high-concurrency throughput sweep is still deferred/not run: the live sweep was completed on 13 Oct 2026 after Sparkie's competing jobbot became idle. Preserve unrelated server observations.`;
    await kb.writeConcept(owner, { type: "Server", title: "Sparkie" }, paragraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: paragraph, new_text: corrected } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain("Select the index of exactly ONE CURRENT stale status sentence");
        expect(prompt).toContain(stale);
        return { object: { index: 12, new_text: corrected } };
      }).mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(`${history}${corrected}`);
    expect(body).toContain("The Sparkie router remains on port 31000.");
  });

  it("narrows a whole-server proposal across paragraphs", async () => {
    staged();
    const owner = "/servers/sparkie.md";
    const intro = "Sparkie hosts Qwen3.8 mmap and an unrelated GPU service. ".repeat(12).trim();
    const oldClaim = "The Claes Laya high-concurrency throughput sweep is still pending until the competing jobbot becomes idle.";
    const updated = "The Claes Laya high-concurrency throughput sweep completed on 13 Oct 2026 after the competing jobbot became idle.";
    const body = `${intro}\n\n${oldClaim} Other Sparkie routing remains on port 31000.`;
    const instruction = `In \`${owner}\`, correct ONLY the obsolete current claim that the Claes Laya high-concurrency throughput sweep is still deferred/not run: the live sweep was completed on 13 Oct 2026 after Sparkie's competing jobbot became idle. Preserve unrelated routing facts.`;
    await kb.writeConcept(owner, { type: "Server", title: "Sparkie" }, body, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: body, new_text: updated } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(oldClaim);
        return { object: { index: 12, new_text: updated } };
      }).mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(`${intro}\n\n${updated} Other Sparkie routing`);
  });

  it("rejects a local server clause with no named throughput topic", async () => {
    staged();
    const owner = "/servers/sparkie.md";
    const unrelated = "Sparkie Qwen GPU performance remains unchanged since the last checkpoint.";
    const instruction = `In \`${owner}\`, correct ONLY the obsolete current claim that the Claes Laya high-concurrency throughput sweep is still deferred/not run: the live sweep was completed on 13 Oct 2026. Preserve unrelated server facts.`;
    await kb.writeConcept(owner, { type: "Server", title: "Sparkie" },
      `Claes Laya throughput runs on the same host. ${unrelated}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: unrelated,
      new_text: "Sparkie Qwen GPU performance completed on 13 Oct 2026." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain(unrelated);
  });

  it("refuses to rewrite a historical local sweep plan as a current status", async () => {
    staged();
    const owner = "/servers/sparkie.md";
    const historical = "Claes Laya high-concurrency throughput sweep was deferred on 2 Oct 2026 before the router became idle.";
    const instruction = `In \`${owner}\`, correct ONLY the obsolete current claim that the Claes Laya high-concurrency throughput sweep is still deferred/not run: the live sweep was completed on 13 Oct 2026. Preserve historical planning.`;
    await kb.writeConcept(owner, { type: "Server", title: "Sparkie" }, historical, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: historical,
      new_text: "Claes Laya high-concurrency throughput sweep completed on 13 Oct 2026." } })
      .mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain(historical);
  });

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

  it("rejects a capped staged safety decision even when its JSON says safe", async () => {
    staged();
    const owner = "/projects/owner.md";
    const paragraph = "This complete owner paragraph has a distinctive beginning and a distinctive ending.";
    await kb.writeConcept(owner, { type: "Project", title: "Owner" },
      `Before.\n\n${paragraph}\n\nAfter.`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true }, finishReason: "length" });
    const result = await runMutation(kb,
      `In \`${owner}\`, remove ONLY the exact paragraph beginning \`This complete owner paragraph\` and ending \`a distinctive ending.\`.`);
    expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("output cap") });
    expect((await kb.readConcept(owner)).body).toContain(paragraph);
  });

  it("uses only the final validated streamed safety object for local staged mutations", async () => {
    staged();
    vi.stubEnv("MUTATION_STRUCTURED_STREAM", "true");
    const owner = "/projects/owner.md";
    const paragraph = "This complete owner paragraph has a distinctive beginning and a distinctive ending.";
    await kb.writeConcept(owner, { type: "Project", title: "Owner" },
      `Before.\n\n${paragraph}\n\nAfter.`, "seed");
    let complete!: (value: { safe: boolean }) => void;
    const object = new Promise<{ safe: boolean }>((resolve) => { complete = resolve; });
    streamObjectMock.mockReturnValueOnce({
      object,
      finishReason: Promise.resolve("stop"),
      fullStream: (async function* () { complete({ safe: true }); yield { type: "finish" }; })(),
    });
    const instruction = `In \`${owner}\`, remove ONLY the exact paragraph beginning \`This complete owner paragraph\` and ending \`a distinctive ending.\`.`;
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect(streamObjectMock).toHaveBeenCalledOnce();
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("rejects a capped streamed decision before applying a mutation", async () => {
    staged();
    vi.stubEnv("MUTATION_STRUCTURED_STREAM", "true");
    const owner = "/projects/owner.md";
    const paragraph = "This complete owner paragraph has a distinctive beginning and a distinctive ending.";
    await kb.writeConcept(owner, { type: "Project", title: "Owner" },
      `Before.\n\n${paragraph}\n\nAfter.`, "seed");
    streamObjectMock.mockReturnValueOnce({ object: Promise.resolve({ safe: true }), finishReason: Promise.resolve("length"),
      fullStream: (async function* () { yield { type: "finish" }; })() });
    const instruction = `In \`${owner}\`, remove ONLY the exact paragraph beginning \`This complete owner paragraph\` and ending \`a distinctive ending.\`.`;
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("output cap") });
    expect((await kb.readConcept(owner)).body).toContain(paragraph);
  });

  it("removes only one uniquely endpoint-anchored paragraph after checking its middle", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    const paragraph = "Claes Jev diagnostics decision (Oct 2026): keep strict _valid_answer. The Claes-only validation report passed. orchestrator fixed and verified 110 tests.";
    const instruction = `In \`${owner}\`, remove ONLY the exact paragraph beginning \`Claes Jev diagnostics decision (Oct 2026): keep strict _valid_answer\` and ending \`orchestrator fixed and verified 110 tests.\` This paragraph describes the unrelated Claes Jev project. Do not change any other material, and do not add it anywhere else.`;
    await kb.writeConcept(owner, { type: "Project", title: "Synthetic Dataset Generation" },
      `FS generation facts.\n\n${paragraph}\n\nRetain FS review results.`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`PARAGRAPH TO REMOVE:\n${paragraph}`);
      expect(prompt).toContain(`OWNER ${owner}:`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toBe("FS generation facts.\n\nRetain FS review results.\n");
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("treats an independently verified prior retraction as a no-op without changing the owner", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    const instruction = `In \`${owner}\`, remove ONLY the exact paragraph beginning \`Claes Jev diagnostics decision (Oct 2026): keep strict _valid_answer\` and ending \`orchestrator fixed and verified 110 tests.\` Do not change any other material.`;
    await kb.writeConcept(owner, { type: "Project", title: "Synthetic Dataset Generation" },
      "FS generation facts.\n\nRetain FS review results.", "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(owner)).body).toBe("FS generation facts.\n\nRetain FS review results.\n");
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("does not claim absence if the paragraph start remains with Markdown ticks", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    const instruction = `In \`${owner}\`, remove ONLY the exact paragraph beginning \`Claes Jev diagnostics decision (Oct 2026): keep strict _valid_answer\` and ending \`orchestrator fixed and verified 110 tests.\` Do not change any other material.`;
    await kb.writeConcept(owner, { type: "Project", title: "Synthetic Dataset Generation" },
      "Claes Jev diagnostics decision (Oct 2026): keep strict `_valid_answer` for checkpoint integrity. Unfinished claim.", "seed");
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("rejects ambiguous, incomplete, mismatched, and stale endpoint-anchored removals", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering/synthetic-dataset-generation.md";
    const paragraph = "Claes Jev diagnostics decision (Oct 2026): keep strict _valid_answer. Claes-only content. orchestrator fixed and verified 110 tests.";
    const instruction = `In \`${owner}\`, remove ONLY the exact paragraph beginning \`Claes Jev diagnostics decision (Oct 2026): keep strict _valid_answer\` and ending \`orchestrator fixed and verified 110 tests.\` Do not change any other material.`;
    const seed = async (body: string) => kb.writeConcept(owner, { type: "Project", title: "Synthetic Dataset Generation" }, body, "seed");
    await seed(`Before.\n\n${paragraph}\n\n${paragraph}\n\nAfter.`);
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    await seed(`Before.\n\n${paragraph} Added material.\n\nAfter.`);
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    await seed(`Before.\n\n${paragraph}\n\nAfter.`);
    expect(await runMutation(kb, instruction, { ownerHint: "/projects/other.md" }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    generateObjectMock.mockImplementationOnce(async () => {
      await seed("Concurrent external revision.");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, instruction)).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toBe("Concurrent external revision.\n");
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

  it("rejects a correction append when no exact old clause can be repaired", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: {
      ...baseProposal, action: "append", new_text: "Billing API support tooling creates ad-hoc charges only after an approval check.",
    } }).mockResolvedValueOnce({ object: { path: "", old_text: "", new_text: "" } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
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
    } }).mockResolvedValueOnce({ object: {
      new_text: "Monthly charges are scheduled; support tooling creates ad-hoc charges only after approval and records customer SSNs.",
    } });
    const result = await runMutation(kb, updateInstruction);
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
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

  it("rechecks duplicate verdicts claim-by-claim before accepting partial-topic owner overlap", async () => {
    await billingFixture(); staged();
    const ownerPath = "/apis/billing-api.md";
    const fact = "The Billing API logs a request ID for each charge created by support tooling; retry attempts retain the same request ID.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false, reason: "duplicate" } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain("COMPLETE OWNER");
        expect(prompt).toContain("Monthly charges are scheduled");
        expect(prompt).toContain("retry attempts retain the same request ID");
        return { object: { non_duplicate: true, relevant: true, supported: true, consistent: true, certain: true } };
      });
    const result = await runMutation(kb, fact, { ownerHint: ownerPath, preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [ownerPath] } });
    expect((await kb.readConcept(ownerPath)).body).toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["complete duplicate", { non_duplicate: false, relevant: true, supported: true, consistent: true, certain: true }],
    ["contradiction", { non_duplicate: true, relevant: true, supported: true, consistent: false, certain: true }],
    ["uncertainty", { non_duplicate: true, relevant: true, supported: true, consistent: true, certain: false }],
    ["unrelated owner", { non_duplicate: true, relevant: false, supported: true, consistent: true, certain: true }],
  ])("does not accept a duplicate recheck with %s", async (_label, recheck) => {
    await billingFixture(); staged();
    const ownerPath = "/apis/billing-api.md";
    const fact = "The Billing API logs a request ID for each charge created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false, reason: "duplicate" } })
      .mockResolvedValueOnce({ object: recheck });
    const result = await runMutation(kb, fact, { ownerHint: ownerPath, preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(ownerPath)).body).not.toContain(fact);
  });

  it("does not recheck or override contradiction and uncertain safety failures", async () => {
    await billingFixture(); staged();
    const ownerPath = "/apis/billing-api.md";
    const fact = "The Billing API logs a request ID for each charge created by support tooling.";
    for (const reason of ["contradiction", "uncertain"] as const) {
      generateObjectMock.mockReset();
      generateObjectMock.mockResolvedValueOnce({ object: { safe: false, reason } });
      const result = await runMutation(kb, fact, { ownerHint: ownerPath, preflightInput: fact, directAdd: true });
      expect(result).toMatchObject({ ok: false, status: "failed" });
      expect(generateObjectMock).toHaveBeenCalledTimes(1);
      expect((await kb.readConcept(ownerPath)).body).not.toContain(fact);
    }
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

  it("routes an explicitly named personal boundary to its positively matched Person", async () => {
    staged();
    const person = "/users/dan-smart.md";
    const project = "/projects/euroeval.md";
    await kb.writeConcept(person, { type: "Person", title: "Dan Smart" }, "Dan works on evaluation tooling.", "seed");
    await kb.writeConcept(project, { type: "Project", title: "EuroEval" }, "Dan reviewed several EuroEval pull requests.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([
      { path: project, type: "Project", title: "EuroEval", score: 100, confidence: 100, confidenceQualified: true },
    ]);
    const fact = "Dan's GitHub PR review boundary (feedback on EuroEval PR #2237, 16 Oct 2026): While a PR review is in progress, a follow-up such as 'MKQA_NO_CONFIG needs to also have Nynorsk, unless explicitly excluded; check that' is a request to investigate and add a line-level review comment, NOT permission to edit/commit/push to the contributor's branch. Even if the agent has checked out that branch for a separate publishing/testing task, keep code changes read-only until Dan explicitly asks to implement the fix. The agent erroneously pushed 690fc448 and Dan corrected this; do not revert it now. Recorded the boundary in dotfiles/agentic/skills/github/SKILL.md commit a1f7cea.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("COMPETING OWNER /projects/euroeval.md");
      expect(prompt).toContain("Dan reviewed several EuroEval pull requests.");
      return { object: { safe: true } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fact);
      expect(prompt).toContain(`OWNER ${person}`);
      return { object: { safe: true, reason: "Supported personal boundary." } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [person] } });
    expect((await kb.readConcept(person)).body).toContain(fact);
    expect((await kb.readConcept(project)).body).not.toContain(fact);
  });

  it("refuses a personal boundary when a competing owner claims it", async () => {
    staged();
    const person = "/users/dan-smart.md";
    const competitor = "/reviews/euroeval-pr-2237.md";
    await kb.writeConcept(person, { type: "Person", title: "Dan Smart" }, "Dan reviews pull requests.", "seed");
    await kb.writeConcept(competitor, { type: "Review", title: "EuroEval PR #2237" },
      "Dan's GitHub PR review boundary is already recorded here.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: competitor, type: "Review",
      title: "EuroEval PR #2237", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Dan's GitHub PR review boundary: Do not commit to a contributor branch while only reviewing their PR.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`COMPETING OWNER ${competitor}`);
      expect(prompt).toContain("already recorded here");
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(person)).body).not.toContain(fact);
  });

  it("routes an explicitly named personal preference to its positively matched Person despite misleading hits", async () => {
    staged();
    const person = "/users/dan-smart.md";
    const project = "/projects/euroeval.md";
    await kb.writeConcept(person, { type: "Person", title: "Dan Smart" }, "Dan works on evaluation tooling.", "seed");
    await kb.writeConcept(project, { type: "Project", title: "EuroEval" }, "Dan reviewed several EuroEval pull requests.", "seed");
    const search = vi.spyOn(kb, "search").mockImplementation(async (query) => query.toLowerCase() === "dan"
      ? Array.from({ length: 200 }, (_, index) => ({ path: `/projects/incidental-${index}.md`,
          type: "Project", title: `Dan incidental project ${index}`, score: 100 - index,
          confidence: 100, confidenceQualified: true }))
      : [{ path: project, type: "Project", title: "EuroEval", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Dan's GitHub review preference: Always thank the contributor for their contribution in the review text, including when requesting changes. EuroEval PR #2234 review 5518245608 was updated to open with 'Thanks for contributing this, Eirik.' while retaining CHANGES_REQUESTED.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("COMPETING OWNER /projects/euroeval.md");
      expect(prompt).toContain("Dan reviewed several EuroEval pull requests.");
      return { object: { safe: true } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fact);
      expect(prompt).toContain(`OWNER ${person}`);
      return { object: { safe: true, reason: "" } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [person] } });
    expect(search.mock.calls.some(([query]) => query.toLowerCase() === "dan")).toBe(false);
    expect((await kb.readConcept(person)).body).toContain(fact);
    expect((await kb.readConcept(project)).body).not.toContain(fact);
  });

  it("does not route ordinary project history mentioning Dan to the Person owner", async () => {
    staged();
    const person = "/users/dan-smart.md";
    const project = "/projects/euroeval.md";
    await kb.writeConcept(person, { type: "Person", title: "Dan Smart" }, "Dan works on evaluation tooling.", "seed");
    await kb.writeConcept(project, { type: "Project", title: "EuroEval" }, "EuroEval review history is tracked here.", "seed");
    const fact = "EuroEval project history: Dan reviewed pull request 2234, which retained CHANGES_REQUESTED.";
    vi.spyOn(kb, "search").mockResolvedValue([
      { path: project, type: "Project", title: "EuroEval", score: 100, confidence: 100, confidenceQualified: true },
      { path: person, type: "Person", title: "Dan Smart", score: 1, confidence: 1, confidenceQualified: false },
    ]);
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true, reason: "" } });
    const result = await runMutation(kb, fact, {
      ownerHint: project, preflightInput: fact, directAdd: true,
    });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [project] } });
    expect((await kb.readConcept(project)).body).toContain(fact);
    expect((await kb.readConcept(person)).body).not.toContain(fact);
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

  it("routes a leading repository decision to its canonical owner without PR syntax", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const misleading = "/repos/pi-agent/rate-limit-retry.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Job-bot" },
      "Job-bot manages account verification.", "seed");
    await kb.writeConcept(misleading, { type: "Extension", title: "Rate Limit Retry" },
      "Pi retries provider HTTP 429 responses.", "seed");
    const fact = "Job-bot verification recovery UI decision (Dan, 13 Oct 2026): A blocked login should not show both 'Send verification link' and 'Sign out and sign in again' simultaneously. Deployed e55d148: initially only Send verification link; after successful send replace it with sign-out action. Generic non-verification 401 still offers sign-out. Both AppView and LoggedInView follow this pattern.";
    const search = vi.spyOn(kb, "search").mockResolvedValue([{
      path: misleading, type: "Extension", title: "Rate Limit Retry", score: 100,
      confidence: 100, confidenceQualified: true,
    }]);
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("COMPETING OWNER")) {
        expect(prompt).toContain("Pi retries provider HTTP 429 responses.");
        expect(prompt).toContain(owner);
      } else {
        expect(prompt).toContain(`OWNER ${owner}`);
        expect(prompt).toContain(fact);
      }
      return { object: { safe: true, reason: "Supported by the explicit repository identity." } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect(search.mock.calls.some(([query]) => query.toLowerCase() === "job-bot")).toBe(false);
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(misleading)).body).not.toContain(fact);
  });

  it("routes the exact Pi dotfiles extensions heading to its positively anchored repository", async () => {
    staged();
    const owner = "/repos/dotfiles.md";
    const misleading = "/repos/pi-agent/overview.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Dotfiles Repository" },
      "The repository owns agentic/pi/extensions/ and their tests.", "seed");
    await kb.writeConcept(misleading, { type: "Repository", title: "Pi Agent Overview" },
      "Pi agent overview and runtime notes.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: misleading, type: "Repository",
      title: "Pi Agent Overview", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Pi dotfiles extensions tests: `agentic/pi/extensions/package.json` is CommonJS and TypeScript `.test.ts` files fail under direct `node --experimental-strip-types --test` (and this Node 24 build has no `--experimental-default-type=module`). Use `(cd agentic/pi/extensions && npm test)` instead: `test-runner.mjs` loads TS tests via Jiti. For loop/notify integration, `_loop_state/state.ts` exports process-wide `isLoopActive()`; loop remains active at `agent_end` and transitions on `agent_settled`, so notify can suppress iteration-end alerts there.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fact);
      if (prompt.includes("COMPETING OWNER")) {
        expect(prompt).toContain("Pi agent overview and runtime notes.");
      } else {
        expect(prompt).toContain(`OWNER ${owner}`);
      }
      return { object: { safe: true, reason: "Supported and consistent." } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(misleading)).body).not.toContain(fact);
  });

  it.each([
    { type: "Repository", title: "Dotfiles Archive", body: "The repository owns agentic/pi/extensions/." },
    { type: "Project", title: "Dotfiles Repository", body: "The repository owns agentic/pi/extensions/." },
    { type: "Repository", title: "Dotfiles Repository", body: "The repository owns agentic/pi/themes/." },
  ])("does not locate dotfiles from noncanonical metadata or source path ($type, $title)", async ({ type, title, body }) => {
    staged();
    const owner = "/repos/dotfiles.md";
    await kb.writeConcept(owner, { type, title }, body, "seed");
    const fact = "Pi dotfiles extensions tests: agentic/pi/extensions/ serializes interactive prompts.";
    vi.spyOn(kb, "search").mockResolvedValue([]);
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: false } })
      .mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("blocks the canonical dotfiles append when a ranked specialized owner conflicts", async () => {
    staged();
    const owner = "/repos/dotfiles.md";
    const competitor = "/extensions/pi-interactive-queue.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Dotfiles Repository" },
      "The repository owns agentic/pi/extensions/ and their tests.", "seed");
    await kb.writeConcept(competitor, { type: "Extension", title: "Pi Interactive Queue" },
      "The Pi dotfiles extensions tests claim conflicts: prompts are not serialized.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: competitor, type: "Extension",
      title: "Pi Interactive Queue", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Pi dotfiles extensions tests: agentic/pi/extensions/ serializes interactive prompts.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("COMPETING OWNER");
      expect(prompt).toContain("prompts are not serialized");
      return { object: { safe: false, reason: "The specialized owner conflicts." } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("blocks a canonical repository append when a ranked owner conflicts", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const competitor = "/decisions/job-bot-login.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Job-bot" },
      "Job-bot maintains login flows.", "seed");
    await kb.writeConcept(competitor, { type: "Decision", title: "Login Recovery" },
      "The previous Job-bot login recovery decision conflicts with this proposed UI behavior.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: competitor, type: "Decision",
      title: "Login Recovery", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Job-bot verification recovery UI decision: Initially show only Send verification link; after success show sign-out.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("COMPETING OWNER");
      expect(prompt).toContain("conflicts with this proposed UI behavior");
      return { object: { safe: false } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
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
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("COMPETING OWNER")) expect(prompt).toContain("Pi retries provider HTTP 429 responses.");
      else expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("routes a named pull-request fact to its existing repository despite unrelated search hits", async () => {
    staged();
    const owner = "/repos/prezzi.md";
    const unrelated = "/repos/pi-agent/node-runtime.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Syv.ai Slidev Visual Editor (prezzi)" },
      "Prezzi edits Slidev presentation decks and maintains its visual editor.", "seed");
    await kb.writeConcept(unrelated, { type: "Repository", title: "Pi Node Runtime" },
      "Pi uses a Node runtime for its extensions.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: unrelated, type: "Repository", title: "Pi Node Runtime",
      score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Prezzi PR #18 testing gotchas: deck-local Slidev 52.16.0 differs from root Slidev 52.19.1. Restart the Vite server after changing visual-sync.ts.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: false } })
      .mockImplementationOnce(async (request: { prompt: string }) => {
        expect(request.prompt).toContain(`OWNER ${owner}`);
        expect(request.prompt).toContain("Pi Node Runtime");
        return { object: { safe: true } };
      });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
  });

  it("routes an exact EuroEval PR outcome to its canonical Repository despite a misleading PR #2157 hit", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    const misleading = "/procedures/euroeval-sklep-slovak-datasets-pr-2157.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval Repository" },
      "EuroEval is the repository for evaluation datasets.", "seed");
    await kb.writeConcept(misleading, { type: "Procedure", title: "EuroEval SkLEP Slovak Datasets PR #2157" },
      "A Slovak dataset procedure unrelated to MKQA publication outcomes.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: misleading, type: "Procedure",
      title: "EuroEval SkLEP Slovak Datasets PR #2157", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "EuroEval PR #2237 operational outcome (16 Oct 2026): Ran src/scripts/dataset_creation/create_mkqa.py on contributor branch, publishing all 13 EuroEval/mkqa-{da,de,en,es,fi,fr,hu,it,nl,no,pl,pt,sv}-mini Hugging Face dataset repositories privately; authenticated load_dataset verified each has train=1024, val=256, test=2048 text/label rows. CHECK_DATASET=mkqa-en tests passed (15, 1 skipped). Pushed 690fc448 to avalyset/EuroEval:add-mkqa-unofficial, adding NORWEGIAN_NYNORSK to MKQA_NO_CONFIG because upstream MKQA only labels 'no' as Norwegian and does not specify an exclusion. PR review 5545034216 remains PENDING and its publication request comment 4292481293 was deleted. IMPORTANT: dataset_utils.upload_private deletes the target repository before republishing; check no existing targets before re-running the script. Known review issues (cross-split duplicate questions and unit-cued distractors) still apply to the published private datasets.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("COMPETING OWNER")) {
        expect(prompt).toContain(`COMPETING OWNER ${misleading}`);
        expect(prompt).toContain("A Slovak dataset procedure unrelated to MKQA publication outcomes.");
      } else {
        expect(prompt).toContain(`OWNER ${owner}`);
        expect(prompt).toContain(fact);
      }
      return { object: { safe: true, reason: "The outcome belongs to EuroEval." } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(misleading)).body).not.toContain(fact);
  });

  it.each([
    { type: "Repository", title: "EuroEval Repository Archive" },
    { type: "Project", title: "EuroEval Repository" },
  ])("does not route to a canonical path with noncanonical metadata ($type)", async ({ type, title }) => {
    staged();
    const owner = "/repos/euroeval.md";
    const misleading = "/procedures/euroeval-sklep-slovak-datasets-pr-2157.md";
    await kb.writeConcept(owner, { type, title }, "Existing EuroEval-related content.", "seed");
    await kb.writeConcept(misleading, { type: "Procedure", title: "EuroEval SkLEP Slovak Datasets PR #2157" },
      "A Slovak dataset procedure.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: misleading, type: "Procedure",
      title: "EuroEval SkLEP Slovak Datasets PR #2157", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "EuroEval PR #2237 operational outcome (16 Oct 2026): Verified the private dataset publication outcome.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: false } })
      .mockResolvedValueOnce({ object: { safe: false, reason: "The selected procedure is not the owner." } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect((await kb.readConcept(misleading)).body).not.toContain(fact);
  });

  it("does not append a canonical EuroEval fact when a competing PR owner conflicts", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    const competitor = "/procedures/euroeval-sklep-slovak-datasets-pr-2157.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval Repository" },
      "EuroEval is the repository for evaluation datasets.", "seed");
    await kb.writeConcept(competitor, { type: "Procedure", title: "EuroEval SkLEP Slovak Datasets PR #2157" },
      "The EuroEval PR #2237 operational outcome contradicts the supplied fact.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: competitor, type: "Procedure",
      title: "EuroEval SkLEP Slovak Datasets PR #2157", score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "EuroEval PR #2237 operational outcome (16 Oct 2026): Verified the private dataset publication outcome.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`COMPETING OWNER ${competitor}`);
      expect(prompt).toContain("contradicts the supplied fact");
      return { object: { safe: false, reason: "The competing owner conflicts." } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("creates a distinct named pull-request owner only after checking complete competing bodies", async () => {
    staged();
    const unrelated = "/repos/pi-agent/node-runtime.md";
    await kb.writeConcept(unrelated, { type: "Repository", title: "Pi Node Runtime" },
      "Pi uses a Node runtime for its extensions.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: unrelated, type: "Repository", title: "Pi Node Runtime",
      score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Prezzi PR #18 testing gotchas: deck-local Slidev 52.16.0 differs from root Slidev 52.19.1. Restart the Vite server after changing visual-sync.ts.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: false } })
      .mockImplementationOnce(async (request: { prompt: string }) => {
        expect(request.prompt).toContain("FULL EXISTING SEARCH EVIDENCE");
        expect(request.prompt).toContain("Pi uses a Node runtime");
        expect(request.prompt).toContain("/repos/prezzi.md");
        return { object: { safe: true } };
      });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: ["/repos/prezzi.md"] } });
    expect((await kb.readConcept("/repos/prezzi.md")).body).toBe(`${fact}\n`);
  });

  it("finds an existing named PR owner absent from the first search hits", async () => {
    staged();
    const owner = "/projects/prezzi-editor.md";
    const unrelated = "/repos/pi-agent/node-runtime.md";
    const fact = "Prezzi PR #18 testing gotchas: deck-local Slidev 52.16.0 differs from root Slidev 52.19.1.";
    await kb.writeConcept(owner, { type: "Project", title: "Prezzi Editor" },
      "Prezzi edits Slidev decks and supports a visual editor.", "seed");
    await kb.writeConcept(unrelated, { type: "Repository", title: "Pi Node Runtime" },
      "Pi uses a Node runtime for its extensions.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "Prezzi"
      ? [{ path: owner, type: "Project", title: "Prezzi Editor", score: 5, confidenceQualified: false }]
      : [{ path: unrelated, type: "Repository", title: "Pi Node Runtime", score: 100,
        confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: true } })
      .mockImplementationOnce(async (request: { prompt: string }) => {
        expect(request.prompt).toContain(`OWNER ${owner}`);
        return { object: { safe: true } };
      });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect(await kb.bundle.exists("/repos/prezzi.md")).toBe(false);
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("does not duplicate an exact fact in a named PR owner absent from initial hits", async () => {
    staged();
    const owner = "/projects/prezzi-editor.md";
    const unrelated = "/repos/pi-agent/node-runtime.md";
    const fact = "Prezzi PR #18 testing gotchas: deck-local Slidev 52.16.0 differs from root Slidev 52.19.1.";
    await kb.writeConcept(owner, { type: "Project", title: "Prezzi Editor" }, fact, "seed");
    await kb.writeConcept(unrelated, { type: "Repository", title: "Pi Node Runtime" },
      "Pi uses a Node runtime for its extensions.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "Prezzi"
      ? [{ path: owner, type: "Project", title: "Prezzi Editor", score: 5, confidenceQualified: false }]
      : [{ path: unrelated, type: "Repository", title: "Pi Node Runtime", score: 100,
        confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: true } })
      .mockResolvedValueOnce({ object: { supported: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(await kb.bundle.exists("/repos/prezzi.md")).toBe(false);
    expect((await kb.readConcept(owner)).body.match(/deck-local Slidev/g)).toHaveLength(1);
  });

  it("rejects a named pull-request owner when the independent evidence detects a duplicate", async () => {
    staged();
    const existing = "/projects/prezzi-editor.md";
    const fact = "Prezzi PR #18 testing gotchas: deck-local Slidev 52.16.0 differs from root Slidev 52.19.1.";
    await kb.writeConcept(existing, { type: "Project", title: "Prezzi Editor" }, fact, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: existing, type: "Project", title: "Prezzi Editor",
      score: 100, confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(await kb.bundle.exists("/repos/prezzi.md")).toBe(false);
  });

  it("does not infer a repository from a named PR without software-checkout evidence", async () => {
    staged();
    const unrelated = "/repos/pi-agent/node-runtime.md";
    await kb.writeConcept(unrelated, { type: "Repository", title: "Pi Node Runtime" },
      "Pi uses a Node runtime for its extensions.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: unrelated, type: "Repository", title: "Pi Node Runtime",
      score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Rate PR #18 policy decides who is allowed to approve billing invoices.";
    generateObjectMock.mockResolvedValueOnce({ object: { repositorySubject: false } })
      .mockResolvedValueOnce({ object: { path: "" } })
      .mockResolvedValueOnce({ object: { ...baseProposal, action: "defer", reason: "No owner." } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(await kb.bundle.exists("/repos/rate.md")).toBe(false);
  });

  it("routes a canonical Job Bot knowledge owner despite unrelated ranked hits", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const unrelated = "/repos/other-tool.md";
    const fact = "Job-bot PWA install nudge (commit 224557e, deployed 2026-10-14): Browser-only mobile banner appears for authenticated AppView after 3 separate visits to the app on that device, counting a new visit only after at least 30 minutes between foreground openings (not every route change or Auth0 redirect). " +
      "Standalone Home Screen apps and desktop are excluded; offline view excludes it. iOS shows Safari Share → Add to Home Screen instructions; Android uses beforeinstallprompt if available, else Chrome menu instructions. " +
      "Dismissing snoozes 30 days via localStorage key `job-bot-install-visits`; passive install information remains available in Settings, but first-visit landing install card was removed. Tests cover visit threshold/snooze/storage failure. Production SW uses content-hash cache version and precaches updated AppView.";
    await kb.writeConcept(owner, { type: "Knowledge", title: "Job Bot" },
      "Job-bot has a PWA install guide for authenticated users.", "seed");
    await kb.writeConcept(unrelated, { type: "Repository", title: "Other Tool" },
      "Other Tool stores unrelated deployment notes.", "seed");
    const search = vi.spyOn(kb, "search").mockResolvedValue([{ path: unrelated, type: "Repository", title: "Other Tool",
      score: 100, confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(owner);
      expect(prompt).toContain(fact);
      return { object: { safe: true } };
    });

    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
    expect(search.mock.calls.some(([query]) => query.toLowerCase() === "job-bot")).toBe(true);
  });

  it("blocks a Knowledge repo append when an unranked named owner has the full fact", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const competitor = "/features/job-bot-pwa.md";
    const fact = "Job-bot PWA install nudge (commit 224557e): Three mobile visits show an install banner after 30-minute foreground gaps.";
    await kb.writeConcept(owner, { type: "Knowledge", title: "Job Bot" },
      "Job-bot has a PWA install guide.", "seed");
    await kb.writeConcept(competitor, { type: "Feature", title: "Job-bot PWA" }, fact, "seed");
    const search = vi.spyOn(kb, "search").mockImplementation(async (query: string) =>
      query.toLowerCase() === "job-bot" ? [{ path: competitor, type: "Feature", title: "Job-bot PWA", score: 100 }]
        : [{ path: owner, type: "Knowledge", title: "Job Bot", score: 100 }]);
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed",
      error: expect.stringContaining("exact fact already belongs to another named owner") });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect(search.mock.calls.some(([query]) => query.toLowerCase() === "job-bot")).toBe(true);
  });

  it.each([false, true])("finds a late named owner beyond 200 incidental mentions (duplicate=%s)", async (duplicate) => {
    staged();
    const owner = "/repos/job-bot.md";
    const competitor = "/features/job-bot-auth0.md";
    const fact = "Job-bot Auth0 diagnosis (Oct 2026): a database/password identity was unverified, " +
      "while a separate Google identity for that address was verified.";
    await kb.writeConcept(owner, { type: "Knowledge", title: "Job Bot" },
      "Job-bot uses Auth0 authentication.", "seed");
    await kb.writeConcept(competitor, { type: "Feature", title: "Job-bot Auth0" },
      duplicate ? fact : "Job-bot Auth0 deployment notes, without the identity diagnosis.", "seed");
    const noiseDir = path.join(root, "noise");
    await fs.mkdir(noiseDir);
    await Promise.all(Array.from({ length: 201 }, (_, i) => fs.writeFile(path.join(noiseDir, `note-${i}.md`),
      "---\ntype: Note\ntitle: Other topic\n---\nIncidental Job-bot mention.\n")));
    const namedHits = [
      { path: owner, type: "Knowledge", title: "Job Bot", score: 100, confidenceQualified: false },
      ...Array.from({ length: 201 }, (_, i) => ({ path: `/noise/note-${i}.md`, type: "Note",
        title: "Other topic", score: 50, confidenceQualified: false })),
      { path: competitor, type: "Feature", title: "Job-bot Auth0", score: 1, confidenceQualified: false },
    ];
    const search = vi.spyOn(kb, "search").mockImplementation(async (query: string, options) =>
      query.toLowerCase() === "job-bot" ? namedHits.slice(0, options?.limit) : namedHits.slice(0, 3));
    generateObjectMock.mockResolvedValue({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    if (duplicate) {
      expect(result).toMatchObject({ ok: false, status: "failed",
        error: expect.stringContaining("exact fact already belongs to another named owner") });
      expect((await kb.readConcept(owner)).body).not.toContain(fact);
    } else {
      expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
      expect((await kb.readConcept(owner)).body).toContain(fact);
    }
    expect(search.mock.calls.some(([query, options]) => query.toLowerCase() === "job-bot" &&
      (options?.limit ?? 0) > 200)).toBe(true);
  });

  it("routes a project add to the unique continuation owning its exact artifact, while checking parent and peers", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const continuation = "/projects/fs-fotovalidering-continuation-17a45f887bbc97ab.md";
    const unrelated = "/projects/fs-fotovalidering/auto-crop-investigation.md";
    const artifact = "file:///tmp/fs-test-all-labels-prior-ribbons-20261015.html";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks the 257 test photos.", "seed");
    await kb.writeConcept(continuation, { type: "Project Continuation", title: "FS Fotovalidering — Annotation UI" },
      `Earlier annotation behavior in ${artifact}: Up/Down navigates without marking a photo reviewed.`, "seed");
    await kb.writeConcept(unrelated, { type: "Note", title: "FS Fotovalidering — Auto Crop" },
      `The YuNet auto-crop study only cross-references ${artifact}, plus a distinct ${artifact}.bak file.`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: continuation, type: "Project Continuation",
      title: "FS Fotovalidering — Annotation UI", score: 100, confidenceQualified: true }]);
    const fact = `FS Fotovalidering test-photo annotation HTML ${artifact}. Show the position progress bar independently of reviewed count, and manual heights for exactly 94 photos, never for the other 163.`;
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("COMPETING PROJECT OWNERS:")) {
        expect(prompt).toContain(`The proposed owner is ${continuation}`);
        expect(prompt).toContain(owner);
        expect(prompt).toContain(unrelated);
        expect(prompt).not.toContain(`"path":"${continuation}"`);
      } else {
        expect(prompt).toContain(`OWNER ${continuation}`);
      }
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [continuation] } });
    expect((await kb.readConcept(continuation)).body).toContain(fact);
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("selects the UI continuation over a privacy continuation mentioning the same file", async () => {
    staged();
    const parent = "/projects/fs-fotovalidering.md";
    const artifact = "file:///tmp/shared-review.html";
    const privacy = "/projects/fs-fotovalidering-continuation-11111111.md";
    const ui = "/projects/fs-fotovalidering-continuation-22222222.md";
    await kb.writeConcept(parent, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "The FS test-photo project has an annotation page.", "seed");
    await kb.writeConcept(privacy, { type: "Project Continuation", title: "FS Fotovalidering — Private Photos" },
      `Private photos in ${artifact} are served only through the token-gated local image server.`, "seed");
    await kb.writeConcept(ui, { type: "Project Continuation", title: "FS Fotovalidering — Annotation UI" },
      `The annotation UI in ${artifact} saves the reviewed count and lets Up/Down arrows navigate without marking photos reviewed.`, "seed");
    const fact = `FS Fotovalidering annotation page ${artifact} now moves its visual position bar on Up/Down navigation independently of reviewed count.`;
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Read-only artifact owner selection")) {
        expect(prompt).toContain(privacy);
        expect(prompt).toContain(ui);
        return { object: { path: ui } };
      }
      if (prompt.includes("COMPETING PROJECT OWNERS:")) {
        expect(prompt).toContain(`The proposed owner is ${ui}`);
        expect(prompt).toContain(privacy);
        expect(prompt).toContain(parent);
      } else {
        expect(prompt).toContain(`OWNER ${ui}`);
      }
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [ui] } });
    expect((await kb.readConcept(ui)).body).toContain(fact);
    expect((await kb.readConcept(privacy)).body).not.toContain(fact);
  });

  it("keeps the parent route and reports paths when two continuations own the same artifact", async () => {
    staged();
    const parent = "/projects/fs-fotovalidering.md";
    const artifact = "file:///tmp/shared-review.html";
    const first = "/projects/fs-fotovalidering-continuation-11111111.md";
    const second = "/projects/fs-fotovalidering-continuation-22222222.md";
    await kb.writeConcept(parent, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "The FS test-photo project has several annotation pages.", "seed");
    for (const path of [first, second]) {
      await kb.writeConcept(path, { type: "Project Continuation", title: "FS Fotovalidering — Annotation" },
        `The annotation page is ${artifact}.`, "seed");
    }
    const warning = vi.spyOn(console, "warn").mockImplementation(() => {});
    generateObjectMock.mockResolvedValue({ object: { safe: false } });
    const fact = `FS Fotovalidering annotation page ${artifact} now has an independent position progress bar.`;
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(warning).toHaveBeenCalledWith("[understory] artifact owner not unique", {
      project: parent, matches: expect.arrayContaining([first, second]),
    });
    expect((await kb.readConcept(first)).body).not.toContain(fact);
    expect((await kb.readConcept(second)).body).not.toContain(fact);
  });

  it("does not bypass a competing-project duplicate when rerouting to an artifact owner", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const continuation = "/projects/fs-fotovalidering-continuation-17a45f887bbc97ab.md";
    const competitor = "/projects/fs-fotovalidering/auto-crop-investigation.md";
    const artifact = "file:///tmp/fs-test-all-labels-prior-ribbons-20261015.html";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" }, "Parent project.", "seed");
    await kb.writeConcept(continuation, { type: "Project Continuation", title: "FS Fotovalidering — Annotation UI" },
      `Earlier UI behavior for ${artifact}.`, "seed");
    await kb.writeConcept(competitor, { type: "Note", title: "FS Fotovalidering — Auto Crop" },
      "Competing complete claim about the same review count.", "seed");
    const fact = `FS Fotovalidering annotation HTML ${artifact}: display manual values for the 94 reviewed sample IDs.`;
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("COMPETING PROJECT OWNER:") && prompt.includes(`"path":"${competitor}"`)) {
        return { object: { safe: false } };
      }
      return { object: { safe: !prompt.includes("COMPETING PROJECT OWNERS:") } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed",
      error: `Staged mutation deferred: competing project owner has a duplicate, contradiction, or stronger claim (${competitor}; target=${continuation}).` });
    expect(generateObjectMock.mock.calls.some(([request]) =>
      request.prompt.includes("COMPETING PROJECT OWNER:") && request.prompt.includes(`"path":"${competitor}"`))).toBe(true);
    expect((await kb.readConcept(continuation)).body).not.toContain(fact);
  });

  it("routes an explicitly named project add to its canonical owner despite a narrower search hit", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const narrower = "/projects/fs-fotovalidering/auto-crop-investigation.md";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks the 257-image Excel-first baseline and YuNet factor gate.", "seed");
    await kb.writeConcept(narrower, { type: "Note", title: "FS Fotovalidering — Automatic Passport-Crop Investigation" },
      "YuNet crop experiments use earlier test photos and owner framing reviews.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: narrower, type: "Note",
      title: "FS Fotovalidering — Automatic Passport-Crop Investigation", score: 100,
      confidence: 100, confidenceQualified: true }]);
    const fact = "FS Fotovalidering 15 Oct 2026 owner-completed YuNet factor gate: 94 owner choices and Excel-first label update; factor 1.15 remains calibration-only.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("COMPETING PROJECT OWNERS:");
      expect(request.prompt).toContain(narrower);
      expect(request.prompt).toContain("YuNet crop experiments");
      return { object: { safe: true } };
    }).mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(`OWNER ${owner}`);
      expect(request.prompt).toContain("FS Fotovalidering tracks the 257-image Excel-first baseline");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(narrower)).body).not.toContain(fact);
  });

  it("continues a near-full explicitly named project only after complete owner checks", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const narrower = "/projects/fs-fotovalidering/auto-crop-investigation.md";
    const body = `FS Fotovalidering tracks the YuNet factor gate.\n${"x".repeat(63_000)}`;
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" }, body, "seed");
    await kb.writeConcept(narrower, { type: "Note", title: "FS Fotovalidering — Automatic Passport-Crop Investigation" },
      "YuNet crop experiments from an earlier checkpoint.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: narrower, type: "Note",
      title: "FS Fotovalidering — Automatic Passport-Crop Investigation", score: 100,
      confidence: 100, confidenceQualified: true }]);
    const fact = `FS Fotovalidering 15 Oct 2026 owner-completed YuNet factor gate: ${"94 verified owner choices. ".repeat(52)}`;
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("COMPETING PROJECT OWNERS:");
      expect(request.prompt).toContain(narrower);
      return { object: { safe: true } };
    }).mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(`OWNER ${owner}`);
      expect(request.prompt).toContain("x".repeat(63_000));
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [expect.stringMatching(
      /^\/projects\/fs-fotovalidering-continuation-[a-f0-9]{16}\.md$/,
    )] } });
    expect((await kb.readConcept(owner)).body).toBe(`${body}\n`);
    expect((await kb.readConcept(result.result!.filesChanged[0])).body).toContain(fact.trim());
  });

  it("permits a distinct synthetic UI fact after independent full-peer adjudication", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const peer = "/projects/fs-fotovalidering-tono-onot.md";
    const fact = "FS Fotovalidering synthetic review UI on port 8790 now persists annotation edits in IndexedDB and an authenticated SQLite autosave service.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "The project evaluates photo quality with a synthetic dataset.", "seed");
    await kb.writeConcept(peer, { type: "Project", title: "FS Fotovalidering — TONO+ONOT" },
      "The TONO+ONOT experiment compares synthetic training sets and F1 scores.", "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("DISPUTED FULL PEER:")) {
        expect(prompt).toContain("The TONO+ONOT experiment");
        return { object: { distinct_topic: true, complete_duplicate: false,
          contradiction: false, stronger_owner: false, certain: true } };
      }
      if (prompt.includes("COMPETING PROJECT OWNERS:")) return { object: { safe: false } };
      expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(peer)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("allows a new test-environment observation when a peer shares the project topic", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const peer = "/projects/fs-fotovalidering-tono-onot.md";
    const fact = "FS Fotovalidering test environment gotcha: a fresh worktree used CPython 3.14 and missed _pillow_heif; uv sync --all-groups --python 3.12 and npm ci fixed make test.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering has a Python test suite.", "seed");
    await kb.writeConcept(peer, { type: "Project", title: "FS Fotovalidering — TONO+ONOT" },
      "The TONO+ONOT experiment uses synthetic data for FS Fotovalidering model training.", "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("DISPUTED FULL PEER:")) {
        expect(prompt).toContain("The TONO+ONOT experiment uses synthetic data");
        return { object: { distinct_topic: false, complete_duplicate: false,
          contradiction: false, stronger_owner: false, certain: true } };
      }
      if (prompt.includes("COMPETING PROJECT OWNERS:")) return { object: { safe: false } };
      expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(peer)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("rejects a false complete duplicate when the peer lacks the exact Python failure claim", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const peer = "/projects/fs-fotovalidering-tono-onot.md";
    const fact = "FS Fotovalidering test environment gotcha: in a fresh worktree, CPython 3.14 missed _pillow_heif; uv sync --all-groups --python 3.12 fixed make test after npm ci.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering has a Python test suite.", "seed");
    await kb.writeConcept(peer, { type: "Project", title: "FS Fotovalidering — TONO+ONOT" },
      "The TONO+ONOT experiment passed make test and make check.", "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Independent evidence audit")) {
        expect(prompt).toContain("The TONO+ONOT experiment passed make test");
        return { object: { missingFactClause: "CPython 3.14 missed _pillow_heif", absent: true, certain: true } };
      }
      if (prompt.includes("DISPUTED FULL PEER:")) return { object: { distinct_topic: false,
        complete_duplicate: true, contradiction: false, stronger_owner: false, certain: true } };
      if (prompt.includes("COMPETING PROJECT OWNERS:")) return { object: { safe: false } };
      expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(peer)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it.each([
    { missingFactClause: "CPython 3.15 failed", peerBody: "CPython 3.14 failed; make test passed." },
    { missingFactClause: "CPython 3.14 failed", peerBody: "CPython 3.14 failed; make test passed." },
    { missingFactClause: "make test passed", peerBody: "make test passed." },
  ])("keeps a duplicate veto without a grounded missing claim ($missingFactClause)", async ({ missingFactClause, peerBody }) => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const peer = "/projects/fs-fotovalidering-tono-onot.md";
    const fact = "FS Fotovalidering: CPython 3.14 failed; make test passed after npm ci.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering" }, "Project tests.", "seed");
    await kb.writeConcept(peer, { type: "Project", title: "FS Fotovalidering — TONO+ONOT" }, peerBody, "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Independent evidence audit")) {
        return { object: { missingFactClause, absent: true, certain: true } };
      }
      if (prompt.includes("DISPUTED FULL PEER:")) return { object: { distinct_topic: false,
        complete_duplicate: true, contradiction: false, stronger_owner: false, certain: true } };
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed",
        error: `Staged mutation deferred: competing project owner has a duplicate, contradiction, or stronger claim (${peer}; target=${owner}).` });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("challenges a false stronger-owner claim about a distinct project experiment", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const peer = "/projects/fs-fotovalidering-tono-onot.md";
    const fact = "FS Fotovalidering October owner-blur baseline: godkendt_id2 is image_blurred=1; 257 photos yield TP245 TN8 FP1 FN3.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "The project tracks owner-labelled photos and a dated test baseline.", "seed");
    await kb.writeConcept(peer, { type: "Project", title: "FS Fotovalidering — TONO+ONOT Experiment" },
      "The TONO+ONOT experiment compared synthetic training sets in September.", "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Independent audit of a disputed STRONGER OWNER verdict")) {
        expect(prompt).toContain("FULL PEER BODY:\nThe TONO+ONOT experiment compared synthetic training sets");
        expect(prompt).toContain(fact);
        return { object: { distinctObservation: true, peerTopicQuote: "TONO+ONOT",
          factItemQuote: "godkendt_id2 is image_blurred=1", certain: true } };
      }
      if (prompt.includes("DISPUTED FULL PEER:")) return { object: { distinct_topic: false,
        complete_duplicate: false, contradiction: false, stronger_owner: true, certain: true } };
      if (prompt.includes("COMPETING PROJECT OWNERS:")) return { object: { safe: false } };
      expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(peer)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it.each([
    { peerBody: "The TONO+ONOT experiment compared synthetic training sets.",
      peerTopicQuote: "FS Fotovalidering", factItemQuote: "godkendt_id2", certain: true },
    { peerBody: "The TONO+ONOT experiment already tested godkendt_id2 in the owner baseline.",
      peerTopicQuote: "TONO+ONOT", factItemQuote: "godkendt_id2", certain: true },
    { peerBody: "The TONO+ONOT experiment compared synthetic training sets.",
      peerTopicQuote: "TONO+ONOT", factItemQuote: "godkendt_id2", certain: false },
    { peerBody: "The TONO+ONOT experiment compared synthetic training sets.",
      peerTopicQuote: "TONO+ONOT", factItemQuote: "another_id9", certain: true },
    { peerBody: "The TONO+ONOT experiment also tracks GODKENDT_ID9 in the same baseline.",
      peerTopicQuote: "TONO+ONOT", factItemQuote: "godkendt_id2", certain: true,
      fact: "FS Fotovalidering October baseline: godkendt_id2 is image_blurred=1 and godkendt_id9 has framing_too_close." },
    { peerBody: "The TONO+ONOT experiment compared synthetic training sets.",
      peerTitle: "FS Fotovalidering — TONO+ONOT Experiment GODKENDT_ID9",
      peerTopicQuote: "TONO+ONOT", factItemQuote: "godkendt_id2", certain: true,
      fact: "FS Fotovalidering October baseline: godkendt_id2 is image_blurred=1 and godkendt_id9 has framing_too_close." },
  ])("keeps a stronger-owner veto without a certain, grounded scope distinction ($factItemQuote, $certain)",
    async ({ peerBody, peerTitle, peerTopicQuote, factItemQuote, certain, fact: givenFact }) => {
      staged();
      const owner = "/projects/fs-fotovalidering.md";
      const peer = "/projects/fs-fotovalidering-tono-onot.md";
      const fact = givenFact ?? "FS Fotovalidering October baseline: godkendt_id2 is image_blurred=1.";
      await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
        "The project tracks owner-labelled photos.", "seed");
      await kb.writeConcept(peer, { type: "Project", title: peerTitle ?? "FS Fotovalidering — TONO+ONOT Experiment" },
        peerBody, "seed");
      generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
        if (prompt.includes("Independent audit of a disputed STRONGER OWNER verdict")) {
          return { object: { distinctObservation: true, peerTopicQuote, factItemQuote, certain } };
        }
        if (prompt.includes("DISPUTED FULL PEER:")) return { object: { distinct_topic: false,
          complete_duplicate: false, contradiction: false, stronger_owner: true, certain: true } };
        return { object: { safe: false } };
      });
      expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
        .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing project owner") });
      expect((await kb.readConcept(owner)).body).not.toContain(fact);
      expect((await kb.readConcept(peer)).body).toBe(`${peerBody}\n`);
      expect(generateObjectMock).toHaveBeenCalledTimes(3);
    });

  it.each([
    { finding: { distinct_topic: true, complete_duplicate: true, contradiction: false, stronger_owner: false, certain: true } },
    { finding: { distinct_topic: false, complete_duplicate: false, contradiction: false, stronger_owner: true, certain: true } },
    { finding: { distinct_topic: false, complete_duplicate: false, contradiction: true, stronger_owner: false, certain: true } },
    { finding: { distinct_topic: false, complete_duplicate: false, contradiction: false, stronger_owner: false, certain: false } },
  ])("blocks a disputed peer on duplicate, contradiction, or uncertainty ($finding)", async ({ finding }) => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const peer = "/projects/fs-fotovalidering-tono-onot.md";
    const fact = "FS Fotovalidering synthetic review UI on port 8790 persists annotations to SQLite.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "The project evaluates photos.", "seed");
    await kb.writeConcept(peer, { type: "Project", title: "FS Fotovalidering — TONO+ONOT" },
      "Synthetic review data is tracked by the TONO+ONOT experiment.", "seed");
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Independent evidence audit")) {
        return { object: { missingFactClause: "", absent: false, certain: false } };
      }
      if (prompt.includes("Independent audit of a disputed STRONGER OWNER verdict")) {
        return { object: { distinctObservation: false, peerTopicQuote: "", factItemQuote: "", certain: false } };
      }
      return prompt.includes("DISPUTED FULL PEER:") ? { object: finding } : { object: { safe: false } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed",
        error: `Staged mutation deferred: competing project owner has a duplicate, contradiction, or stronger claim (${peer}; target=${owner}).` });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(finding.complete_duplicate || finding.stronger_owner ? 3 : 2);
  });

  it("rechecks every complete peer when aggregate overlap mistakes a shared corpus for the new observation", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const corpus = "/projects/fs-fotovalidering-continuation-12345678abcdef90.md";
    const history = "/projects/fs-fotovalidering/tono-onot.md";
    const fact = "FS Fotovalidering 15 Oct 2026 near-80 head-height review: 24,285 verified images yielded 719 unreviewed choices (471 fitted, 248 excluded); measurement JSONL SHA fc0186ef8d647356c9f69113b8e134c781a4acee49055c358ce489a71d9fa8c0.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks a dated baseline.", "seed");
    await kb.writeConcept(corpus, { type: "Project Continuation", title: "FS Fotovalidering — Continuation" },
      "An earlier YuNet factor calibration left synthetic 24,285 labels unchanged; no near-80 review was recorded.", "seed");
    await kb.writeConcept(history, { type: "Note", title: "TONO+ONOT Experiment" },
      "The earlier TONO+ONOT experiment uses a different image set.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: corpus, type: "Project Continuation",
      title: "FS Fotovalidering — Continuation", score: 100, confidenceQualified: false }]);
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("COMPETING PROJECT OWNERS:");
      expect(prompt).toContain(corpus);
      expect(prompt).toContain(history);
      return { object: { safe: false } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`\"path\":\"${corpus}\"`);
      expect(prompt).toContain("24,285 labels unchanged");
      return { object: { safe: true } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`\"path\":\"${history}\"`);
      expect(prompt).toContain("different image set");
      return { object: { safe: true } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(corpus)).body).not.toContain("719 unreviewed choices");
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("still rejects a single conflicting peer after a mixed aggregate refusal", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const partial = "/projects/fs-fotovalidering-continuation-12345678abcdef90.md";
    const conflicting = "/projects/fs-fotovalidering/near-80-review.md";
    const fact = "FS Fotovalidering 15 Oct 2026 near-80 head-height review: 719 images were selected.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks a dated baseline.", "seed");
    await kb.writeConcept(partial, { type: "Project Continuation", title: "FS Fotovalidering — Continuation" },
      "The 24,285-image synthetic corpus predates the head-height review.", "seed");
    await kb.writeConcept(conflicting, { type: "Note", title: "Near-80 review" },
      "The near-80 review selected 721 images, not 719.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: partial, type: "Project Continuation",
      title: "FS Fotovalidering — Continuation", score: 100, confidenceQualified: false }]);
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(partial);
        return { object: { safe: true } };
      }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(conflicting);
        return { object: { safe: false } };
      });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("rejects a named project's add when a competing project note already owns it", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const narrower = "/projects/fs-fotovalidering/auto-crop-investigation.md";
    const fact = "FS Fotovalidering 15 Oct 2026 owner-completed YuNet factor gate: 94 owner choices and factor 1.15 calibration-only.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks the 257-image Excel-first baseline.", "seed");
    await kb.writeConcept(narrower, { type: "Note", title: "FS Fotovalidering — Automatic Passport-Crop Investigation" },
      "Owner's 94 measured photos already favor factor 1.15 for YuNet gating; keep the production factor.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: narrower, type: "Note",
      title: "FS Fotovalidering — Automatic Passport-Crop Investigation", score: 100,
      confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("COMPETING PROJECT OWNERS:");
      expect(request.prompt).toContain(narrower);
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("checks an omitted sibling project note for a paraphrased duplicate", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const searchHit = "/projects/unrelated-project.md";
    const competing = "/projects/fs-fotovalidering/owner-framing-review.md";
    const fact = "FS Fotovalidering 15 Oct 2026: YuNet factor 1.15 is best for 94 owner-labelled photos; keep the production setting.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks an Excel-first baseline.", "seed");
    await kb.writeConcept(searchHit, { type: "Project", title: "Unrelated Project" },
      "Old YuNet experiments.", "seed");
    await kb.writeConcept(competing, { type: "Note", title: "Owner Framing Review" },
      "For the 94 manually scored images, gate scaling of 1.15 had the best agreement; do not change the live factor.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: searchHit, type: "Project",
      title: "Unrelated Project", score: 100, confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("COMPETING PROJECT OWNERS:");
      expect(request.prompt).toContain(competing);
      expect(request.prompt).toContain("gate scaling of 1.15");
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("rejects a named project add duplicated in an omitted project continuation", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const competing = "/projects/fs-fotovalidering-continuation-12345678abcdef90.md";
    const fact = "FS Fotovalidering 15 Oct 2026: YuNet factor 1.15 is best for 94 owner-labelled photos.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering tracks an Excel-first baseline.", "seed");
    await kb.writeConcept(competing, { type: "Project Continuation", title: "FS Fotovalidering — Continuation" },
      fact, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project",
      title: "FS Fotovalidering — Phase 1 PoC", score: 100, confidence: 100,
      confidenceQualified: true }]);
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).not.toHaveBeenCalled();
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("checks every same-project owner beyond twelve in complete bounded batches", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const fact = "FS Fotovalidering 15 Oct 2026: 94 owner-labelled photos favor YuNet factor 1.15; leave production unchanged.";
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering — Phase 1 PoC" },
      "FS Fotovalidering uses YuNet on the dated test baseline.", "seed");
    for (let i = 0; i < 13; i++) {
      await kb.writeConcept(`/projects/fs-fotovalidering/note-${i.toString().padStart(2, "0")}.md`,
        { type: "Note", title: `FS note ${i}` },
        i === 12 ? `The last owner's 94 measurements favor 1.15; live factor unchanged. ${"x".repeat(5_900)}`
          : `Unrelated observation ${i}. ${"x".repeat(5_900)}`, "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project",
      title: "FS Fotovalidering — Phase 1 PoC", score: 100, confidence: 100,
      confidenceQualified: true }]);
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).not.toContain("The last owner's 94 measurements");
      return { object: { safe: true } };
    }).mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("The last owner's 94 measurements");
      expect(request.prompt).toContain("note-12.md");
      return { object: { safe: false } };
    }).mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("DISPUTED FULL PEER:");
      expect(request.prompt).toContain("note-12.md");
      return { object: { distinct_topic: false, complete_duplicate: true,
        contradiction: false, stronger_owner: true, certain: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
  });

  it("creates a named Pi CLI gotcha instead of writing to an unrelated project continuation", async () => {
    staged();
    const unrelated = "/projects/fs-fotovalidering-continuation-abcdef1234567890.md";
    await kb.writeConcept(unrelated, { type: "Project", title: "FS Fotovalidering — Continuation" },
      "FS Fotovalidering tracks baseline photo labels.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "xlsx" ? [] : [{
      path: unrelated, type: "Project", title: "FS Fotovalidering — Continuation",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "Pi xlsx skill gotcha on FS Fotovalidering labels.xlsx: `xlsx unpack` and `xlsx sheets` misresolve absolute worksheet relationship Targets; use a surgical ZIP XML edit.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("/tools/xlsx.md");
      expect(request.prompt).toContain("FS Fotovalidering tracks baseline photo labels.");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: ["/tools/xlsx.md"] } });
    expect((await kb.readConcept("/tools/xlsx.md")).body).toBe(`${fact}\n`);
    expect((await kb.readConcept("/tools/xlsx.md")).frontmatter.type).toBe("CLI Tool");
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
  });

  it("routes a named Pi CLI gotcha to an existing tool absent from initial hits", async () => {
    staged();
    const owner = "/tools/xlsx.md";
    const unrelated = "/projects/fs-fotovalidering-continuation-abcdef1234567890.md";
    await kb.writeConcept(owner, { type: "CLI Tool", title: "Xlsx" }, "Xlsx edits Excel workbooks.", "seed");
    await kb.writeConcept(unrelated, { type: "Project", title: "FS Fotovalidering" },
      "FS Fotovalidering tracks baseline photo labels.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "xlsx" ? [{
      path: owner, type: "CLI Tool", title: "Xlsx", score: 5, confidenceQualified: false,
    }] : [{ path: unrelated, type: "Project", title: "FS Fotovalidering",
      score: 100, confidence: 100, confidenceQualified: true }]);
    const fact = "Pi xlsx skill gotcha on FS Fotovalidering labels.xlsx: `xlsx sheets` misresolves absolute worksheet relationship Targets.";
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(unrelated)).body).not.toContain(fact);
  });

  it("does not create a duplicate named Pi CLI owner at another path", async () => {
    staged();
    const owner = "/tools/xlsx-cli.md";
    const unrelated = "/projects/fs-fotovalidering-continuation-abcdef1234567890.md";
    const fact = "Pi xlsx skill gotcha on FS Fotovalidering labels.xlsx: `xlsx sheets` misresolves absolute worksheet relationship Targets.";
    await kb.writeConcept(owner, { type: "CLI Tool", title: "Xlsx CLI" }, fact, "seed");
    await kb.writeConcept(unrelated, { type: "Project", title: "FS Fotovalidering" },
      "FS Fotovalidering tracks baseline photo labels.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "xlsx" ? [{
      path: owner, type: "CLI Tool", title: "Xlsx CLI", score: 5, confidenceQualified: false,
    }] : [{ path: unrelated, type: "Project", title: "FS Fotovalidering",
      score: 100, confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockResolvedValueOnce({ object: { supported: true } });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(await kb.bundle.exists("/tools/xlsx.md")).toBe(false);
    expect((await kb.readConcept(owner)).body).toBe(`${fact}\n`);
  });

  it("finds an existing xlsx gotcha outside the initial project hits", async () => {
    staged();
    const owner = "/gotchas/xlsx-absolute-worksheet-relationships.md";
    const unrelated = "/projects/fs-fotovalidering-continuation-abcdef1234567890.md";
    const fact = "Pi xlsx skill gotcha on FS Fotovalidering labels.xlsx: absolute worksheet relationships misresolve in `xlsx sheets`.";
    await kb.writeConcept(owner, { type: "Gotcha", title: "Xlsx Absolute Worksheet Relationships" },
      "The xlsx CLI can mishandle workbook relationship Targets.", "seed");
    await kb.writeConcept(unrelated, { type: "Project", title: "FS Fotovalidering" },
      "FS Fotovalidering tracks baseline photo labels.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "xlsx" ? [{
      path: owner, type: "Gotcha", title: "Xlsx Absolute Worksheet Relationships", score: 5,
      confidenceQualified: false,
    }] : [{ path: unrelated, type: "Project", title: "FS Fotovalidering",
      score: 100, confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect(await kb.bundle.exists("/tools/xlsx.md")).toBe(false);
    expect((await kb.readConcept(owner)).body).toContain(fact);
  });

  it("rejects a paraphrased duplicate in a competing named xlsx gotcha", async () => {
    staged();
    const owner = "/tools/xlsx.md";
    const competing = "/gotchas/xlsx-absolute-worksheet-relationships.md";
    const fact = "Pi xlsx skill gotcha: `xlsx sheets` resolves absolute worksheet relationship targets to `xl/`.";
    await kb.writeConcept(owner, { type: "CLI Tool", title: "Xlsx" },
      "Xlsx edits spreadsheets surgically.", "seed");
    await kb.writeConcept(competing, { type: "Gotcha", title: "Xlsx Absolute Worksheet Relationships" },
      "Workbook targets beginning with /xl/worksheets cause xlsx sheets to resolve to the xl directory.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "xlsx" ? [
      { path: owner, type: "CLI Tool", title: "Xlsx", score: 5, confidenceQualified: false },
      { path: competing, type: "Gotcha", title: "Xlsx Absolute Worksheet Relationships", score: 4,
        confidenceQualified: false },
    ] : [{ path: owner, type: "CLI Tool", title: "Xlsx", score: 100,
      confidence: 100, confidenceQualified: true }]);
    generateObjectMock.mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("Workbook targets beginning with /xl/worksheets");
      return { object: { safe: false } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).not.toContain(fact);
    expect((await kb.readConcept(competing)).body).not.toContain(fact);
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

  it("accepts grounded multi-fact provenance in an unhinted project continuation", async () => {
    staged();
    const continuation = "/projects/fs-fotovalidering-continuation-873508413abcdef0.md";
    await kb.writeConcept(continuation, { type: "Project Continuation", title: "FS Fotovalidering — Label Audit" },
      "Unknown labels in FS Fotovalidering round-5 are retained with row-level provenance, including workbook sheet, original row number, and source image identifier. Review labels are checked against held-out annotations rather than inferred from neighboring photos.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: continuation, type: "Project Continuation", title: "FS Fotovalidering — Label Audit",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "FS Fotovalidering round-5 unknown labels retain row-level source provenance, including workbook sheet, original row number, and source image identifier; review labels are checked against held-out annotations rather than inferred from neighboring photos.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: continuation } })
      .mockResolvedValue({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [continuation] } });
    expect((await kb.readConcept(continuation)).body).toContain(fact);
  });

  it("does not treat broad project overlap as grounding for an unrelated continuation", async () => {
    staged();
    const continuation = "/projects/fs-fotovalidering-continuation-873508413abcdef0.md";
    await kb.writeConcept(continuation, { type: "Project Continuation", title: "FS Fotovalidering — Label Audit" },
      "FS Fotovalidering photo navigation records arrow-key position and review counts.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: continuation, type: "Project Continuation", title: "FS Fotovalidering — Label Audit",
      score: 100, confidence: 100, confidenceQualified: true,
    }]);
    const fact = "FS Fotovalidering round-5 unknown labels retain row-level source provenance, including workbook sheet, original row number, and source image identifier; review labels are checked against held-out annotations rather than inferred from neighboring photos.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: continuation } })
      .mockResolvedValue({ object: { safe: true } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("unhinted continuation") });
    expect((await kb.readConcept(continuation)).body).not.toContain(fact);
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

  it("considers a named existing repository excluded by long-note search ranking", async () => {
    staged();
    const owner = "/repos/understory.md";
    const wrong = "/repos/pi-agent/understory-query-latency.md";
    await kb.writeConcept(owner, { type: "Repository", title: "Understory Repository" },
      "Understory stores agent memory through an async mutation queue.", "seed");
    await kb.writeConcept(wrong, { type: "Repository", title: "Understory Query Latency" },
      "Understory recall latency depends on search throughput.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: wrong, type: "Repository",
      title: "Understory Query Latency", score: 10, confidenceQualified: false }]);
    const fact = "Understory memory-queue recovery checks historical done receipts before replaying a failed update.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`COMPETING OWNER ${wrong}`);
      expect(prompt).toContain("Understory recall latency depends on search throughput.");
      return { object: { safe: true } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`OWNER ${owner}`);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true })).toMatchObject({
      ok: true, result: { filesChanged: [owner] },
    });
    expect((await kb.readConcept(owner)).body).toContain(fact);
    expect((await kb.readConcept(wrong)).body).not.toContain(fact);
  });

  it("checks all five original owners when routing to a canonical repository", async () => {
    staged();
    const named = "/repos/understory.md";
    const original = [1, 2, 3, 4, 5].map((i) => `/repos/understory-topic-${i}.md`);
    await kb.writeConcept(named, { type: "Repository", title: "Understory Repository" },
      "General Understory context. ".repeat(1_800), "seed");
    for (const [i, owner] of original.entries()) {
      await kb.writeConcept(owner, { type: "Repository", title: `Understory Topic ${i + 1}` },
        `Understory topic ${i + 1} context. `.repeat(500), "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue(original.map((owner, i) => ({
      path: owner, type: "Repository", title: `Understory Topic ${i + 1}`,
      score: 10 - i, confidenceQualified: false,
    })));
    const fact = "Understory's topic 5 already records this incident fact.";
    const checked: string[] = [];
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      const match = prompt.match(/COMPETING OWNER (\/repos\/understory-topic-\d\.md)/);
      expect(match).not.toBeNull();
      checked.push(match![1]);
      expect(prompt).toContain(`Understory topic ${Number(match![1].match(/\d+/)![0])} context.`);
      return { object: { safe: match![1] !== original[4] } };
    });
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing named owner") });
    expect(checked).toEqual(original);
    expect((await kb.readConcept(named)).body).not.toContain(fact);
    expect((await kb.readConcept(original[4])).body).not.toContain(fact);
  });

  it("defers rather than bypassing an oversized canonical repository", async () => {
    staged();
    const named = "/repos/understory.md";
    const relevant = "/repos/understory-memory.md";
    await kb.writeConcept(named, { type: "Repository", title: "Understory Repository" },
      "General Understory context. ".repeat(2_600), "seed");
    await kb.writeConcept(relevant, { type: "Repository", title: "Understory Memory" },
      "Understory memory facts are retained in an async queue.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: relevant, type: "Repository",
      title: "Understory Memory", score: 10, confidenceQualified: false }]);
    const fact = "Understory Memory queue checks completed receipts before retrying a failed write.";
    expect(await runMutation(kb, fact, { preflightInput: fact, directAdd: true }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("named repository owner exceeds evidence limit") });
    expect((await kb.readConcept(relevant)).body).not.toContain(fact);
  });

  it("reports the rejected exact-fact owner and reason without appending", async () => {
    await billingFixture(); staged();
    const fact = "Billing API logs a request ID for each charge created by support tooling.";
    generateObjectMock.mockResolvedValueOnce({ object: { path: "/apis/billing-api.md" } })
      .mockResolvedValueOnce({ object: { safe: false, reason: "contradiction" } });
    const result = await runMutation(kb, fact, { preflightInput: fact, directAdd: true });
    expect(result).toMatchObject({ ok: false, status: "failed",
      error: expect.stringContaining("(contradiction at /apis/billing-api.md)"),
    });
    expect((await kb.readConcept("/apis/billing-api.md")).body).not.toContain(fact);
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

  it.each([
    { ownerHasOldHash: true, peerHasOldHash: false, ownerHasItem: true, passesAnchor: true },
    { ownerHasOldHash: true, peerHasOldHash: false, ownerHasItem: true, strongerPeerTitle: true, passesAnchor: true },
    { ownerHasOldHash: false, peerHasOldHash: true, ownerHasItem: true, passesAnchor: false },
    { ownerHasOldHash: true, peerHasOldHash: true, ownerHasItem: true, passesAnchor: false },
    { ownerHasOldHash: true, peerHasOldHash: false, ownerHasItem: false, passesAnchor: false },
    { ownerHasOldHash: true, peerHasOldHash: false, ownerHasItem: true, laterHasOldHash: true, passesAnchor: false },
  ])("anchors metadata-selected hash corrections to one complete owner ($ownerHasOldHash, $peerHasOldHash, $ownerHasItem, $laterHasOldHash)",
    async ({ ownerHasOldHash, peerHasOldHash, ownerHasItem, laterHasOldHash, strongerPeerTitle, passesAnchor }) => {
      staged();
      const owner = "/projects/label-ledger.md";
      const peer = "/projects/experiment-history.md";
      const oldHash = "e27ae2eaab586175e6952d2411a39904a24817dd57f509cb36925dc43e57c7b3";
      const newHash = "c18af64b9a88cd7ce06f5679d0d3709c6d1fa8b099cd1fb9159349ce70077421";
      const instruction = `Correct the FS Fotovalidering 2026-10-14 baseline owner-blur correction (godkendt_id2): the FINAL authoritative XLSX SHA-256 is \`${newHash}\`, not intermediate \`${oldHash}\`. The current Godkendt!C2 tick was cleared to blank.`;
      const ownerBody = `FS Fotovalidering corrected ${ownerHasItem ? "godkendt_id2" : "godkendt_id9"} on 15 Oct. Workbook SHA old f253 → new ${ownerHasOldHash ? oldHash : "another hash"}. ` +
        "Other baseline observations and notes. ".repeat(1_200);
      const peerBody = `The TONO+ONOT experiment ${peerHasOldHash ? `reused ${oldHash}` : "used synthetic photos"}. ` +
        "Separate experiment history. ".repeat(1_600);
      await kb.writeConcept(owner, { type: "Project Continuation", title: "Label Ledger" }, ownerBody, "seed");
      await kb.writeConcept(peer, { type: "Project", title: strongerPeerTitle
        ? "FS Fotovalidering Baseline Overview" : "September Experiment" }, peerBody, "seed");
      const neutral = "/projects/neutral-notes.md";
      const late = "/projects/later-source.md";
      if (laterHasOldHash) {
        await kb.writeConcept(neutral, { type: "Project", title: "Neutral Notes" }, "Different project notes.", "seed");
        await kb.writeConcept(late, { type: "Project", title: "Later Source" },
          `A copied historical workbook hash ${oldHash}, from another record.`, "seed");
      }
      vi.spyOn(kb, "search").mockResolvedValue([owner, peer, ...(laterHasOldHash ? [neutral, late] : [])]
        .map((conceptPath) => ({
          path: conceptPath, type: "Project", title: conceptPath === owner
            ? "Label Ledger" : conceptPath === peer
              ? strongerPeerTitle ? "FS Fotovalidering Baseline Overview" : "September Experiment" : "Other Project",
          score: 10, confidenceQualified: false,
        })));
      generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
        if (prompt.includes("CANDIDATE METADATA")) {
          expect(prompt).not.toContain("Workbook SHA old f253");
          return { object: { path: owner } };
        }
        expect(prompt).toContain(`CANDIDATES:`);
        expect(prompt).toContain(ownerBody);
        return { object: { ...baseProposal, action: "defer", reason: "owner anchor passed" } };
      });
      const result = await runMutation(kb, instruction, { preflightInput: instruction });
      expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining(
        passesAnchor ? "owner anchor passed" : "metadata-selected owner lacks a positive discriminative anchor") });
      expect(generateObjectMock).toHaveBeenCalledTimes(passesAnchor ? 2 : 1);
      expect((await kb.readConcept(owner)).body).toBe(`${ownerBody}\n`);
    });

  it("does not accept a hash-only replacement for a requested current-cell clearing", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering.md";
    const oldHash = "e27ae2eaab586175e6952d2411a39904a24817dd57f509cb36925dc43e57c7b3";
    const newHash = "c18af64b9a88cd7ce06f5679d0d3709c6d1fa8b099cd1fb9159349ce70077421";
    const oldSentence = `Workbook SHA old f253 → new ${oldHash}.`;
    const body = `FS Fotovalidering godkendt_id2: original \`Godkendt!C2=x\` remains. ${oldSentence}`;
    await kb.writeConcept(owner, { type: "Project", title: "FS Fotovalidering" }, body, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{
      path: owner, type: "Project", title: "FS Fotovalidering", score: 10, confidenceQualified: false,
    }]);
    const instruction = `Correct FS Fotovalidering: final SHA is ${newHash}, not ${oldHash}. In the current workbook, \`Godkendt!C2\` approval tick was cleared to blank; the backup retains C2=x.`;
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldSentence, new_text: oldSentence.replace(oldHash, newHash) } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction })).toMatchObject({
      ok: false, status: "failed", error: "Staged mutation deferred: replacement omits requested current-cell clearing.",
    });
    expect((await kb.readConcept(owner)).body).toContain(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("does not require a quoted continuation path's hash suffix in the corrected body", async () => {
    staged();
    const owner = "/projects/fs-fotovalidering-continuation-4fcee3173b2bf469.md";
    const oldText = "The current workbook C2 approval tick is x.";
    const newText = "The current workbook C2 approval tick is blank.";
    await kb.writeConcept(owner, { type: "Project Continuation", title: "FS Correction Ledger" },
      `FS Fotovalidering: ${oldText}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project Continuation",
      title: "FS Correction Ledger", score: 10, confidenceQualified: false }]);
    const instruction = `Correct \`${owner}\`: the current workbook C2 approval tick is blank, not x.`;
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldText, new_text: newText } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Only the current tick changed." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(newText);
    expect((await kb.readConcept(owner)).body).not.toContain(oldText);
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

  it("updates a uniquely owned named run-status paragraph with several metrics", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const competitor = "/papers/dfm-mimir-v1.md";
    const oldParagraph = "Mimir zero-shot run status: paused at 1,361 rows. " +
      "The Danish and English choice and score evaluations use the resumable checkpoint. " +
      "Jev agreement reflects imitation rather than independently verified accuracy. " +
      "The two unsolicited CSV files were removed; source manifest hashes are retained.";
    const newParagraph = oldParagraph.replace("paused at 1,361 rows", "completed all 12,000 rows") +
      " Danish choice agreement is 82.75% (Brier 0.02827); English choice agreement is 81.17% (Brier 0.02934).";
    const instruction = "Update Claes Mimir zero-shot status: completed all 12,000 rows; Danish choice agreement 82.75%, Brier 0.02827; English choice agreement 81.17%, Brier 0.02934.";
    const surrounding = "Unrelated Claes model-development history remains unchanged. ".repeat(5);
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" },
      `${surrounding}${oldParagraph} ${surrounding}\n\nUnrelated Claes work remains unchanged.`, "seed");
    await kb.writeConcept(competitor, { type: "Paper", title: "DFM Mimir v1" },
      "Mimir zero-shot is an imitation experiment, not a verified accuracy result.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, competitor].map((path, i) => ({
      path, type: i ? "Paper" : "Project", score: 10 - i,
    })));
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "The full status paragraph is corrected." } });
    const outcome = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(outcome.ok, JSON.stringify(outcome)).toBe(true);
    expect(outcome).toMatchObject({ result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(`${surrounding}${newParagraph} ${surrounding}`);
    expect((await kb.readConcept(owner)).body).toContain("Unrelated Claes work remains unchanged.");
    expect((await kb.readConcept(competitor)).body).toContain("not a verified accuracy result");
  });

  it("corrects a Claes Laya throughput status without changing Sparkie Qwen history", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const unrelated = "/notes/sparkie-qwen38-throughput-roadmap.md";
    const history = "The Jev-only taxonomy sampler decision remains part of the project history.";
    const old = "Claes Laya throughput status: High-concurrency live sweep WAS NOT run: the user deferred it until Sparkie's jobbot became idle. The 2–3 day million-sample estimate is unvalidated. " +
      "The task-first benchmark uses a 1,000-name pool with about 38 decisions per assignment and routes to vllm-jev on port 31000. " + history;
    const newStatus = old
      .replace("High-concurrency live sweep WAS NOT run: the user deferred it until Sparkie's jobbot became idle.",
        "High-concurrency live sweep was run once Sparkie's competing jobbot became idle on 13 Oct 2026.")
      .replace("The 2–3 day million-sample estimate is unvalidated.",
        "5.223/s projects to 2.22 days for 1M taxonomy assignments, excluding Luna question generation and Jev answering.") +
      " 100 assignments per worker level at 1/2/4/8/16/32/64 yielded 0.529/1.374/2.478/3.426/4.459/5.029/5.069 assignments/s. " +
      "A second sustained 300-assignment-per-level sweep at 32/48/64/96/128 workers measured 5.223/5.163/5.083/5.112/5.095 assignments/s. " +
      "Throughput plateaus at ~32 workers; recommend --workers 32 on an idle Sparkie. " +
      "All 2,200 benchmark assignments completed; router, Laya and jobbot health endpoints all returned HTTP 200 after the run. " +
      "Reports: /tmp/claes-laya-sweep-20261013/throughput-5ce5dd84cd9d1cf50823d5762168e8ef7f12ed1cee3058a753810fd78760d15c-seed20261013-n100.json and /tmp/claes-laya-sustain-20261013/throughput-5ce5dd84cd9d1cf50823d5762168e8ef7f12ed1cee3058a753810fd78760d15c-seed20261014-n300.json. " +
      "Existing docs/generation.md records results and recommended flag. No million-sample production run was started.";
    const instruction = "Correct Claes Laya throughput status: the high-concurrency live sweep was run once Sparkie's competing jobbot became idle on 13 Oct 2026. Router port 31000, convaiinnovations/laya-multilingual vllm-jev, 1,000-name task pool, task-first 38-ish decisions/assignment, 100 assignments per worker level at 1/2/4/8/16/32/64 yielded 0.529/1.374/2.478/3.426/4.459/5.029/5.069 assignments/s. A second sustained 300-assignment-per-level sweep at 32/48/64/96/128 workers measured 5.223/5.163/5.083/5.112/5.095 assignments/s. Throughput plateaus at ~32 workers; recommend --workers 32 on an idle Sparkie. 5.223/s projects to 2.22 days for 1M taxonomy assignments, excluding Luna question generation and Jev answering. All 2,200 benchmark assignments completed; router, Laya and jobbot health endpoints all returned HTTP 200 after the run. Reports: /tmp/claes-laya-sweep-20261013/throughput-5ce5dd84cd9d1cf50823d5762168e8ef7f12ed1cee3058a753810fd78760d15c-seed20261013-n100.json and /tmp/claes-laya-sustain-20261013/throughput-5ce5dd84cd9d1cf50823d5762168e8ef7f12ed1cee3058a753810fd78760d15c-seed20261014-n300.json. Existing docs/generation.md records results and recommended flag. No million-sample production run was started.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" },
      `Prior Claes history.\n\n${old}\n\nUnrelated experiments stay intact.`, "seed");
    await kb.writeConcept(unrelated, { type: "Note", title: "Sparkie Qwen3.8 Throughput Roadmap" },
      "Qwen3.8 throughput and PLE mmap synchronization on Sparkie remain a separate topic.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, unrelated].map((path, i) => ({
      path, type: i ? "Note" : "Project", title: i ? "Sparkie Qwen3.8 Throughput Roadmap" : "Claes Jev Diagnostics",
      score: 10 - i, confidenceQualified: false,
    })));
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Propose exactly one safe knowledge-base mutation")) return { object: {
        ...baseProposal, action: "replace", path: owner, old_text: old, new_text: newStatus,
      } };
      if (prompt.includes("Independent competing-owner check for a named throughput status")) {
        expect(prompt).toContain(`COMPETING OWNER ${unrelated}`);
        expect(prompt).toContain("Qwen3.8 throughput");
        return { object: { certain: true, sameClaimOwner: false } };
      }
      return { object: { safe: true, reason: "Full owner and competing topic checked." } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(newStatus);
    expect(body).toContain(history);
    expect(body).toContain("Prior Claes history.");
    expect(body).toContain("Unrelated experiments stay intact.");
    expect((await kb.readConcept(unrelated)).body).not.toContain("5.223/s");
  });

  it("does not replace a Claes throughput status duplicated in another owner", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const peer = "/projects/claes-laya-benchmark.md";
    const old = "Claes Laya throughput status: High-concurrency live sweep WAS NOT run. " +
      "The previous 2–3 day million-sample estimate has not been measured. " +
      "The benchmark uses a 1,000-name task pool and the vllm-jev router. " +
      "The pilot ran only serial assignments; later high-concurrency decisions remain untested.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, old, "seed");
    await kb.writeConcept(peer, { type: "Project", title: "Claes Laya Benchmark" }, old, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((path, i) => ({
      path, type: "Project", title: i ? "Claes Laya Benchmark" : "Claes Jev Diagnostics", score: 10 - i,
    })));
    const instruction = "Correct Claes Laya throughput status: high-concurrency sweep was run on 13 Oct 2026; 32 workers measured 5.223 assignments/s.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: old, new_text: old.replace("WAS NOT run", "was run on 13 Oct 2026") +
        " 32 workers measured 5.223 assignments/s." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain("WAS NOT run");
    expect((await kb.readConcept(peer)).body).toContain("WAS NOT run");
  });

  it("does not remove a second unrelated negation with the sweep-status inversion", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const old = "Claes Laya throughput status: High-concurrency live sweep WAS NOT run. " +
      "The unrelated archived Qwen report was NOT compiled during the earlier run. " +
      "Task-first benchmark uses the Sparkie router and a 1,000-name pool. " +
      "Jev validation records remain in their earlier state.";
    const newText = old.replace("WAS NOT run", "was run once on 13 Oct 2026")
      .replace("was NOT compiled", "was compiled") + " 32 workers measured 5.223 assignments/s.";
    const instruction = "Correct Claes Laya throughput status: the high-concurrency live sweep was run once on 13 Oct 2026; 32 workers measured 5.223 assignments/s.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, old, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", title: "Claes Jev Diagnostics", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: old, new_text: newText } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain("was NOT compiled");
  });

  it("blocks a named throughput status when a peer paraphrases the same stale claim", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const peer = "/projects/claes-laya-benchmark.md";
    const old = "Claes Laya throughput status: High-concurrency live sweep WAS NOT run. " +
      "The planned benchmark uses a 1,000-name pool and vllm-jev on the idle Sparkie router. " +
      "The earlier million-assignment duration was unvalidated, because concurrent levels had not been measured. " +
      "The unrelated taxonomy schema is retained for the next experiment.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, old, "seed");
    await kb.writeConcept(peer, { type: "Project", title: "Claes Laya Benchmark" },
      "The Laya concurrent worker run remains pending; throughput measurements at high concurrency are unavailable.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((path, i) => ({
      path, type: "Project", title: i ? "Claes Laya Benchmark" : "Claes Jev Diagnostics", score: 10 - i,
    })));
    const instruction = "Correct Claes Laya throughput status: high-concurrency live sweep was run on 13 Oct 2026; 32 workers measured 5.223 assignments/s.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Propose exactly one safe knowledge-base mutation")) return { object: {
        ...baseProposal, action: "replace", path: owner, old_text: old,
        new_text: old.replace("WAS NOT run", "was run on 13 Oct 2026") + " 32 workers measured 5.223 assignments/s.",
      } };
      if (prompt.includes("Independent competing-owner check for a named throughput status")) {
        expect(prompt).toContain(`COMPETING OWNER ${peer}`);
        return { object: { certain: true, sameClaimOwner: true } };
      }
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain("WAS NOT run");
    expect((await kb.readConcept(peer)).body).toContain("remains pending");
  });

  it.each([false, true])("distinguishes historical peer deferral from a current contradiction (%s)", async (currentStale) => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const peer = "/servers/sparkie.md";
    const old = "Claes Laya throughput status: High-concurrency live sweep WAS NOT run. " +
      "The task-first benchmark uses a 1,000-name pool on router port 31000. " +
      "The unrelated Jev validation history is retained for this project. " +
      "The worker-count checkpoint policy remains separately documented.";
    const newer = old.replace("WAS NOT run", "was run on 13 Oct 2026") +
      " 32 workers measured 5.223 assignments/s.";
    const peerBody = "Earlier scheduling plan on 2 Oct 2026 deferred the Claes Laya high-concurrency sweep. " +
      "Claes Laya high-concurrency sweep completed on 13 Oct 2026 after jobbot became idle. " +
      (currentStale ? "Current Claes Laya sweep status: still pending despite that completed run." : "Qwen GPU roadmap is separate.");
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, old, "seed");
    await kb.writeConcept(peer, { type: "Server", title: "Sparkie" }, peerBody, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((path, i) => ({
      path, type: i ? "Server" : "Project", title: i ? "Sparkie" : "Claes Jev Diagnostics", score: 10 - i,
    })));
    const instruction = "Correct Claes Laya throughput status: the high-concurrency live sweep was run on 13 Oct 2026 after jobbot idle; 32 workers measured 5.223 assignments/s.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Propose exactly one safe knowledge-base mutation")) return { object: {
        ...baseProposal, action: "replace", path: owner, old_text: old, new_text: newer,
      } };
      if (prompt.includes("Independent competing-owner check for a named throughput status")) {
        return { object: { certain: true, sameClaimOwner: true } };
      }
      if (prompt.includes("Independent temporal peer check")) return { object: {
        certain: true, currentStaleQuote: currentStale ? "Current Claes Laya sweep status: still pending" : "",
      } };
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result.ok, JSON.stringify(result)).toBe(!currentStale);
    expect((await kb.readConcept(owner)).body).toContain(currentStale ? "WAS NOT run" : "was run on 13 Oct 2026");
    expect((await kb.readConcept(peer)).body).toBe(`${peerBody}\n`);
  });

  it.each([-1, 0])("requires exact peer-clause adjudication for an invented stale quote (%i)", async (currentStaleIndex) => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const peer = "/servers/sparkie.md";
    const old = "Claes Laya throughput status: High-concurrency live sweep WAS NOT run. " +
      "The task-first benchmark uses router port 31000 on Sparkie. " +
      "The Jev-only taxonomy history stays documented here. " +
      "The checkpoint policy remains independent of the worker count.";
    const newer = old.replace("WAS NOT run", "was run on 13 Oct 2026") + " 32 workers measured 5.223 assignments/s.";
    const peerBody = "Earlier scheduling plan on 2 Oct 2026 deferred the Claes Laya high-concurrency sweep. " +
      "Claes Laya high-concurrency sweep completed on 13 Oct 2026 after jobbot became idle.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, old, "seed");
    await kb.writeConcept(peer, { type: "Server", title: "Sparkie" }, peerBody, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((path, i) => ({
      path, type: i ? "Server" : "Project", title: i ? "Sparkie" : "Claes Jev Diagnostics", score: 10 - i,
    })));
    const instruction = "Correct Claes Laya throughput status: the live sweep was run on 13 Oct 2026 after jobbot idle; 32 workers measured 5.223 assignments/s.";
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      if (prompt.includes("Propose exactly one safe knowledge-base mutation")) return { object: {
        ...baseProposal, action: "replace", path: owner, old_text: old, new_text: newer,
      } };
      if (prompt.includes("Independent competing-owner check for a named throughput status")) {
        return { object: { certain: true, sameClaimOwner: true } };
      }
      if (prompt.includes("Independent temporal peer check")) return { object: {
        certain: true, currentStaleQuote: "Invented current claim nowhere in the peer.",
      } };
      if (prompt.includes("Independent exact-clause adjudication")) {
        expect(prompt).toContain(peerBody);
        return { object: { certain: true, currentStaleIndex } };
      }
      return { object: { safe: true } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result.ok, JSON.stringify(result)).toBe(currentStaleIndex === -1);
    expect((await kb.readConcept(peer)).body).toContain("completed on 13 Oct 2026");
  });

  it("does not let a shared throughput word make Sparkie Qwen the Claes owner", async () => {
    staged();
    const wrong = "/notes/sparkie-qwen38-throughput-roadmap.md";
    const old = "Claes Laya throughput status is mentioned as background. " +
      "Sparkie Qwen3.8 single-stream throughput remains a separate roadmap. " +
      "PLE mmap synchronisation and Qwen optimization are this note's own subject. " +
      "The host also runs vllm-jev for an independent Laya experiment.";
    await kb.writeConcept(wrong, { type: "Note", title: "Sparkie Qwen3.8 Throughput Roadmap" }, old, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: wrong, type: "Note",
      title: "Sparkie Qwen3.8 Throughput Roadmap", score: 100 }]);
    const instruction = "Correct Claes Laya throughput status: the live sweep was run on 13 Oct 2026 with 32 workers at 5.223 assignments/s.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: wrong,
      old_text: old, new_text: old.replace("independent Laya experiment", "completed Laya experiment") +
        " The live sweep was run on 13 Oct 2026 with 32 workers at 5.223 assignments/s." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(wrong)).body).toContain("separate roadmap");
    expect((await kb.readConcept(wrong)).body).not.toContain("5.223");
  });

  it("repairs a named run-status rewrite that drops unrelated history", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const history = "The initial comparison was an imitation experiment and not independently verified accuracy.";
    const oldParagraph = `Mimir zero-shot status: paused at 1,361 rows. ${history} ` +
      "The Danish choice and English choice groups use the resumable checkpoint. " +
      "The two unsolicited CSVs were removed, and the source manifest retains its hashes.";
    const broad = oldParagraph.replace("paused at 1,361 rows", "completed 12,000 rows")
      .replace(history, "The initial comparison used the resumable checkpoint") +
      " Danish choice agreement is 82.75%; English choice agreement is 81.17%.";
    const repaired = oldParagraph.replace("paused at 1,361 rows", "completed 12,000 rows") +
      " Danish choice agreement is 82.75%; English choice agreement is 81.17%.";
    const instruction = "Update Claes Mimir zero-shot status: completed 12,000 rows; Danish choice agreement 82.75%; English choice agreement 81.17%.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: broad } })
      .mockResolvedValueOnce({ object: { new_text: repaired } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Old history preserved and status updated." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(repaired);
    expect((await kb.readConcept(owner)).body).toContain(history);
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("restores an unrequested old status clause after an incomplete model repair", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const history = "The obsolete Julia checksum calibration procedure is archived in a separate report.";
    const oldParagraph = `Mimir zero-shot status: paused at 1,361 rows. ${history} ` +
      "The Danish and English choices use the resumable manifest. " +
      "Jev agreement remains imitation, not independently verified accuracy. " +
      "The two unsolicited CSVs were removed from the checkout.";
    const broad = oldParagraph.replace("paused at 1,361 rows", "completed all 12,000 rows")
      .replace(`${history} `, "") + " Danish choice agreement is 82.75%.";
    const instruction = "Update Claes Mimir zero-shot status: completed all 12,000 rows; Danish choice agreement 82.75%.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: broad } })
      .mockResolvedValueOnce({ object: { new_text: broad } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "History restored without contradiction." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(history);
    expect(body).toContain("completed all 12,000 rows");
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("allows close wording of unrelated status history but not a changed old number", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const history = "The old baseline used 61 rows and was not independently verified accuracy.";
    const oldParagraph = `Mimir zero-shot status: paused at 1,361 rows. ${history} ` +
      "Danish and English choice groups use the resumable checkpoint and verified source manifest hashes. " +
      "The unsolicited CSVs were removed, and the experimental scores remain imitation.";
    const newParagraph = oldParagraph.replace("paused at 1,361 rows", "completed 12,000 rows")
      .replace(history, "The old baseline used 61 rows and is not independently verified accuracy.") +
      " Danish choice agreement is 82.75%; English choice agreement is 81.17%.";
    const instruction = "Update Claes Mimir zero-shot status: completed 12,000 rows; Danish choice agreement 82.75%; English choice agreement 81.17%.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Old 61-row baseline remains unchanged." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(newParagraph);

    const changedNumber = newParagraph.replace("old baseline used 61 rows", "old baseline used 62 rows");
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "reset");
    vi.mocked(generateObjectMock).mockReset();
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: changedNumber } })
      .mockResolvedValueOnce({ object: { new_text: changedNumber } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
  });

  it("rejects a multi-clause status rewrite if the old paragraph has another owner", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const competitor = "/papers/dfm-mimir-v1.md";
    const oldParagraph = "Mimir zero-shot run status: paused at 1,361 rows. " +
      "Danish and English choice agreement reflect the resumable evaluation checkpoint. " +
      "Jev agreement remains imitation, not independently verified accuracy. " +
      "Source manifest hashes and removed unsolicited CSVs remain recorded.";
    const newParagraph = oldParagraph.replace("paused at 1,361 rows", "completed all 12,000 rows") +
      " Danish agreement is 82.75%; English agreement is 81.17%.";
    const instruction = "Update Claes Mimir zero-shot status: completed all 12,000 rows; Danish agreement 82.75%; English agreement 81.17%.";
    for (const conceptPath of [owner, competitor]) {
      await kb.writeConcept(conceptPath, { type: "Project", title: conceptPath === owner ? "Claes Mimir zero-shot status" : "DFM Mimir v1" },
        oldParagraph, "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue([owner, competitor].map((path, i) => ({
      path, type: "Project", score: 10 - i,
    })));
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect((await kb.readConcept(competitor)).body).toContain(oldParagraph);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("rejects a named status that moves a negation onto an unrelated clause", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const oldParagraph = "Mimir zero-shot status: the run paused at 1,361 rows. " +
      "Jev agreement is not independently verified accuracy. " +
      "The unsolicited CSVs were removed. " +
      "The model computes forced-code probabilities, which remain experimental. " +
      "The resumable manifest retains source hashes for the 32-language evaluation.";
    const newParagraph = oldParagraph.replace("paused at 1,361 rows", "completed 12,000 rows")
      .replace("Jev agreement is not independently verified accuracy", "Jev agreement is independently verified accuracy")
      .replace("The unsolicited CSVs were removed", "The unsolicited CSVs were not removed");
    const instruction = "Update Claes Mimir zero-shot status: the run completed 12,000 rows. Jev agreement is not independently verified accuracy. The unsolicited CSVs remain deleted.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" },
      oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Incorrect approval must not authorize this edit." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("locates a unique named incident section outside the broad hits", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const unrelated = [1, 2, 3, 4, 5].map((n) => `/repos/unrelated-${n}.md`);
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const oldClaim =  "Nine sampled retries succeeded.";
    const instruction = "Update the Understory memory_add/memory_update incident note (Oct 15 2026): nine sampled retries failed, not succeeded.";
    for (const [i, conceptPath] of unrelated.entries()) {
      await kb.writeConcept(conceptPath, { type: "Repository", title: `Understory unrelated ${i + 1}` },
        `Unrelated provider notes ${i + 1}. `.repeat(1200), "seed");
    }
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      `${heading} ${oldClaim}\n\nUnrelated memory-async operations remain unchanged.`, "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) =>
      query.includes("Understory memory_add/memory_update incident")
        ? [{ path: owner, type: "Repository", title: "Pi Agent — Memory-Async Extension", score: 5 }]
        : unrelated.map((conceptPath, i) => ({ path: conceptPath, type: "Repository",
          title: `Understory unrelated ${i + 1}`, score: 10 - i })));
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`"path":"${owner}"`);
      expect(prompt).not.toContain(`"path":"${unrelated[0]}"`);
      return { object: { ...baseProposal, action: "replace", path: owner,
        old_text: oldClaim, new_text: "Nine sampled retries failed." } };
    }).mockResolvedValueOnce({ object: { safe: true, reason: "verified against full owner" } });
    const outcome = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(outcome).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain("Nine sampled retries failed.");
    expect((await kb.readConcept(unrelated[0])).body).toContain("Unrelated provider notes");
  });

  it("corrects one superseded PR review sentence with the complete supported approval status", async () => {
    staged();
    const owner = "/repos/prezzi.md";
    const oldStatus = "The prior CHANGES_REQUESTED GitHub review remains unless Dan explicitly asks to update it.";
    const unchanged = "The unrelated visual-sync cleanup remains part of PR #21 history.";
    const newStatus = "Dan explicitly approved the remaining nav behavior, stopped the /tmp demo server on localhost:3031 and submitted a new APPROVED review on PR head c6ee887. GitHub reviewDecision is now APPROVED (the earlier CHANGES_REQUESTED review has been superseded).";
    const instruction = `Update the Syv-ai/prezzi PR #21 status: ${newStatus} Correct any older note saying the review still remains changes requested.`;
    await kb.writeConcept(owner, { type: "Repository", title: "Prezzi" },
      `Prezzi PR #21: ${unchanged} ${oldStatus}\n\nOther PR #18 history is unchanged.`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", title: "Prezzi", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldStatus, new_text: newStatus } })
      .mockResolvedValueOnce({ object: { safe: true } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(`OWNER ${owner}`);
        expect(prompt).toContain(unchanged);
        expect(prompt).toContain(oldStatus);
        expect(prompt).toContain(newStatus);
        return { object: { safe: true, reason: "Approval supersedes the one old review sentence." } };
      });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(newStatus);
    expect(body).toContain(unchanged);
    expect(body).toContain("Other PR #18 history is unchanged.");
    expect(body).not.toContain(oldStatus);
  });

  it("locates a final PR status by repository and number instead of the reviewer's person note", async () => {
    staged();
    const owner = "/repos/euroeval-review.md";
    const person = "/users/dan-smart.md";
    const oldReview = "PR #2234 remains open with changes requested.";
    const finalStatus = "Dan approved it with a review thanking Eirik, then squash-merged it on 15 Oct 2026 as commit ac33731f88806b1a45593c83dbfd473f1cc877fa. Remote feature branch feat/invalid-model-dataset-1453 was deleted; no local branch of that name existed. Local main was fast-forwarded to ac33731f and is clean/up-to-date.";
    const instruction = `EuroEval PR #2234 final status: ${finalStatus} Replace any prior status saying PR is open/changes requested.`;
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval review history" },
      `EuroEval PR #2234: The six review threads were resolved. ${oldReview}\n\nPR #2213 remains independent.`, "seed");
    await kb.writeConcept(person, { type: "Person", title: "Dan Smart" },
      "Dan reviews EuroEval PRs. His prior preference for prompt reviews remains unchanged.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "2234"
      ? [{ path: owner, type: "Repository", title: "EuroEval review history", score: 9 }]
      : [{ path: person, type: "Person", title: "Dan Smart", score: 10 }]);
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`OLD STATUS SPAN:\n${oldReview}`);
      expect(prompt).toContain(`NEW TEXT:\n${finalStatus}`);
      return { object: { safe: true } };
    });
    const outcome = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(outcome, JSON.stringify(outcome)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(finalStatus);
    expect((await kb.readConcept(owner)).body).not.toContain(oldReview);
    expect((await kb.readConcept(owner)).body).toContain("six review threads were resolved");
    expect((await kb.readConcept(person)).body).toContain("prior preference");
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("preserves a mixed-scope PR sentence rather than erasing unrelated status", async () => {
    staged();
    const owner = "/repos/euroeval-review.md";
    const oldText = "PR #2234 remains open, but CI for PR #2200 is also failing.";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval review history" },
      `EuroEval PR #2234: ${oldText}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    const instruction = "EuroEval PR #2234 final status: Dan approved and squash-merged it as commit ac33731f88806b1a45593c83dbfd473f1cc877fa. Replace any prior status saying PR is open.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("MIXED SENTENCE:");
      return { object: { old_text: oldText } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, JSON.stringify(result)).toMatchObject({
      ok: false, status: "failed", error: expect.stringContaining("no unique review-only sentence"),
    });
    expect((await kb.readConcept(owner)).body).toContain(oldText);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("replaces only a whole sentence when every clause is superseded PR status", async () => {
    staged();
    const owner = "/users/dan-smart.md";
    const oldText = "PR #2234 remains open, and its CHANGES_REQUESTED review remains active.";
    const status = "Dan approved it with a review thanking Eirik, then squash-merged it on 15 Oct 2026 as commit ac33731f88806b1a45593c83dbfd473f1cc877fa. Remote feature branch feat/invalid-model-dataset-1453 was deleted; no local branch of that name existed. Local main was fast-forwarded to ac33731f and is clean/up-to-date.";
    const instruction = `EuroEval PR #2234 final status: ${status} Replace any prior status saying PR is open/changes requested.`;
    await kb.writeConcept(owner, { type: "Person", title: "Dan Smart" },
      `EuroEval PR #2234: Prior six review threads were resolved. ${oldText}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Person", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: "" } })
      .mockResolvedValueOnce({ object: { safe: true } })
      .mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(status);
    expect(body).toContain("Prior six review threads were resolved");
    expect(body).not.toContain(oldText);
  });

  it("replaces only the isolated stale PR span inside a mixed sentence", async () => {
    staged();
    const owner = "/users/dan-smart.md";
    const stale = "EuroEval PR #2234 remains open with changes requested";
    const otherClaim = "but all six review threads have been resolved.";
    await kb.writeConcept(owner, { type: "Person", title: "Dan Smart" },
      `EuroEval PR #2234: ${stale}, ${otherClaim}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Person", score: 10 }]);
    const status = "Dan approved it with a review thanking Eirik, then squash-merged it on 15 Oct 2026 as commit ac33731f88806b1a45593c83dbfd473f1cc877fa. Remote feature branch feat/invalid-model-dataset-1453 was deleted; no local branch of that name existed. Local main was fast-forwarded to ac33731f and is clean/up-to-date.";
    const instruction = `EuroEval PR #2234 final status: ${status} Replace any prior status saying PR is open/changes requested.`;
    generateObjectMock.mockResolvedValueOnce({ object: { old_text: stale } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(`OLD STATUS SPAN:\n${stale}`);
        expect(prompt).toContain(otherClaim);
        return { object: { safe: true } };
      });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(status);
    expect(body).toContain(otherClaim);
    expect(body).not.toContain(stale);
  });

  it("does not edit a person's note when a final PR status has no observed owner", async () => {
    staged();
    const person = "/users/dan-smart.md";
    await kb.writeConcept(person, { type: "Person", title: "Dan Smart" },
      "Dan reviews repositories and thanks maintainers for feedback.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "2234" ? [] :
      [{ path: person, type: "Person", title: "Dan Smart", score: 10 }]);
    const instruction = "EuroEval PR #2234 final status: Dan approved and squash-merged it. Replace any prior status saying PR is open.";
    expect(await runMutation(kb, instruction, { preflightInput: instruction })).toMatchObject({
      ok: false, status: "failed", error: expect.stringContaining("no observed active review owner"),
    });
    expect((await kb.readConcept(person)).body).toContain("thanks maintainers");
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("locates the stale PR review outside the repository named in the correction", async () => {
    staged();
    const topic = "/repos/prezzi.md";
    const owner = "/repos/syv-ai-agentbase.md";
    const stale = "The prior CHANGES_REQUESTED GitHub review remains unless he explicitly asks to update it.";
    const corrected = "Dan explicitly approved the remaining nav behavior. GitHub reviewDecision is now APPROVED; the earlier CHANGES_REQUESTED review has been superseded.";
    const instruction = `Update the Syv-ai/prezzi PR #21 status: ${corrected} Correct any older note saying the review still remains changes requested.`;
    const context = "Syv-ai/prezzi PR #21: Dan tested navigation before approving it. ";
    await kb.writeConcept(topic, { type: "Repository", title: "Prezzi" },
      "In Syv-ai/prezzi PR #21, the earlier CHANGES_REQUESTED review was superseded. During review, 82/83 tests passed.", "seed");
    await kb.writeConcept(owner, { type: "Repository", title: "Syv-ai Agentbase" },
      `${context}${stale}\n\nOther history remains.`, "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) => query === "CHANGES_REQUESTED"
      ? [{ path: topic, type: "Repository", score: 10 }, { path: owner, type: "Repository", score: 9 }]
      : [{ path: topic, type: "Repository", title: "Prezzi", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: stale, new_text: corrected } })
      .mockResolvedValueOnce({ object: { safe: true } })
      .mockImplementation(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(`OWNER ${owner}`);
        expect(prompt).toContain(context);
        return { object: { safe: true, reason: "Only this active PR review changed." } };
      });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(corrected);
    expect((await kb.readConcept(owner)).body).toContain(context);
    expect((await kb.readConcept(topic)).body).toContain("CHANGES_REQUESTED review was superseded");
    expect((await kb.readConcept(topic)).body).toContain("82/83 tests passed");
  });

  it("uses exact caller status when a PR-review proposal paraphrases unsupported facts", async () => {
    staged();
    const owner = "/repos/syv-ai-agentbase.md";
    const oldReview = "The prior CHANGES_REQUESTED GitHub review remains unless he explicitly asks to update it.";
    const status = "Dan explicitly approved the remaining nav behavior, and Pi stopped the /tmp demo server on localhost:3031 and submitted a new APPROVED review on PR head c6ee887. GitHub reviewDecision is now APPROVED (the earlier CHANGES_REQUESTED review has been superseded).";
    const instruction = `Update the Syv-ai/prezzi PR #21 status: ${status} Correct any older note saying the review still remains changes requested.`;
    await kb.writeConcept(owner, { type: "Repository", title: "Syv-ai Agentbase" },
      `Syv-ai/prezzi PR #21: Earlier tests remain documented. ${oldReview}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", title: "Syv-ai Agentbase", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldReview, new_text: "The review was APPROVED and all customer data was exported to an external party." } })
      .mockResolvedValueOnce({ object: { safe: true } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(`NEW TEXT:\n${status}`);
        expect(prompt).not.toContain("all customer data was exported");
        return { object: { safe: true, reason: "Exact requested status only." } };
      });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(status);
    expect((await kb.readConcept(owner)).body).toContain("Earlier tests remain documented.");
  });

  it("rejects ambiguous exact PR review owners before proposing a replacement", async () => {
    staged();
    const instruction = "Update the Syv-ai/prezzi PR #21 status: APPROVED replaces the prior CHANGES_REQUESTED review.";
    const body = "Syv-ai/prezzi PR #21: The prior CHANGES_REQUESTED review remains active.";
    const paths = ["/repos/prezzi.md", "/repos/review-copy.md"];
    for (const owner of paths) await kb.writeConcept(owner, { type: "Repository", title: "Prezzi" }, body, "seed");
    vi.spyOn(kb, "search").mockResolvedValue(paths.map((owner) => ({ path: owner, type: "Repository", score: 10 })));
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("multiple active owners") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("narrows a broad proposed operand to the one uniquely scoped PR review sentence", async () => {
    staged();
    const owner = "/repos/syv-ai-agentbase.md";
    const history = "Dan tested the follow-up and approved the nav behavior. ";
    const oldReview = "The prior CHANGES_REQUESTED GitHub review remains unless he explicitly asks to update it.";
    const newReview = "Dan explicitly approved the remaining nav behavior. GitHub reviewDecision is now APPROVED; the earlier CHANGES_REQUESTED review has been superseded.";
    const instruction = `Update the Syv-ai/prezzi PR #21 status: ${newReview} Correct any older note saying the review still remains changes requested.`;
    await kb.writeConcept(owner, { type: "Repository", title: "Syv-ai Agentbase" },
      `Syv-ai/prezzi PR #21: ${history}${oldReview}\n\nUnrelated repository history remains.`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", title: "Syv-ai Agentbase", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: `${history}${oldReview}`, new_text: newReview } })
      .mockResolvedValueOnce({ object: { safe: true } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(`OLD TEXT:\n${oldReview}`);
        expect(prompt).toContain(history);
        return { object: { safe: true, reason: "Only the old review assertion changed." } };
      });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(history);
    expect(body).toContain(newReview);
    expect(body).not.toContain(oldReview);
    expect(body).toContain("Unrelated repository history remains.");
  });

  it.each(["wrong PR", "mixed PR", "mixed claim", "wrong owner", "competing owner", "scope refusal", "verifier refusal"])(
    "does not relax a superseded PR review when there is a %s", async (caseName) => {
      staged();
      const owner = caseName === "wrong owner" ? "/repos/other.md" : "/repos/prezzi.md";
      const oldStatus = "The prior CHANGES_REQUESTED GitHub review remains unless Dan explicitly asks to update it." +
        (caseName === "mixed claim" ? " The unrelated build passed." : "");
      const newStatus = "Dan explicitly approved the remaining nav behavior. GitHub reviewDecision is now APPROVED; the earlier CHANGES_REQUESTED review has been superseded.";
      const instruction = `Update the Syv-ai/prezzi PR #21 status: ${newStatus} Correct any older note saying the review still remains changes requested.`;
      const paragraph = `Prezzi PR #${caseName === "wrong PR" ? 22 : 21}` +
        (caseName === "mixed PR" ? " and PR #22" : "") + `: ${oldStatus}`;
      await kb.writeConcept(owner, { type: "Repository", title: "Prezzi" }, paragraph, "seed");
      const competitor = "/repos/other-copy.md";
      if (caseName === "competing owner") {
        await kb.writeConcept(competitor, { type: "Repository", title: "Prezzi Archive" }, paragraph, "seed");
      }
      vi.spyOn(kb, "search").mockResolvedValue([owner, ...(caseName === "competing owner" ? [competitor] : [])]
        .map((conceptPath) => ({ path: conceptPath, type: "Repository", title: "Prezzi", score: 10 })));
      generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
        old_text: oldStatus, new_text: newStatus } });
      if (caseName === "scope refusal") {
        generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
      } else if (caseName === "verifier refusal") {
        generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => ({ object: {
          safe: !prompt.includes("Independent safety check."), reason: "Uncertain.",
        } }));
      }
      expect(await runMutation(kb, instruction, { preflightInput: instruction }))
        .toMatchObject({ ok: false, status: "failed" });
      expect((await kb.readConcept(owner)).body).toContain(oldStatus);
    },
  );

  it("corrects the uniquely owned Claes training limit without erasing dataset evidence", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const intro = "Claes training-design investigation (Oct 2026): current repo has only dataset generation and Jev answering, no student model/training/evaluation. In first 3,000 live rows read-only: 32 languages; 1,536 choice, 1,017 noul, 447 score; one choice selected ID not probability argmax. ";
    const stale = "Existing public syvai/claes-preview HF checkpoint is a 322M-parameter mmBERT/Laya fine-tune, not under 1M. For strict <1M multilingual student recommend evaluating a shared byte-input CNN-downsample + small Transformer common-context encoder with reusable candidate text encoding and lightweight candidate-to-context scoring, rather than re-encoding state up to 250 times. ";
    const retained = "Distil actual distributions with soft cross entropy/KL at T=1; confidence and fractional score require explicit handling, and held-out human labels are needed to assess truth/calibration vs Jev imitation. These were design recommendations, not an approved architecture decision.";
    const corrected = "The user's actual model limit is <1B; the earlier tiny byte/CNN/Transformer recommendation is superseded. Existing public syvai/claes-preview HF checkpoint is a 322M-parameter mmBERT/Laya fine-tune and fits under <1B; upstream convaiinnovations/laya-multilingual is also 322M and English ModernBERT-large Laya is 421M. Recommend fine-tuning/evaluating these with 1M Jev soft targets rather than building a tiny model from scratch. ";
    const instruction = "Correct the Claes training-design note recorded Oct 2026: the user's actual model limit is <1 BILLION parameters, not <1 million. The earlier tiny byte/CNN/Transformer student recommendation was conditional on the mistaken limit and is superseded. Under <1B, the existing 322M syvai/claes-preview multilingual mmBERT/Laya checkpoint fits and preserves pretraining; upstream convaiinnovations/laya-multilingual is also 322M, and English ModernBERT-large Laya is 421M. Recommend fine-tuning/evaluating these with the 1M Jev soft targets, checking choice/noul/score support and multilingual regression, rather than building a tiny model from scratch. Retain the accurate dataset schema and sample stats.";
    const oldParagraph = intro + stale + retained;
    const newParagraph = intro + corrected + retained;
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" },
      `${oldParagraph}\n\nIndependent survey history remains intact.`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", title: "Claes Jev Diagnostics", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: intro + corrected } })
      .mockResolvedValueOnce({ object: { new_text: newParagraph } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain(`OWNER ${owner}`);
        expect(prompt).toContain("1,536 choice, 1,017 noul, 447 score");
        return { object: { safe: true, reason: "Unrelated evidence retained." } };
      });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(newParagraph);
    expect(body).not.toContain(stale);
    expect(body).toContain("Independent survey history remains intact.");
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("restores a Claes dataset clause verbatim after an incomplete repair", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const oldParagraph = "Claes training-design investigation (Oct 2026): current repo has no student model/training/evaluation. For strict <1M multilingual student recommend evaluating a byte-input CNN-downsample student. Existing public syvai/claes-preview checkpoint is 322M, not under 1M. In first 3,000 live rows: 32 languages and 1,536 choice cases.";
    const proposed = "Claes training-design investigation (Oct 2026): current repo has no student model/training/evaluation. The earlier tiny byte/CNN/Transformer student recommendation is superseded. Existing public syvai/claes-preview checkpoint is 322M and fits under <1B.";
    const instruction = "Correct the Claes training-design note recorded Oct 2026: actual limit <1 BILLION, not <1 million. The earlier tiny byte/CNN/Transformer student recommendation was conditional on the mistaken limit and is superseded. Under <1B, the existing 322M syvai/claes-preview checkpoint fits. Retain the accurate dataset schema and sample stats.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", title: "Claes Jev Diagnostics", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: proposed } })
      .mockResolvedValueOnce({ object: { new_text: proposed } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Original counts and no-student claim retained." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, `${JSON.stringify(result)}; calls=${generateObjectMock.mock.calls.length}`).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain("In first 3,000 live rows: 32 languages and 1,536 choice cases.");
    expect(body).toContain("no student model/training/evaluation");
    expect(body).not.toContain("For strict <1M multilingual student");
  });

  it("rejects a Claes training correction when the stale paragraph has two owners", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const duplicate = "/projects/claes-jev-training-copy.md";
    const oldParagraph = "Claes training-design investigation (Oct 2026): For strict <1M multilingual student recommend evaluating a shared byte-input CNN-downsample student. Existing public syvai/claes-preview HF checkpoint is 322M, not under 1M. The dataset has 3,000 sampled rows.";
    const newParagraph = "Claes training-design investigation (Oct 2026): The actual limit is <1B rather than <1M; the tiny byte/CNN/Transformer student recommendation is superseded. Existing public syvai/claes-preview HF checkpoint is 322M and fits under <1B. The dataset has 3,000 sampled rows.";
    const instruction = "Correct the Claes training-design note recorded Oct 2026: actual limit is <1 BILLION, not <1 million; the earlier tiny byte/CNN/Transformer student recommendation is superseded. The 322M checkpoint fits under <1B. Retain the sample stats.";
    const competingClaim = "Claes training: for a <1M model, recommend byte-input CNN-downsample student for Jev distributions.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    await kb.writeConcept(duplicate, { type: "Project", title: "Claes Jev Training Copy" }, competingClaim, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, duplicate].map((path) =>
      ({ path, type: "Project", title: "Claes Jev Diagnostics", score: 10 })));
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect((await kb.readConcept(duplicate)).body).toContain(competingClaim);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("rejects a Claes training correction that reverses a no-student claim", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const oldParagraph = "Claes training-design investigation (Oct 2026): current repo has no student model or training. For strict <1M multilingual student recommend evaluating a byte-input CNN-downsample student. Existing public syvai/claes-preview checkpoint is 322M, not under 1M. The sample stats are retained.";
    const newParagraph = "Claes training-design investigation (Oct 2026): current repo has student model or training. The actual limit is <1B rather than <1M, and the earlier tiny byte/CNN/Transformer student recommendation is superseded. Existing public syvai/claes-preview checkpoint is 322M and fits under <1B. The sample stats are retained.";
    const instruction = "Correct the Claes training-design note recorded Oct 2026: actual limit is <1 BILLION, not <1 million; the earlier tiny byte/CNN/Transformer student recommendation is superseded. The 322M checkpoint fits under <1B. Retain the sample stats.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", title: "Claes Jev Diagnostics", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain("no student model or training");
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("rejects a Claes training correction that drops unrelated dataset history", async () => {
    staged();
    const owner = "/projects/claes-jev-diagnostics.md";
    const oldParagraph = "Claes training-design investigation (Oct 2026): first 3,000 rows contain 1,536 choice and 447 score. Existing public syvai/claes-preview is 322M, not under 1M. For strict <1M multilingual student recommend evaluating a shared byte-input CNN-downsample student. Historical sample audit found one choice selected ID not probability argmax.";
    const newParagraph = "Claes training-design investigation (Oct 2026): Existing public syvai/claes-preview is 322M and fits under 1B. Recommend fine-tuning it rather than a tiny student.";
    const instruction = "Correct the Claes training-design note recorded Oct 2026: actual limit <1 BILLION, not <1 million; the 322M syvai/claes-preview fits <1B and tiny student is superseded. Retain the accurate sample stats.";
    await kb.writeConcept(owner, { type: "Project", title: "Claes Jev Diagnostics" }, oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Project", title: "Claes Jev Diagnostics", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain("one choice selected ID not probability argmax");
  });

  it("corrects multiple clauses only inside a uniquely named incident paragraph", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const oldParagraph = `${heading} One MultiBFCL add recovered; nine sampled retries succeeded; queue canaries were paused.`;
    const newParagraph = `${heading} One MultiBFCL add recovered; nine sampled retries failed; queue canaries were paused and resumed.`;
    const instruction = `Update the Understory memory_add/memory_update incident note (Oct 15 2026): One MultiBFCL add recovered; nine sampled retries failed; queue canaries were paused and resumed.`;
    const unrelated = "Other agent facts. Their independent history remains intact.";
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      `${oldParagraph}\n\n${unrelated}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: newParagraph } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Only the named paragraph changed." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(newParagraph);
    expect(body).toContain(unrelated);
    expect(body).not.toContain("nine sampled retries succeeded");
  });

  it("repairs a broad named-incident rewrite by preserving the old paragraph", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const oldParagraph = `${heading} One MultiBFCL add recovered; nine sampled retries succeeded; queue canaries were paused.`;
    const newParagraph = `${heading} One MultiBFCL add recovered; nine sampled retries failed; queue canaries were paused and resumed.`;
    const instruction = "Update the Understory memory_add/memory_update incident note (Oct 15 2026): One MultiBFCL add recovered; nine sampled retries failed; queue canaries were paused and resumed.";
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      `${oldParagraph}\n\nAn unrelated paragraph stays unchanged.`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Nine sampled retries failed; queue canaries were paused and resumed." } })
      .mockResolvedValueOnce({ object: { new_text: newParagraph } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "History retained." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(newParagraph);
    expect((await kb.readConcept(owner)).body).toContain("An unrelated paragraph stays unchanged.");
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("rejects an unrepairable broad named-incident rewrite without writing", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const oldParagraph = `${heading} One MultiBFCL add recovered; nine sampled retries succeeded; queue canaries were paused.`;
    const instruction = "Update the Understory memory_add/memory_update incident note (Oct 15 2026): nine sampled retries failed.";
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Nine sampled retries failed." } })
      .mockResolvedValueOnce({ object: { new_text: "" } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("appends an exact dated incident follow-up only after independent full-body approval", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const oldParagraph = `${heading} Initial metadata owner-selection fix shipped; historical failures were not requeued.`;
    const followUp = "Later sampled retries failed; the queue resumed after one-job canaries.";
    const instruction = `Update the Understory memory_add/memory_update incident note (Oct 15 2026): ${followUp}`;
    const unrelated = "Other deployment history remains intact.";
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      `${oldParagraph}\n\n${unrelated}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: followUp } })
      .mockResolvedValueOnce({ object: { new_text: followUp } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "No contradictory or duplicate claim." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toContain(`${oldParagraph}\n\n${followUp}\n\n${unrelated}`);
  });

  it("rejects a dated follow-up that contradicts the existing incident", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const oldParagraph = `${heading} All metadata rejections were permanently fixed.`;
    const followUp = "Nine metadata retries still fail after the fix.";
    const instruction = `Update the Understory memory_add/memory_update incident note (Oct 15 2026): ${followUp}`;
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      oldParagraph, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: followUp } })
      .mockResolvedValueOnce({ object: { new_text: followUp } })
      .mockResolvedValueOnce({ object: { safe: false, reason: "Old assertion contradicts new status." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("follow-up failed independent") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
  });

  it("does not broaden a named-section correction to an unrelated paragraph", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const unrelated = "Nine sampled retries succeeded; other separate events succeeded.";
    const instruction = "Update the Understory memory_add/memory_update incident note (Oct 15 2026): nine sampled retries failed; other separate events failed.";
    await kb.writeConcept(owner, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
      `${heading} An independent incident occurred.\n\n${unrelated}`, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([{ path: owner, type: "Repository", score: 10 }]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: unrelated,
      new_text: "Nine sampled retries failed; other separate events failed." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(unrelated);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it.each([false, true])("locates a dated FS status across all siblings and rejects a duplicate (%s)", async (duplicate) => {
    staged();
    const owner = "/projects/fs-fotovalidering-continuation-f5bd33e9.md";
    const copy = "/projects/fs-fotovalidering-continuation-copy.md";
    const heading = "FS Fotovalidering owner-reviewed v3 retrain (13 Oct 2026):";
    const instruction = "Update FS Fotovalidering owner-reviewed v3 retrain status recorded on 13 Oct 2026: a fresh 24-head Sparkie training run completed; the earlier job remains historical.";
    await kb.writeConcept(owner, { type: "Project Continuation", title: "FS Continuation" },
      `${heading} Mac MPS was interrupted; fresh Sparkie training still pending.`, "seed");
    await kb.writeConcept(copy, { type: "Project Continuation", title: "Another Continuation" },
      duplicate ? `${heading} A copied status note.` : "Separate observations.", "seed");
    const competing = ["/projects/other-a.md", "/projects/other-b.md"];
    for (const candidate of competing) {
      await kb.writeConcept(candidate, { type: "Project Continuation", title: "Other Project" },
        "Unrelated observations. ".repeat(1_600), "seed");
    }
    // Ranked results omit the actual dated owner and the non-ranked duplicate.
    vi.spyOn(kb, "search").mockResolvedValue(competing.map((conceptPath) => ({
      path: conceptPath, type: "Project Continuation", title: "Other Project", score: 10,
      confidenceQualified: false,
    })));
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`"path":"${owner}"`);
      return { object: { ...baseProposal, action: "defer", reason: "dated status owner located" } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining(
      duplicate ? "named section is not unique" : "dated status owner located") });
    expect(generateObjectMock).toHaveBeenCalledTimes(duplicate ? 0 : 1);
    expect((await kb.readConcept(owner)).body).toContain("still pending");
  });

  it("rejects an incident section that appears in two complete owners", async () => {
    staged();
    const owner = "/repos/pi-agent/memory-async.md";
    const competitor = "/repos/pi-agent/incident-copy.md";
    const unrelated = [1, 2, 3, 4, 5].map((n) => `/repos/unrelated-${n}.md`);
    const heading = "Understory memory_add/memory_update incident (Oct 15 2026):";
    const instruction = "Update the Understory memory_add/memory_update incident note (Oct 15 2026): nine retries failed.";
    for (const conceptPath of unrelated) {
      await kb.writeConcept(conceptPath, { type: "Repository", title: "Unrelated" }, "Other work.", "seed");
    }
    for (const conceptPath of [owner, competitor]) {
      const label = conceptPath === owner ? heading : `**${heading.slice(0, -1)}:**`;
      await kb.writeConcept(conceptPath, { type: "Repository", title: "Pi Agent — Memory-Async Extension" },
        `${label} Nine retries succeeded.`, "seed");
    }
    vi.spyOn(kb, "search").mockImplementation(async (query) =>
      query.includes("Understory memory_add/memory_update incident")
        ? [owner, competitor].map((conceptPath) => ({ path: conceptPath, type: "Repository", score: 5 }))
        : unrelated.map((conceptPath) => ({ path: conceptPath, type: "Repository", score: 10 })));
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("named section is not unique") });
    expect(generateObjectMock).not.toHaveBeenCalled();
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

  it.each([false, true])("locates a unique FS keyboard note beyond ranked page hits (duplicate=%s)", async (duplicate) => {
    staged();
    const ordering = "/projects/fs-fotovalidering-continuation-e0d0ea92dc7fde84.md";
    const keyboard = "/projects/fs-fotovalidering-continuation-8559c6035bedff77.md";
    const hiddenPeer = "/projects/fs-fotovalidering-continuation-hidden.md";
    const oldOrdering = "FS Fotovalidering 25-class annotation buttons are sorted by config/categories.yaml priority. " +
      "The running /tmp/fs-photo-candidates-25.html page on port 8772 keeps the same localStorage key.";
    const controls = "FS Fotovalidering 25-class annotation UI preference: keyboard-first controls with keys 1–9 mapped in rank order. " +
      "Enter saves and advances. The /tmp/fs-photo-candidates-25.html page runs on port 8772; localStorage is preserved.";
    const instruction = "Correct the FS Fotovalidering 25-class annotation UI knowledge: controls are binary selected/unselected. " +
      "Number keys 1–9 select categories. The running /tmp/fs-photo-candidates-25.html page on port 8772 " +
      "was updated in place; localStorage key unchanged.";
    await kb.writeConcept(ordering, { type: "Project Continuation", title: "FS Fotovalidering — Phase 1 PoC — Continuation e0d0ea92" },
      `${oldOrdering} ${"Documented ordering history. ".repeat(1_200)}`, "seed");
    await kb.writeConcept(keyboard, { type: "Project Continuation", title: "FS Fotovalidering — Keyboard Notes" },
      controls, "seed");
    await kb.writeConcept(hiddenPeer, { type: "Project Continuation", title: "FS Fotovalidering — Later Review" },
      duplicate ? controls : "Reviewed unknown labels without keyboard controls.", "seed");
    // The top hit is the page-ordering note; the matching keyboard note is
    // absent from ranked search, and a duplicate can also be absent from it.
    vi.spyOn(kb, "search").mockResolvedValue([{ path: ordering, type: "Project Continuation",
      title: "FS Fotovalidering — Phase 1 PoC — Continuation e0d0ea92", score: 10, confidenceQualified: false }]);
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(controls);
      expect(prompt).not.toContain(oldOrdering);
      return { object: { ...baseProposal, action: "defer", reason: "keyboard owner found" } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining(duplicate
      ? "keyboard-controls owner is missing or duplicated" : "keyboard owner found") });
    expect(generateObjectMock).toHaveBeenCalledTimes(duplicate ? 0 : 1);
    expect((await kb.readConcept(ordering)).body).toContain(oldOrdering);
  });

  it.each([false, true])("locates the dated zero-row workbook owner beyond ranked retirement hits (duplicate=%s)", async (duplicate) => {
    staged();
    const workbook = "/projects/fs-fotovalidering-continuation-21b944fc45cea087.md";
    const retirement = "/projects/fs-fotovalidering-continuation-2443b1fe8100f1af.md";
    const hiddenPeer = "/projects/fs-fotovalidering-continuation-other.md";
    const oldList = "FS Fotovalidering 2026-10-12 Jottacloud source workbook: across the workbook sheets, " +
      "exactly six rejection-reason rows have zero `x` marks: Motiv 5, Briller 3, " +
      "Hovedbeklædning 2, Hovedbeklædning 3, Hovedbeklædning 4, and Anstødeligt indhold. " +
      "Separate approval-status row 'Godkendt med beskæring' also has zero marks but is not a rejection category.";
    const later = "FS Fotovalidering category retirement: Hovedbeklædning 4 / headwear_face_shadow " +
      "was removed from the active workbook; canonical-24 is active and experiment-88 remains 25-head.";
    const instruction = "Update FS Fotovalidering facts after the 2026-10-12 Hovedbeklædning 4 retirement: " +
      "active registry is canonical-24 and experiment-88 stays 25-head. " +
      "Correct the earlier point-in-time 2026-10-12 workbook zero-annotation list: Hovedbeklædning 4 was removed " +
      "from active source, not merely left unannotated. Keep 'Godkendt med beskæring' untouched.";
    await kb.writeConcept(workbook, { type: "Project Continuation", title: "FS Fotovalidering — Workbook Notes" }, oldList, "seed");
    await kb.writeConcept(retirement, { type: "Project Continuation", title: "FS Fotovalidering — Retirement" }, later, "seed");
    await kb.writeConcept(hiddenPeer, { type: "Project Continuation", title: "FS Fotovalidering — Other" },
      duplicate ? oldList : "An unrelated annotation experiment on a different workbook.", "seed");
    // A later retirement owner can outrank the actual historical paragraph;
    // a duplicate historical paragraph hidden from search must still veto.
    vi.spyOn(kb, "search").mockResolvedValue([{ path: retirement, type: "Project Continuation",
      title: "FS Fotovalidering — Retirement", score: 10, confidenceQualified: false }]);
    generateObjectMock.mockImplementation(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(oldList);
      expect(prompt).not.toContain(later);
      return { object: { ...baseProposal, action: "defer", reason: "dated workbook owner found" } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: false, status: "failed", error: expect.stringContaining(duplicate
      ? "dated zero-row workbook owner is missing or duplicated" : "dated workbook owner found") });
    expect(generateObjectMock).toHaveBeenCalledTimes(duplicate ? 0 : 1);
    expect((await kb.readConcept(workbook)).body).toContain(oldList);
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

  it("repairs an unsafe correction create without making a duplicate concept", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "create",
      path: "/apis/billing-approval.md", body: updateInstruction,
      frontmatter: { type: "API Endpoint", title: "Billing Approval", description: "Approval." } } })
      .mockResolvedValueOnce({ object: { path: "/apis/billing-api.md",
        old_text: "ad-hoc charges come from support tooling.",
        new_text: "ad-hoc charges come from support tooling only after approval." } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "verified" } });
    expect(await runMutation(kb, updateInstruction, { preflightInput: updateInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect(await kb.bundle.exists("/apis/billing-approval.md")).toBe(false);
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain("only after approval");
  });

  it("repairs an unsafe correction append as one exact verified replacement", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "append",
      path: "/apis/billing-api.md", new_text: "Support tooling requires approval." } })
      .mockResolvedValueOnce({ object: { path: "/apis/billing-api.md",
        old_text: "ad-hoc charges come from support tooling.",
        new_text: "ad-hoc charges come from support tooling only after approval." } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "verified" } });
    const result = await runMutation(kb, updateInstruction, { preflightInput: updateInstruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    const body = (await kb.readConcept("/apis/billing-api.md")).body;
    expect(body).toContain("only after approval");
    expect(body).not.toContain("ad-hoc charges come from support tooling.");
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("verifies a bounded paragraph correction at an explicitly named existing owner", async () => {
    await billingFixture(); staged();
    const instruction = "In `/apis/billing-api.md`, correct the Billing API: monthly charges are scheduled only after approval; ad-hoc charges come from support tooling only after approval.";
    const corrected = "Monthly charges are scheduled only after approval; ad-hoc charges come from support tooling only after approval.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace",
      path: "/apis/billing-api.md", old_text: oldBody, new_text: corrected } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "verified" } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(corrected);
  });

  it("allows a bounded multi-clause correction repair only after independent approval", async () => {
    await billingFixture(); staged();
    const instruction = "Correct Billing API: monthly charges are scheduled only after approval; ad-hoc charges come from support tooling only after approval.";
    const corrected = "Monthly charges are scheduled only after approval; ad-hoc charges come from support tooling only after approval.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "append",
      path: "/apis/billing-api.md", new_text: corrected } })
      .mockResolvedValueOnce({ object: { path: "/apis/billing-api.md", old_text: oldBody,
        new_text: corrected } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "verified" } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: ["/apis/billing-api.md"] } });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toContain(corrected);
  });

  it("repairs several job-bot findings inside one uniquely owned paragraph without rolling back later status", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot security findings: a site-wide provider spending ceiling is an open recommendation; " +
      "successful cover letters were fully logged; Selenium port 4444 is exposed in production; " +
      "Auth0 subjects remain isolated for account rate limits.";
    const later = "Later hardening: local Selenium mapping was removed and Tika runs in a bounded subprocess.";
    const corrected = "Job-bot security findings: Dan does not want a site-wide provider spending ceiling; " +
      "only malformed cover-letter output was included in HTTP 422 and was sanitised at 08a090c; " +
      "Selenium port 4444 was not exposed in production; Auth0 subjects remain isolated for account rate limits.";
    const instruction = "Update job-bot security findings: remove the open recommendation for a site-wide provider spending ceiling, Dan does not want one; " +
      "correct the old claim that successful cover letters were fully logged: only malformed cover-letter output was included in HTTP 422 and was sanitised at 08a090c; " +
      "Selenium port 4444 was not exposed in production. Preserve later Tika hardening.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" },
      `${oldParagraph}\n\n${later}`, "seed");
    await kb.writeConcept("/users/dan-smart.md", { type: "Knowledge", title: "Dan Smart" },
      "Dan does not want a site-wide provider spending ceiling for job-bot.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([
      { path: owner, type: "Repository", title: "Job-bot", score: 100 },
      { path: "/users/dan-smart.md", type: "Person", title: "Dan Smart", score: 90 },
    ]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: corrected } })
      .mockResolvedValueOnce({ object: { new_text: corrected.replace(" Auth0 subjects remain isolated for account rate limits.", "") } })
      .mockResolvedValueOnce({ object: { new_text: corrected } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Corrections are grounded and later status retained." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, `${JSON.stringify(result)}; calls=${generateObjectMock.mock.calls.map(([request]) => request.prompt.slice(0, 75)).join(" | ")}`).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toBe(`${corrected}\n\n${later}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("repairs a bounded stale span inside a long job-bot audit without rewriting its history", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldSpan = "Job-bot security findings: a site-wide provider spending ceiling is recommended; " +
      "successful cover letters were fully logged; Selenium port 4444 is exposed in production.";
    const newSpan = "Job-bot security findings: Dan does not want a site-wide provider spending ceiling; " +
      "only malformed cover-letter output was included in HTTP 422; Selenium port 4444 was not exposed in production.";
    const history = Array.from({ length: 10 }, (_, index) =>
      `Historical audit checkpoint ${index + 1} retains the Auth0 rate-limit reservation and CV upload budget.`).join(" ");
    const paragraph = `${oldSpan} ${history}`;
    const instruction = "Update job-bot security findings: remove the site-wide provider spending ceiling recommendation because Dan does not want it; " +
      "correct the claim that successful cover letters were fully logged: only malformed cover-letter output was included in HTTP 422; " +
      "Selenium port 4444 is not exposed in production.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" }, paragraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldSpan, new_text: newSpan } })
      .mockResolvedValueOnce({ object: { new_text: paragraph } })
      .mockResolvedValueOnce({ object: { old_text: oldSpan, new_text: newSpan } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Only the stale span changes." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toBe(`${newSpan} ${history}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("does not mark a multi-finding update complete after correcting only its first clause", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot security audit: no site-wide provider spending ceiling was added; " +
      "multiple accounts can exhaust provider budgets and Tika parsing was initially unbounded.";
    const instruction = "Update job-bot security findings: Dan does NOT want a site-wide provider spending ceiling. " +
      "Ignore the optional Accelerate advisory. Malformed cover-letter HTTP 422 output was sanitised at 08a090c " +
      "and deployed at de3eedf; keep later Selenium and Tika hardening.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" }, oldParagraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: "no site-wide provider spending ceiling was added",
      new_text: "Dan does NOT want a site-wide provider spending ceiling" } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Incorrectly claimed completeness." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed",
        error: "Staged mutation deferred: replacement omits requested correction identifiers." });
    expect((await kb.readConcept(owner)).body).toBe(`${oldParagraph}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("qualifies a true historical audit instead of erasing it when later mitigation exists", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot audit (Oct 2026): No site-wide provider spending ceiling was added; " +
      "multiple accounts could exhaust budgets; Tika could consume resources before hard limits were added.";
    const later = "Later hardening: the Tika parser runs in a subprocess with a 30s deadline, and production Selenium port 4444 has no host mapping.";
    const instruction = "Update job-bot security findings: Dan does not want a site-wide provider spending ceiling; " +
      "successful cover letters were not fully logged, only malformed output in HTTP 422 was sanitised; " +
      "production Selenium port 4444 is not host-exposed; Tika no longer lacks a hard timeout after later hardening.";
    const addition = "Job-bot security follow-up: Dan does not want a site-wide provider spending ceiling. " +
      "Successful cover letters were not fully logged: only malformed output in HTTP 422 was sanitised. " +
      "The later Tika 30s deadline and production Selenium port 4444 host-mapping fix close those older findings.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" }, `${oldParagraph}\n\n${later}`, "seed");
    const instructionWithFacts = `${instruction} Verified current summary: ${addition}`;
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot security findings are closed." } })
      .mockResolvedValueOnce({ object: { new_text: oldParagraph } })
      .mockResolvedValueOnce({ object: { qualifiable: true, certain: true, addition } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Old audit remains historical; later fixes remain current." } });
    const result = await runMutation(kb, instructionWithFacts, { preflightInput: instructionWithFacts });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(owner)).body).toBe(`${oldParagraph}\n\n${later}\n\n${addition}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it.each([
    "Only docker-compose-local.yaml maps 0.0.0.0:4444 for the scrape profile.",
    "Tika CV parsing still has no true hard timeout despite the earlier audit.",
  ])("rejects a stale technical follow-up despite a model approval (%s)", async (stale) => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot audit: no provider spending ceiling was added, so several accounts could exhaust budgets; " +
      "Tika could consume resources despite upload and page limits.";
    const later = "Local-only browser.ports removed from docker-compose-local.yaml; " +
      "asyncio.wait now enforces 30s deadline for Tika and kills the parser process group.";
    const instruction = "Update job-bot security findings: Dan does not want a spending ceiling; verify production port exposure and the current Tika deadline.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" }, `${oldParagraph}\n\n${later}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot security audit updated." } })
      .mockResolvedValueOnce({ object: { new_text: oldParagraph } })
      .mockResolvedValueOnce({ object: { qualifiable: true, certain: true,
        addition: `Dan does not want a provider spending ceiling. ${stale}` } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "A mistaken model approval cannot override later status." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toBe(`${oldParagraph}\n\n${later}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("rejects an incomplete multi-finding follow-up even when the model claims it is sufficient", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot security audit: no provider spending ceiling was added and multiple accounts can exhaust budgets; " +
      "Tika was initially bounded only by upload and page limits.";
    const instruction = "Update job-bot security findings: Dan does NOT want a site-wide provider spending ceiling; " +
      "ignore optional Accelerate advisory; malformed HTTP 422 output was sanitised at 08a090c and deployed de3eedf.";
    const incomplete = "Dan does not want a site-wide provider spending ceiling for job-bot.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" }, oldParagraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot findings are closed." } })
      .mockResolvedValueOnce({ object: { new_text: oldParagraph } })
      .mockResolvedValueOnce({ object: { qualifiable: true, certain: true, addition: incomplete } })
      .mockResolvedValueOnce({ object: { new_text: incomplete } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Should never approve omitted findings." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toBe(`${oldParagraph}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("rejects a follow-up when a current false assertion must instead be retracted", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot security findings: successful cover letters are fully logged on every request; " +
      "Selenium port 4444 is exposed in production and the audit marks both as current vulnerabilities.";
    const instruction = "Update job-bot security findings: successful cover letters were not fully logged; " +
      "only malformed HTTP 422 returned model output, and production Selenium port 4444 was not exposed.";
    await kb.writeConcept(owner, { type: "Application", title: "Job-bot" }, oldParagraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot security findings were corrected." } })
      .mockResolvedValueOnce({ object: { new_text: oldParagraph } })
      .mockResolvedValueOnce({ object: { qualifiable: true, certain: true,
        addition: "Successful cover letters were not fully logged; production Selenium port 4444 was not exposed." } })
      .mockResolvedValueOnce({ object: { new_text: "Successful cover letters were not fully logged; malformed HTTP 422 exposed output; production Selenium port 4444 was not exposed." } })
      .mockResolvedValueOnce({ object: { safe: false, reason: "Active false assertions remain." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toBe(`${oldParagraph}\n`);
    expect(generateObjectMock).toHaveBeenCalledTimes(5);
  });

  it("blocks an unscoped multi-clause correction when the paragraph has another owner", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const peer = "/repos/job-bot-security.md";
    const oldParagraph = "Job-bot security findings: a spending ceiling is recommended; " +
      "successful cover letters were fully logged; Selenium port 4444 is exposed in production.";
    for (const conceptPath of [owner, peer]) {
      await kb.writeConcept(conceptPath, { type: "Repository", title: conceptPath === owner ? "Job-bot" : "Job-bot security" },
        oldParagraph, "seed");
    }
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((conceptPath, i) => ({
      path: conceptPath, type: "Repository", title: i ? "Job-bot security" : "Job-bot", score: 100 - i,
    })));
    const instruction = "Update job-bot security findings: no spending ceiling; cover-letter logging was only a malformed HTTP 422; production Selenium is not exposed.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot security findings: no spending ceiling; only malformed HTTP 422 exposed model output; Selenium port 4444 is not exposed in production." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("blocks a paraphrased competing owner before a multi-clause paragraph repair", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const peer = "/repos/job-bot-security.md";
    const oldParagraph = "Job-bot security findings: a site-wide spending cap is recommended; " +
      "successful cover letters were fully logged; Selenium port 4444 is exposed in production.";
    await kb.writeConcept(owner, { type: "Repository", title: "Job-bot" }, oldParagraph, "seed");
    await kb.writeConcept(peer, { type: "Security", title: "Security Review" },
      "The job-bot audit recommends a global provider spend limit because the model-generated letters are logged.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((conceptPath, i) => ({
      path: conceptPath, type: i ? "Security" : "Repository", title: i ? "Security Review" : "Job-bot", score: 100 - i,
    })));
    const instruction = "Update job-bot security findings: remove the spending cap recommendation, correct the successful letter logging, and correct production Selenium port 4444 exposure.";
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot security findings: no spending cap wanted; only malformed letter output was exposed; production Selenium port 4444 is not exposed." } })
      .mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
        expect(prompt).toContain("global provider spend limit");
        return { object: { sameClaimOwner: true, certain: true } };
      });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("keeps related staging authentication when correcting production Selenium exposure", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot security findings: production Selenium port 4444 was exposed; " +
      "Selenium port 4444 staging authentication remains mandatory; provider spending cap is recommended.";
    const instruction = "Update job-bot security findings: production Selenium port 4444 was not host-exposed; Dan does not want the provider spending cap recommendation.";
    const unsafe = "Job-bot security findings: production Selenium port 4444 was not host-exposed; " +
      "provider spending cap is no longer recommended.";
    await kb.writeConcept(owner, { type: "Repository", title: "Job-bot" }, oldParagraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: unsafe } })
      .mockResolvedValueOnce({ object: { new_text: unsafe } })
      .mockResolvedValueOnce({ object: { new_text: unsafe } })
      .mockResolvedValueOnce({ object: { qualifiable: false, certain: false, addition: "" } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Should never approve lost staging assertion." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain("staging authentication remains mandatory");
    expect(generateObjectMock).toHaveBeenCalledTimes(4);
  });

  it("refuses a job-bot paragraph repair that removes unrelated claims even if the verifier approves", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const oldParagraph = "Job-bot security findings: a spending ceiling is recommended; " +
      "successful cover letters were fully logged; Auth0 subjects remain isolated for account rate limits.";
    const instruction = "Update job-bot security findings: Dan does not want a spending ceiling; only malformed cover-letter output was sent in HTTP 422.";
    await kb.writeConcept(owner, { type: "Repository", title: "Job-bot" }, oldParagraph, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: oldParagraph, new_text: "Job-bot security findings: Dan does not want a spending ceiling; only malformed cover-letter output was sent in HTTP 422." } })
      .mockResolvedValueOnce({ object: { new_text: "Job-bot security findings: Dan does not want a spending ceiling; only malformed cover-letter output was sent in HTTP 422." } })
      .mockResolvedValueOnce({ object: { new_text: "Job-bot security findings: Dan does not want a spending ceiling; only malformed cover-letter output was sent in HTTP 422." } })
      .mockResolvedValueOnce({ object: { supersededClauses: ["Auth0 subjects remain isolated for account rate limits"], certain: true } })
      .mockResolvedValueOnce({ object: { qualifiable: false, certain: false, addition: "" } })
      .mockResolvedValueOnce({ object: { safe: true, reason: "Not sufficient to remove Auth0." } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("change spans clauses") });
    expect((await kb.readConcept(owner)).body).toContain(oldParagraph);
    expect(generateObjectMock).toHaveBeenCalledTimes(5);
  });

  it("refuses a correction append when independent replacement verification fails", async () => {
    await billingFixture(); staged();
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "append",
      path: "/apis/billing-api.md", new_text: "Support tooling requires approval." } })
      .mockResolvedValueOnce({ object: { path: "/apis/billing-api.md",
        old_text: "ad-hoc charges come from support tooling.",
        new_text: "ad-hoc charges come from support tooling only after approval." } })
      .mockResolvedValueOnce({ object: { safe: false, reason: "wrong owner" } });
    expect(await runMutation(kb, updateInstruction, { preflightInput: updateInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept("/apis/billing-api.md")).body).toBe(`${oldBody}\n`);
  });

  it("finds a named Mimir correction owner outside the broad top five without bypassing verification", async () => {
    staged();
    const paper = "/papers/dfm-mimir-v1.md";
    const incidental = "/papers/mimir-news.md";
    const unrelated = [1, 2, 3, 4, 5].map((n) => `/projects/unrelated-${n}.md`);
    await kb.writeConcept(incidental, { type: "Paper", title: "Mimir News" },
      "A separate article about Mimir news, not the zero-shot evaluation.", "seed");
    await kb.writeConcept(paper, { type: "Paper", title: "DFM Mimir v1" },
      "Mimir validation-zero-shot-by-language.csv and validation-zero-shot-mimir-matched-by-language.csv reports exist.", "seed");
    for (const [i, owner] of unrelated.entries()) {
      await kb.writeConcept(owner, { type: "Project", title: `Unrelated ${i + 1}` },
        `Unrelated project ${i + 1} notes.`, "seed");
    }
    const search = vi.spyOn(kb, "search").mockImplementation(async (query) => {
      if (query === "mimir") return [
        { path: incidental, type: "Paper", title: "Mimir News", score: 11, confidenceQualified: false },
        { path: paper, type: "Paper", title: "DFM Mimir v1", score: 10, confidenceQualified: false },
      ];
      if (query === "claes") return [];
      return unrelated.map((owner, i) => ({ path: owner, type: "Project", title: `Unrelated ${i + 1}`,
        score: 10, confidenceQualified: false }));
    });
    const instruction = "Correct Claes zero-shot evaluation: Mimir validation-zero-shot-by-language.csv and validation-zero-shot-mimir-matched-by-language.csv reports were removed, not retained.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`"path":"${paper}"`);
      expect(prompt).toContain(`"path":"${incidental}"`);
      expect(prompt).toContain(`"path":"${unrelated[0]}"`);
      return { object: { ...baseProposal, action: "replace", path: paper,
        old_text: "reports exist", new_text: "reports were removed" } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`OWNER ${paper}:`);
      expect(prompt).toContain(unrelated[4]);
      return { object: { safe: false, reason: "uncertain" } };
    });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(search.mock.calls.map(([query]) => query)).toContain("mimir");
    expect((await kb.readConcept(paper)).body).toContain("reports exist");
  });

  it("selects from metadata before a focused correction overfills the proposer", async () => {
    staged();
    const paper = "/papers/dfm-mimir-v1.md";
    const other = [1, 2, 3].map((n) => `/projects/unrelated-${n}.md`);
    await kb.writeConcept(paper, { type: "Paper", title: "DFM Mimir v1" },
      `Mimir CSV reports exist. ${"Mimir checkpoint context. ".repeat(200)}`, "seed");
    for (const [i, owner] of other.entries()) {
      await kb.writeConcept(owner, { type: "Project", title: `Unrelated ${i + 1}` },
        `Unrelated context ${i + 1}. `.repeat(400), "seed");
    }
    vi.spyOn(kb, "search").mockImplementation(async (query) =>
      query === "mimir" ? [{ path: paper, type: "Paper", title: "DFM Mimir v1",
        score: 10, confidenceQualified: false }]
        : other.map((owner, i) => ({ path: owner, type: "Project", title: `Unrelated ${i + 1}`,
          score: 10, confidenceQualified: false })));
    const instruction = "Correct the Mimir evaluation: CSV reports were removed rather than retained.";
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("CANDIDATE METADATA");
      expect(prompt).toContain(`"path":"${paper}"`);
      expect(prompt).toContain(`"path":"${other[2]}"`);
      return { object: { path: paper } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`"path":"${paper}"`);
      expect(prompt).not.toContain(`"path":"${other[0]}"`);
      return { object: { ...baseProposal, action: "defer", reason: "Test only." } };
    });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(paper)).body).toContain("reports exist");
  });

  it("retains the original third owner when a focused name is incidental", async () => {
    staged();
    const original = [1, 2, 3, 4, 5].map((n) => `/projects/project-${n}.md`);
    const incidental = "/projects/hooli.md";
    for (const [i, owner] of original.entries()) {
      await kb.writeConcept(owner, { type: "Project", title: `Project ${i + 1}` },
        `Project ${i + 1} records a distinct claim.`, "seed");
    }
    await kb.writeConcept(incidental, { type: "Project", title: "Hooli" },
      "Hooli is unrelated to the correction.", "seed");
    vi.spyOn(kb, "search").mockImplementation(async (query) =>
      query === "hooli"
        ? [{ path: incidental, type: "Project", title: "Hooli", score: 10, confidenceQualified: false }]
        : original.map((owner, i) => ({ path: owner, type: "Project", title: `Project ${i + 1}`,
          score: 10, confidenceQualified: false })));
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`"path":"${original[2]}"`);
      expect(prompt).toContain(`"path":"${incidental}"`);
      return { object: { ...baseProposal, action: "defer", reason: "Test only." } };
    });
    expect(await runMutation(kb, "Correct the project claim; Hooli is unrelated to this correction.", {
      preflightInput: "Correct the project claim; Hooli is unrelated to this correction.",
    })).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(original[2])).body).toContain("distinct claim");
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

  it("repairs a duplicate old operand using its unique stale-claim context", async () => {
    staged();
    const owner = "/projects/cover-letter.md";
    const stale = "The cover letter uses the legacy JSON schema.";
    const paragraph = `${stale} The applicant's research experience remains relevant.`;
    const retained = "The resume uses the legacy JSON schema.";
    const instruction = `Correct the cover letter schema in \`${owner}\`: replace the legacy JSON schema with the current YAML schema. Preserve the research experience and the resume's legacy JSON schema.`;
    await kb.writeConcept(owner, { type: "Project", title: "Cover Letter" }, `${paragraph}\n\n${retained}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: "legacy JSON schema",
      new_text: "The cover letter uses the current YAML schema.",
    } }).mockResolvedValueOnce({ object: {
      index: 0,
      new_text: "The cover letter uses the current YAML schema. The applicant's research experience remains relevant.",
    } }).mockResolvedValueOnce({ object: { safe: true, reason: "Only the stale cover-letter schema claim changed." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain("current YAML schema");
    expect(body).toContain(retained);
    expect(body).toContain("applicant's research experience remains relevant");
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("grounds an invented correction operand in one exact identifier-matched paragraph", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const old = "The cover_letters table has an account email FK ON DELETE CASCADE. Keep historic migration notes.";
    const other = "The cover_letter_owners table has a separate account link.";
    const instruction = `Correct Job-bot cover_letters: it no longer cascade-deletes with accounts; cover_letter_owners owns the deletable account link. Preserve historic migration notes.`;
    await kb.writeConcept(owner, { type: "Repository", title: "Job Bot" }, `${old}\n\n${other}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: "cover_letters rows cascade-delete with accounts",
      new_text: "The cover_letters table no longer cascade-deletes with accounts.",
    } }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`\"text\":\"${old}\"`);
      expect(prompt).toContain(`\"text\":\"${other}`);
      return { object: { index: 0,
        new_text: "The cover_letters table no longer cascade-deletes with accounts; cover_letter_owners owns the deletable account link. Keep historic migration notes." } };
    }).mockResolvedValueOnce({ object: { safe: true, reason: "Supported correction and unrelated history retained." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain("no longer cascade-deletes");
    expect(body).toContain("Keep historic migration notes.");
    expect(body).toContain(other);
    expect(body).not.toContain("account email FK ON DELETE CASCADE");
  });

  it("refuses an invented correction operand without grounded matching identifiers", async () => {
    staged();
    const owner = "/repos/job-bot.md";
    const body = "This unrelated account migration has no table-specific schema claims.";
    await kb.writeConcept(owner, { type: "Repository", title: "Job Bot" }, body, "seed");
    const patchSpy = vi.spyOn(kb, "patchConcept");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: "cover_letters cascade-delete with accounts",
      new_text: "cover_letters do not cascade-delete with accounts",
    } });
    expect(await runMutation(kb, "Correct Job-bot cover_letters: do not cascade-delete with accounts."))
      .toMatchObject({ ok: false, status: "failed" });
    expect(patchSpy).not.toHaveBeenCalled();
    expect((await kb.readConcept(owner)).body.trim()).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledTimes(1);
  });

  it("defers when duplicate old operands have ambiguous active claims", async () => {
    staged();
    const owner = "/projects/cover-letter.md";
    const stale = "The cover letter uses the legacy JSON schema.";
    const body = `${stale}\n\n${stale} It is repeated in the active storage specification.`;
    const instruction = `Correct the cover letter schema in \`${owner}\`: replace the legacy JSON schema with the current YAML schema.`;
    await kb.writeConcept(owner, { type: "Project", title: "Cover Letter" }, body, "seed");
    const patchSpy = vi.spyOn(kb, "patchConcept");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: "legacy JSON schema",
      new_text: "The cover letter uses the current YAML schema.",
    } }).mockResolvedValueOnce({ object: { index: -1, new_text: "" } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(patchSpy).not.toHaveBeenCalled();
    expect((await kb.readConcept(owner)).body.trim()).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("rejects a repaired duplicate-operand correction if the owner changes before CAS", async () => {
    staged();
    const owner = "/projects/cover-letter.md";
    const stale = "The cover letter uses the legacy JSON schema.";
    const paragraph = `${stale} The applicant's research experience remains relevant.`;
    const instruction = `Correct the cover letter schema in \`${owner}\`: replace the legacy JSON schema with the current YAML schema. Preserve the resume's legacy JSON schema.`;
    await kb.writeConcept(owner, { type: "Project", title: "Cover Letter" }, `${paragraph}\n\nThe resume uses the legacy JSON schema. Unrelated note.`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: "legacy JSON schema",
      new_text: "The cover letter uses the current YAML schema.",
    } }).mockResolvedValueOnce({ object: {
      index: 0,
      new_text: "The cover letter uses the current YAML schema. The applicant's research experience remains relevant.",
    } }).mockImplementationOnce(async () => {
      await kb.writeConcept(owner, { type: "Project", title: "Cover Letter" }, "External revision.", "external");
      return { object: { safe: true, reason: "Approved." } };
    });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body.trim()).toBe("External revision.");
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

  it("repairs unsupported paraphrase words in a Holi icon correction before independent verification", async () => {
    staged();
    const owner = "/repos/holi.md";
    const oldText = "Option 17's right page has a distinct title squiggle above thin note lines.";
    const instruction = "Correct the Holi app-icon option 17 exploration fact: the right page no longer has a distinct title squiggle. Dan rejected the heavier title stroke and asked for ordinary, evenly weighted, gently curved note lines on the right page. Left page remains a checklist. No final icon selected.";
    await kb.writeConcept(owner, { type: "Repository", title: "Holi" },
      `${oldText} Left page remains a checklist. No final icon selected.`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: oldText,
      new_text: "Option 17's right page has gracefully curved note lines and no distinct title squiggle.",
    } }).mockResolvedValueOnce({ object: {
      new_text: "Option 17's right page has ordinary, evenly weighted, gently curved note lines and no distinct title squiggle.",
    } }).mockImplementationOnce(async (request: { prompt: string }) => {
      expect(request.prompt).toContain("NEW TEXT:\nOption 17's right page has ordinary, evenly weighted");
      expect(request.prompt).not.toContain("NEW TEXT:\nOption 17's right page has gracefully curved");
      return { object: { safe: true, reason: "Local correction supported." } };
    });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain("ordinary, evenly weighted, gently curved note lines");
    expect(body).toContain("Left page remains a checklist. No final icon selected.");
    expect(body).not.toContain("above thin note lines");
    expect(generateObjectMock).toHaveBeenCalledTimes(3);
  });

  it("rejects a repaired Holi icon correction that still adds an unrequested feature", async () => {
    staged();
    const owner = "/repos/holi.md";
    const oldText = "Option 17's right page has a distinct title squiggle above thin note lines.";
    const instruction = "Correct Holi option 17: no distinct title squiggle; use ordinary, evenly weighted, gently curved note lines.";
    await kb.writeConcept(owner, { type: "Repository", title: "Holi" }, oldText, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: {
      action: "replace", path: owner, old_text: oldText,
      new_text: "Option 17's right page has gently curved note lines and a glitter border.",
    } }).mockResolvedValueOnce({ object: {
      new_text: "Option 17's right page has ordinary, gently curved note lines and a gold border.",
    } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("replacement introduces facts absent") });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
    expect((await kb.readConcept(owner)).body.trim()).toBe(oldText);
  });

  const fsDatedBaselineOwner = "/projects/fs-fotovalidering-continuation-257a96e360815542.md";
  const fsDatedBaselinePrior = "FS Fotovalidering baseline 2026-10-14 — Dan completed owner annotation for all 258 test photos. On 14 Oct 2026 it was applied exclusively to the source Excel workbook and parsed/ was regenerated. 258 sample IDs and hashes stayed unchanged. Current parsed manifest SHA-256: 3d27505b085a9506a11a87855fa782031500248ea288d48700da44619976ccb6. Source baselines/2026-10-14/regenerate_parsed.py remains the reproducible method.";
  const fsDatedBaselineExtraction = "FS Fotovalidering 14 Oct 2026 source policy: source workbook corrections were kept in the Excel workbook; six source decks now extract 257 unique photos, not 258. No photo pixels were surfaced to the agent.";
  const fsLaterBlurCorrection = "FS Fotovalidering 15 Oct 2026 owner-blur correction: source workbook Godkendt!A4 says blurry and C4=x for the blurred accepted photo. Corrected 257-photo baseline; old manifest SHA 225de8e9d265ea284fd3a5b8de7e6bf56eda9c2dcedb24b5544a84ff26c348b0 was superseded by new d0e41600805f1652c86f84974030192a17db1eed20779a15c20f8af948110ec0.";
  const fsDatedBaselineInstruction = "Update FS Fotovalidering 2026-10-14 dated test baseline and new 24-head retrain evaluation. As of 15 Oct 2026 the current on-disk `~/Jottacloud/syv-ai/faerdselsstyrelsen/baselines/2026-10-14/parsed/manifest.jsonl` contains **257** test images (not the earlier reported 258); SHA-256 `225de8e9d265ea284fd3a5b8de7e6bf56eda9c2dcedb24b5544a84ff26c348b0`. Paths JSONL also has 257 rows, parsed/images 257 files, no unlabelled image exclusions, all 257 have exact canonical 24 binary labels, 247 role=reject and 10 role=accept; all seven recorded source XLSX/PPTX hashes still match. Do not erase historical 258-photo/published-manifest notes: this is the currently parsed on-disk version, and no source edits were made. On Mac MPS, safely evaluated checkpoint SHA `f17cdc68e95d7afea6aa2f53e456d2bab34afaeb1fdd7bc1a91307c4a0114c86` at fixed 0.50, no TTA, 1000 row-bootstrap draws, no HTML/PDF, without displaying or inspecting photo pixels. Results under ignored private run `data/experiments/experiment-89/model/owner-25-review/retrained-sparkie-binary-exclude-review-20261014/test-evaluation-baseline-2026-10-14/`; evaluation exit 0, report SHA `48590b5a949f9cd65c5707f41c183341d54014d78c3b988e70530623330df776`. Whole-photo confusion TP232 TN7 FP3 FN15: false rejection 3/10=30.0%, false approval 15/247=6.07%; macro F1 raw24=0.5496813 but 3 heads lack positive test support, positive-support21 macro F1=0.4853501, micro F1=0.5930599. No exact image digest overlaps with 24,285 synthetic train/validation samples. This test set has been repeatedly used, so numbers are descriptive and must not be represented as independent held-out evidence or proof of FS thresholds. Prior owner policy mapping unknown→false applies to train/validation corpus; this test baseline already has binary labels and was not modified.";

  it("appends the currently parsed FS test baseline without erasing published 258-photo history", async () => {
    staged();
    const prior = `${fsDatedBaselineExtraction}\n\n${fsDatedBaselinePrior}\n\nOther current baseline source policy remains intact.`;
    await kb.writeConcept(fsDatedBaselineOwner, { type: "Project Continuation", title: "FS dated test baseline" }, prior, "seed");
    await kb.writeConcept("/projects/fs-test-peer.md", { type: "Project", title: "FS test details" },
      `The baselines/2026-10-14 source decks yield 257, while a dated published review mentioned 258 photos.\n\n${fsLaterBlurCorrection}`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fsDatedBaselinePrior);
      expect(prompt).toContain("RELEVANT PEERS:");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fsDatedBaselineInstruction, { preflightInput: fsDatedBaselineInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsDatedBaselineOwner] } });
    const current = (await kb.readConcept(fsDatedBaselineOwner)).body;
    expect(current).toContain(prior);
    expect(current).not.toContain("As of 15 Oct 2026 the current on-disk");
    for (const fact of ["Historical FS on-disk dated test baseline and retrain evaluation (before the 15 Oct 2026 owner-blur correction):",
      "225de8e9d265ea284fd3a5b8de7e6bf56eda9c2dcedb24b5544a84ff26c348b0",
      "d0e41600805f1652c86f84974030192a17db1eed20779a15c20f8af948110ec0", "then-current on-disk",
      "48590b5a949f9cd65c5707f41c183341d54014d78c3b988e70530623330df776",
      "TP232 TN7 FP3 FN15", "3 heads lack positive test support", "no source edits were made",
      "independent held-out evidence"])
      expect(current).toContain(fact);
    expect(await runMutation(kb, fsDatedBaselineInstruction, { preflightInput: fsDatedBaselineInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("allows a peer's historical 258-image manifest reference without assigning it ownership", async () => {
    staged();
    await kb.writeConcept(fsDatedBaselineOwner, { type: "Project Continuation", title: "FS baseline" },
      `${fsDatedBaselineExtraction}\n\n${fsDatedBaselinePrior}`, "seed");
    await kb.writeConcept("/projects/fs-test-peer.md", { type: "Project", title: "FS test history" },
      `Prior 258-image test manifest SHA 3d27505b085a9506a11a87855fa782031500248ea288d48700da44619976ccb6 remained historical.\n\n${fsLaterBlurCorrection}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsDatedBaselineInstruction, { preflightInput: fsDatedBaselineInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsDatedBaselineOwner] } });
  });

  it("defers a competing FS test baseline owner or semantically duplicate evaluation", async () => {
    staged();
    await kb.writeConcept(fsDatedBaselineOwner, { type: "Project Continuation", title: "FS baseline" },
      `${fsDatedBaselineExtraction}\n\n${fsDatedBaselinePrior}`, "seed");
    await kb.writeConcept("/projects/fs-test-peer.md", { type: "Project", title: "FS test" },
      `${fsDatedBaselineExtraction}\n\n${fsDatedBaselinePrior}\n\n${fsLaterBlurCorrection}`, "seed");
    expect(await runMutation(kb, fsDatedBaselineInstruction, { preflightInput: fsDatedBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing or already updated peer") });
    await kb.writeConcept("/projects/fs-test-peer.md", { type: "Project", title: "FS test" },
      `Checkpoint f17cdc68e95d7afea6aa2f53e456d2bab34afaeb1fdd7bc1a91307c4a0114c86 was used for another test evaluation: TP232 TN7 FP3 FN15.\n\n${fsLaterBlurCorrection}`, "seed");
    expect(await runMutation(kb, fsDatedBaselineInstruction, { preflightInput: fsDatedBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing or already updated peer") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("defers the FS dated test update if independent ownership review declines", async () => {
    staged();
    await kb.writeConcept(fsDatedBaselineOwner, { type: "Project Continuation", title: "FS baseline" },
      `${fsDatedBaselineExtraction}\n\n${fsDatedBaselinePrior}`, "seed");
    await kb.writeConcept("/projects/fs-test-peer.md", { type: "Project", title: "FS corrected test" },
      fsLaterBlurCorrection, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fsDatedBaselineInstruction, { preflightInput: fsDatedBaselineInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS dated test baseline check declined") });
    expect((await kb.readConcept(fsDatedBaselineOwner)).body).not.toContain("225de8e9d265ea28");
  });

  const fsPriorityInstruction = "Update FS Fotovalidering 15 Oct 2026 retrained 5,441-card synthetic annotation-page order at file:///tmp/fs-v3-merged24-retrained-binary-review-20261014.html. Dan requested focus on the three 2026-10-14 test false-rejection trigger categories, without adding test photos to synthetic annotation: `image_blurred`, `background_not_uniform`, `eyes_not_looking_straight`. Current page uses four stable buckets in order: 657 fitting train/validation samples with a false-positive model flag at fixed0.50 in any target category; 1,483 other fitting samples; 504 excluded-from-fitting target-category false positives; 2,797 other excluded samples. Within each bucket, prior descending class-balanced BCE model-error priority, disagreement-count and stable-ID ties remain unchanged. All 5,441 IDs, category display order, annotations by ID, storage key, server 127.0.0.1:8782 and original immutable queues remain intact. New private queue: `.../retrained-sparkie-binary-exclude-review-20261014/next-review-queue-fit-first-target-fp.jsonl`, SHA `79828fca1165dc1f94d58d9d5f4b90307c9b4f9fca971ad5b70a609f9da869a0`; new page SHA `31bf05689e43c6b3388664d78b88fcf578a724bad7871b30e9ce47f255dc4eb7` supersedes old fit-first page SHA `f478b3d193fd8276a8e25899726fcc98ad4c100a59571dbf5d7642e942e33bb4`. `/tmp/recreate-fs-retrained-binary-review.py` updated to reconstruct current priority from frozen private queue; preserving same page path/key is critical for browser autosave.";
  const fsPriorityOwner = "/projects/fs-fotovalidering/continuation-retrained.md";
  const fsPriorPage = "Inference on all 24,285 synthetic images produced 5,441 class-balanced-BCE-prioritised cards at file:///tmp/fs-v3-merged24-retrained-binary-review-20261014.html, local-only server 127.0.0.1:8782 and page SHA f381c4ab888136d905d0db51bdc336f77ff7a45f68f6de229bbabe2e8daf3bda. The previous display order remains; owner reviews retain their annotations.";

  it("records the retrained FS page's four priority buckets without discarding its earlier order", async () => {
    staged();
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" },
      `${fsPriorPage}\n\nUnrelated review history stays.`, "seed");
    await kb.writeConcept("/projects/fs-fotovalidering.md", { type: "Project", title: "FS Fotovalidering" },
      "The test baseline's three false rejections motivate prioritizing synthetic reviews.", "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fsPriorPage);
      expect(prompt).toContain("RELEVANT PEERS:");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsPriorityOwner] } });
    const body = (await kb.readConcept(fsPriorityOwner)).body;
    expect(body).toContain(fsPriorPage);
    expect(body).not.toContain("Earlier FS retrained synthetic review-page snapshot (superseded by target-category priority):");
    for (const anchor of ["657", "1,483", "504", "2,797", "image_blurred", "background_not_uniform",
      "eyes_not_looking_straight", "79828fca", "31bf0568", "f478b3d1", "127.0.0.1:8782"])
      expect(body).toContain(anchor);
    expect(body).toContain("Unrelated review history stays.");
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(fsPriorityOwner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("corrects a broad superseded label without retracting current FS training policy", async () => {
    staged();
    const mixed = `FS Fotovalidering 14 Oct 2026 owner-confirmed source policy: every image in the current train/validation corpus is synthetic; only the separate test sets are excluded. ${fsPriorPage}`;
    const current = `Current FS retrained synthetic review-page priority (15 Oct 2026): ${fsPriorityInstruction.replace(/^Update FS Fotovalidering 15 Oct 2026 retrained 5,441-card synthetic annotation-page order at /i, "The 5,441-card synthetic annotation page at ")}`;
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" },
      `Earlier FS retrained synthetic review-page snapshot (superseded by target-category priority): ${mixed}\n\n${current}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsPriorityOwner] } });
    const body = (await kb.readConcept(fsPriorityOwner)).body;
    expect(body).toContain(mixed);
    expect(body).toContain(current);
    expect(body).not.toContain("Earlier FS retrained synthetic review-page snapshot (superseded by target-category priority):");
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("does not accept a truncated archived FS review page as a complete repeat", async () => {
    staged();
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" }, fsPriorPage, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: true });
    const completed = (await kb.readConcept(fsPriorityOwner)).body;
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" },
      completed.replace("24,285 synthetic images", "synthetic images"), "damage");
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: false, status: "failed" });
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("refuses to duplicate a differently headed current FS page SHA", async () => {
    staged();
    const newSha = "31bf05689e43c6b3388664d78b88fcf578a724bad7871b30e9ce47f255dc4eb7";
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" },
      `${fsPriorPage}\n\nAnother current page status uses ${newSha}.`, "seed");
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("already present, partial") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("defers the FS page reorder when a peer also owns the current page", async () => {
    staged();
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" }, fsPriorPage, "seed");
    await kb.writeConcept("/projects/fs-fotovalidering.md", { type: "Project", title: "FS overview" },
      `Separate snapshot: ${fsPriorPage}`, "seed");
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("owner is absent, duplicated") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("defers the FS page reorder when independent ownership review declines", async () => {
    staged();
    await kb.writeConcept(fsPriorityOwner, { type: "Project", title: "FS retrained annotation" }, fsPriorPage, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fsPriorityInstruction, { preflightInput: fsPriorityInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS review-page check declined") });
    expect((await kb.readConcept(fsPriorityOwner)).body).toContain(fsPriorPage);
  });

  const fsChangesOwner = "/projects/fs-fotovalidering-continuation-c03a2624495c6bd5.md";
  const fsChangesOld = "Dan's FS Fotovalidering 2026-10-14 baseline change report should be in Danish. The Markdown and PDF at ~`Jottacloud/syv-ai/faerdselsstyrelsen/baselines/2026-10-14/baseline_changes` and .pdf were translated from English on 2026-10-14; the Danish visible/PDF metadata title is 'Ændringer i baselinen fra 2026-10-14'. All 459 photo-ID mentions, 34 approved-photo ID mappings, category-change lists, and numerical claims were preserved. Keep the report ID-based, without hashes, and do not inspect baseline photos.";
  const fsChangesInstruction = "Update the FS Fotovalidering 2026-10-14 baseline change-report knowledge: Dan wants a Danish report focused strictly on differences from the 2026-10-09 baseline, not unchanged items, interim-review history, or a full current-state inventory. The existing ~/Jottacloud/syv-ai/faerdselsstyrelsen/baselines/2026-10-14/baseline_changes.md and .pdf were revised to use title 'Ændringer fra baselinen 2026-10-09 til 2026-10-14'. The current report includes only 30 changed approved-photo ID mappings (23 moved, seven renumbered), removal of godkendt_id30, two previously unlabelled existing photos now labelled, changed presentation counts, 264 added and 28 removed active category marks, new photo-of-a-photo marks / retired Hovedbeklædning 4, and 68 changed embedded photo files by ID. It excludes unchanged 50 workbook-only columns, unchanged approved IDs, and the 43 intermediate question-mark resolutions. Supersede any statement that the CURRENT report preserves 459 ID mentions or all 34 approved-photo mappings; those referred to the earlier non-focused Danish version. Keep it ID-based, no hashes, no inspecting baseline photos.";

  it("archives the earlier FS report and records the complete changes-only edition", async () => {
    staged();
    const later = "Separate source-only baseline milestone retains 5779826 and 257 unique photo SHA-256s.";
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" },
      `${fsChangesOld}\n\n${later}`, "seed");
    await kb.writeConcept("/projects/fs-fotovalidering-continuation-other.md", { type: "Project", title: "FS peer" },
      "2026-10-14 baseline image audit is an unrelated milestone.", "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(fsChangesOld);
      expect(prompt).toContain(later);
      expect(prompt).toContain("RELEVANT PEERS:");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsChangesOwner] } });
    const body = (await kb.readConcept(fsChangesOwner)).body;
    expect(body).toContain(`Earlier non-focused Danish report version (superseded by changes-only edition): ${fsChangesOld}`);
    expect(body).toContain("Current FS Fotovalidering 2026-10-14 changes-only baseline report:");
    for (const item of ["Ændringer fra baselinen 2026-10-09 til 2026-10-14", "30 changed approved-photo ID mappings",
      "23 moved and seven renumbered", "godkendt_id30", "two previously unlabelled", "264 added and 28 removed",
      "Hovedbeklædning 4", "68 changed embedded photo files by ID", "unchanged 50 workbook-only columns",
      "43 intermediate question-mark resolutions", "459 photo-ID mentions and all 34 approved-photo mappings describe only the earlier non-focused Danish version"])
      expect(body).toContain(item);
    expect(body).toContain(later);
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(fsChangesOwner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("uses the observed old paragraph when its Markdown whitespace is reflowed", async () => {
    staged();
    const reflowed = fsChangesOld.replace("All 459 photo-ID mentions,", "All 459 photo-ID\nmentions,");
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" }, reflowed, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsChangesOwner] } });
    const updated = (await kb.readConcept(fsChangesOwner)).body;
    expect(updated).toContain(`Earlier non-focused Danish report version (superseded by changes-only edition): ${reflowed}`);
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
  });

  it("preserves the observed Jottacloud path notation while matching the old FS report", async () => {
    staged();
    const pathVariant = fsChangesOld.replace("~`Jottacloud/syv-ai/faerdselsstyrelsen/baselines/2026-10-14/baseline_changes`",
      "~/`Jottacloud/syv-ai/faerdselsstyrelsen/baselines/2026-10-14/baseline_changes.md`");
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" }, pathVariant, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsChangesOwner] } });
    const body = (await kb.readConcept(fsChangesOwner)).body;
    expect(body).toContain(`Earlier non-focused Danish report version (superseded by changes-only edition): ${pathVariant}`);
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
  });

  it("rejects FS change-report correction if another peer owns the old report", async () => {
    staged();
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" }, fsChangesOld, "seed");
    const peer = "/projects/fs-fotovalidering-continuation-other.md";
    await kb.writeConcept(peer, { type: "Project", title: "FS change report copy" }, fsChangesOld, "seed");
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("owner is absent, duplicated") });
    expect((await kb.readConcept(fsChangesOwner)).body).toContain(fsChangesOld);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("locates the unique complete old FS report in another continuation", async () => {
    staged();
    const movedOwner = "/projects/fs-fotovalidering/continuation-report.md";
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "Other FS milestone" },
      "Other FS source-only baseline checkpoint.", "seed");
    await kb.writeConcept(movedOwner, { type: "Project", title: "FS report" }, fsChangesOld, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [movedOwner] } });
    expect((await kb.readConcept(fsChangesOwner)).body).toContain("Other FS source-only baseline checkpoint");
    expect((await kb.readConcept(movedOwner)).body).toContain("changes-only baseline report");
  });

  it("rejects a nested FS peer claiming the obsolete report without the filename", async () => {
    staged();
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" }, fsChangesOld, "seed");
    const peer = "/projects/fs-fotovalidering/report-duplicate.md";
    await kb.writeConcept(peer, { type: "Project", title: "FS report copy" },
      "The 2026-10-14 baseline change report currently preserves all 459 photo-ID mentions and 34 approved-photo ID mappings.", "seed");
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("contradicted") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("rejects a completed FS report with its earlier full-history paragraph truncated", async () => {
    staged();
    const historical = `Earlier non-focused Danish report version (superseded by changes-only edition): ${fsChangesOld}`;
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" }, fsChangesOld, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [fsChangesOwner] } });
    const completed = (await kb.readConcept(fsChangesOwner)).body;
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" },
      completed.replace(historical, historical.replace("category-change lists, and numerical claims", "category-change lists")), "damage");
    const result = await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(fsChangesOwner)).body).not.toContain(historical);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("rejects FS change-report correction if independent peer check declines", async () => {
    staged();
    await kb.writeConcept(fsChangesOwner, { type: "Project", title: "FS Fotovalidering" }, fsChangesOld, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, fsChangesInstruction, { preflightInput: fsChangesInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent FS change-report check declined") });
    expect((await kb.readConcept(fsChangesOwner)).body).toContain(fsChangesOld);
  });

  it("records a synthetic repository review-thread checkpoint and rejects an authoritative peer collision", async () => {
    staged();
    const instruction = "Acme PR #731: after verifying head a1b2c3d4e5f60718293a4b5c6d7e8f9012345678, all three review threads were resolved. The existing CHANGES_REQUESTED review remains unchanged; no approval or merge occurred.";
    const ownerPath = "/repos/acme/widget-review.md";
    const followup = "Acme PR #731 follow-up: head a1b2c3d4e5f60718293a4b5c6d7e8f9012345678 addresses requested changes; prior review remains CHANGES_REQUESTED until independent approval.";
    await kb.writeConcept(ownerPath, { type: "Repository", title: "Widget review" }, followup, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [ownerPath] } });
    expect((await kb.readConcept(ownerPath)).body).toContain("PR #731 review-thread resolution");

    const peerPath = "/repos/acme/duplicate-review.md";
    await kb.writeConcept(peerPath, { type: "Repository", title: "Duplicate review" },
      "PR #731: three review threads were resolved at head a1b2c3d4e5f60718293a4b5c6d7e8f9012345678.", "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { exampleOnly: false } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("another owner") });
  });

  const euroevalResolvedThreadsInstruction = "EuroEval PR #2234: after verifying head 67fe25b4a06099b423916f9de6285ad340ab7e1e, all six review threads (Dan's four plus Copilot's two) were resolved on GitHub. The CHANGES_REQUESTED review has not been replaced with an approval; resolving threads alone does not change review decision.";
  const euroevalFollowup = "EuroEval PR #2234 follow-up: new head 67fe25b4a06099b423916f9de6285ad340ab7e1e fixes the two substantive requested changes and Dan's two readability notes. Recheck LGTM; prior review still CHANGES_REQUESTED until an explicit approval.";
  const euroevalThreadsOwner = "/repos/euroeval/orthogonal-leaderboard-confidence-intervals.md";

  it("records six resolved EuroEval review threads without approving the existing review", async () => {
    staged();
    const later = "PR #2234 was subsequently squash-merged; retain this separate lifecycle record.";
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      `${euroevalFollowup}\n\n${later}`, "seed");
    await kb.writeConcept("/repos/euroeval.md", { type: "Repository", title: "EuroEval" },
      "EuroEval PR #2234 review 5518245608 was submitted with CHANGES_REQUESTED and four inline comments.", "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(euroevalFollowup);
      expect(prompt).toContain(later);
      expect(prompt).toContain("RELEVANT PEERS:");
      expect(prompt).toContain("CHANGES_REQUESTED and four inline comments");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [euroevalThreadsOwner] } });
    const body = (await kb.readConcept(euroevalThreadsOwner)).body;
    expect(body).toContain(euroevalFollowup);
    expect(body).toContain(later);
    expect(body).toContain(euroevalResolvedThreadsInstruction);
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("classifies a quoted failed EuroEval queue request without assigning it PR ownership", async () => {
    staged();
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      euroevalFollowup, "seed");
    const runbook = "/repos/pi-agent/memory-async.md";
    await kb.writeConcept(runbook, { type: "Repository", title: "Memory queue" },
      `Queue processing history. ${"This is an async request example and not the authoritative EuroEval review. ".repeat(70)} Example failed memory_update: ${euroevalResolvedThreadsInstruction} It hit the generation cap.`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(`PEER ${runbook}:`);
      expect(prompt).toContain("Example failed memory_update");
      expect(prompt).not.toContain("PROPOSED NOTE:");
      return { object: { exampleOnly: true } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("independently classified as queue example");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [euroevalThreadsOwner] } });
    expect((await kb.readConcept(runbook)).body).toContain("Example failed memory_update");
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("rejects a runbook peer that independently asserts EuroEval review-thread status", async () => {
    staged();
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      euroevalFollowup, "seed");
    await kb.writeConcept("/repos/pi-agent/memory-async.md", { type: "Repository", title: "Memory queue" },
      `The runbook currently owns this PR #2234 review record: ${euroevalResolvedThreadsInstruction}`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { exampleOnly: false } });
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("peer asserts competing review-thread status") });
    expect(generateObjectMock).toHaveBeenCalledOnce();
    expect((await kb.readConcept(euroevalThreadsOwner)).body).not.toContain("review-thread resolution at checked head");
  });

  it("defers duplicate or incomplete EuroEval review-thread status", async () => {
    staged();
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      euroevalFollowup, "seed");
    await kb.writeConcept("/repos/euroeval-other.md", { type: "Repository", title: "EuroEval" },
      "PR #2234: six review threads including Copilot were resolved already.", "seed");
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("another owner") });
    await kb.writeConcept("/repos/euroeval-other.md", { type: "Repository", title: "EuroEval" },
      "Different PR status.", "seed");
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      `${euroevalFollowup}\n\nPR #2234 six review threads from Dan and Copilot were resolved, but decision remains under review.`, "seed");
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("partial or conflicting") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("keeps a capped EuroEval review-thread safety verdict from writing", async () => {
    staged();
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      euroevalFollowup, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true }, finishReason: "length" });
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("output cap") });
    expect((await kb.readConcept(euroevalThreadsOwner)).body.trim()).toBe(euroevalFollowup);
  });

  it("defers the EuroEval review-thread update if independent ownership check declines", async () => {
    staged();
    await kb.writeConcept(euroevalThreadsOwner, { type: "Repository", title: "EuroEval review follow-up" },
      euroevalFollowup, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, euroevalResolvedThreadsInstruction, { preflightInput: euroevalResolvedThreadsInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent review-thread check declined") });
    expect((await kb.readConcept(euroevalThreadsOwner)).body.trim()).toBe(euroevalFollowup);
  });

  it("updates a renamed repository's submitted review without changing another PR", async () => {
    staged();
    const owner = "/repos/cedar-reports.md";
    const oldReview = "On 17 Oct Casey created unsubmitted PENDING review 312485 on head abcdef1234567890 with two inline comments: (1) clarify totals; (2) fix sorting. Comment IDs 91001 and 91002. Review was intentionally NOT submitted.";
    const instruction = "Update Cedar Reports PR #73 review status: review ID 312485 by Casey, previously PENDING, was submitted with state CHANGES_REQUESTED. It contains two inline comments. No longer describe it as unsubmitted.";
    await kb.writeConcept(owner, { type: "Repository", title: "Cedar Reports" },
      `${oldReview}\n\nPR #74 remains PENDING for a different review.`, "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: true } });
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const current = (await kb.readConcept(owner)).body;
    expect(current).toContain("review 312485 on head abcdef1234567890 with state CHANGES_REQUESTED and two inline comments");
    expect(current).toContain("Comment IDs 91001 and 91002");
    expect(current).toContain("PR #74 remains PENDING");
    expect(current).not.toContain("Review was intentionally NOT submitted");
    expect(await runMutation(kb, instruction, { preflightInput: instruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
  });

  const euroevalSubmittedInstruction = "Update EuroEval PR #2234 review status: review ID 5518245608 by saattrupdan, previously PENDING, was submitted on 14 Oct 2026 with state CHANGES_REQUESTED. It contains four inline comments: two substantive findings on multilingual ranking refresh and invalid-marker display, plus Dan's two readability/style comments in task_metadata.py and invalid_results.py. No longer describe it as unsubmitted.";
  const euroevalOldReview = "On 14 Oct 2026 Dan's GitHub account saattrupdan created unsubmitted PENDING review 5518245608 on head 0f255b1278c47a2051d348cd222ed0f2f862bdce with two inline comments: (1) leaderboard_generation.py:252 compares only overall ranked membership, so a per-language rank change for an overall-unranked model leaves multilingual CSVs stale on non-force runs; (2) leaderboard_generation.py:841 appends '(invalid)' to orthogonal percentage scores but LeaderboardTable.vue percent formatter and scatter tooltip drop the suffix (and interval). Comment IDs 4271055779 and 4271055802. Review was intentionally NOT submitted.";

  it("confirms submitted EuroEval review without rewriting final PR status or unrelated review history", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    const other = "On 15 Oct PR #2234 was squash-merged as commit ac33731f88806b1a45593c83dbfd473f1cc877fa.";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" },
      `Prior EuroEval review #5510000000 remains historical.\n\n${euroevalOldReview}\n\n${other}`, "seed");
    await kb.writeConcept("/repos/euroeval-followup.md", { type: "Repository", title: "EuroEval followup" },
      "Separate PR #2234 review discussions are historical.", "seed");
    generateObjectMock.mockRejectedValueOnce(new NoObjectGeneratedError({
      message: "No object generated: could not parse the response.", response: {} as never,
      usage: {} as never, finishReason: "stop", text: "invalid JSON",
    })).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(euroevalOldReview);
      expect(prompt).toContain(other);
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).not.toContain("unsubmitted PENDING review 5518245608");
    expect(body).not.toContain("Review was intentionally NOT submitted.");
    expect(body).toContain("review 5518245608 on head 0f255b1278c47a2051d348cd222ed0f2f862bdce with state CHANGES_REQUESTED and four inline comments");
    expect(body).toContain("4271055779 and 4271055802");
    expect(body).toContain("a readability/style comment in task_metadata.py");
    expect(body).toContain("a readability/style comment in invalid_results.py");
    expect(body).toContain(other);
    expect(body).toContain("Prior EuroEval review #5510000000 remains historical.");
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("defers EuroEval review correction for a competing owner or declined independent review", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    const peer = "/projects/euroeval-review.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" }, euroevalOldReview, "seed");
    await kb.writeConcept(peer, { type: "Project", title: "EuroEval review" },
      "Review 5518245608 is still PENDING.", "seed");
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing owner") });
    expect(generateObjectMock).not.toHaveBeenCalled();
    await kb.writeConcept(peer, { type: "Project", title: "EuroEval review" }, "Other PR history.", "replace");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent check declined") });
    expect((await kb.readConcept(owner)).body).toContain(euroevalOldReview);
  });

  it("finds a competing PR review claim even without the review ID", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    const peer = "/users/dan-review-note.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" }, euroevalOldReview, "seed");
    await kb.writeConcept(peer, { type: "Person", title: "Dan review note" },
      "Dan's review on EuroEval PR #2234 is still PENDING and unsubmitted.", "seed");
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing owner") });
    expect((await kb.readConcept(owner)).body).toContain(euroevalOldReview);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("adjudicates a quoted failed queue request without treating the runbook as a review owner", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    const runbook = "/repos/pi-agent/memory-async.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" }, euroevalOldReview, "seed");
    await kb.writeConcept(runbook, { type: "Repository", title: "Memory async" },
      `Queue background. ${"Queue-specific operations, not the EuroEval review record. ".repeat(100)} Example failed request: Update EuroEval PR #2234 review status: Dan's review 5518245608 is PENDING and unsubmitted. The receipt needs retry; do not store the quoted instruction here.`, "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("PEER /repos/pi-agent/memory-async.md:");
      expect(prompt).toContain("Example failed request");
      expect(prompt).not.toContain("CORRECTION:");
      return { object: { exampleOnly: true, reason: "The review is quoted in a queued-write example." } };
    }).mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain("independently classified as a queue example");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    expect((await kb.readConcept(runbook)).body).toContain("Example failed request");
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("rejects a runbook peer when independent review says it owns the pending status", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" }, euroevalOldReview, "seed");
    await kb.writeConcept("/repos/pi-agent/memory-async.md", { type: "Repository", title: "Memory async" },
      "The current EuroEval PR #2234 review 5518245608 by Dan is still PENDING.", "seed");
    generateObjectMock.mockResolvedValueOnce({ object: { exampleOnly: false, reason: "This peer asserts current review status." } });
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("adjudication declined") });
    expect((await kb.readConcept(owner)).body).toContain(euroevalOldReview);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("does not claim a corrected review is complete while another old assertion remains", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" },
      `${euroevalOldReview}\n\nDan's review 5518245608 remains PENDING on PR #2234.`, "seed");
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("another stale owner passage") });
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("does not retry malformed JSON when the generation hit its output cap", async () => {
    staged();
    const owner = "/repos/euroeval.md";
    await kb.writeConcept(owner, { type: "Repository", title: "EuroEval" }, euroevalOldReview, "seed");
    generateObjectMock.mockRejectedValueOnce(new NoObjectGeneratedError({
      message: "No object generated: could not parse the response.", response: {} as never,
      usage: {} as never, finishReason: "length", text: "incomplete JSON",
    }));
    expect(await runMutation(kb, euroevalSubmittedInstruction, { preflightInput: euroevalSubmittedInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("output cap") });
    expect((await kb.readConcept(owner)).body).toContain(euroevalOldReview);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  const confirmedDeployInstruction = "For saattrupdan.com deployment knowledge, correct the prior provisional state: makefile deploy target removal and AGENTS.md Git-deployment guidance were committed as e7159f328f1d54bf6c789b625429b1591e264381 and pushed to origin/main on 2026-10-14. Vercel Git-triggered production deployment dpl_6Q5jrRuUwBbkABbSHwqUhahUQDb8 from that exact SHA reached READY, GitHub Vercel status success, and www.saattrupdan.com aliases the new deployment. Working tree clean.";
  const provisionalDeploy = "On 2026-10-14, the `deploy` target was removed from `makefile` and `AGENTS.md` was updated locally; those changes were not yet committed or pushed in that session.";
  const deploymentBody = `# Deployment Policy\n\nVercel project \`my-website\` deploys on push to main.\n\nThe previous deployment dpl_F6vAFevPgFPEwLrD7YX9CDtjms4L was Ready.\n\n${provisionalDeploy}\n\n# Vercel CLI\n\n\`vercel --prod\` returned Not authorized.`;

  it("keeps permissive non-string YAML titles out of typed search hits", async () => {
    await kb.writeConcept("/notes/malformed-title.md",
      { type: "Note", title: { name: "Vercel" } as unknown as string },
      "Vercel website release history.", "seed");
    const hit = (await kb.search("Vercel")).find((candidate) => candidate.path === "/notes/malformed-title.md");
    expect(hit).toBeDefined();
    expect(hit?.title).toBeUndefined();
  });

  it("corrects the provisional website deployment without losing earlier deployment or CLI history", async () => {
    staged();
    const owner = "/repos/saattrupdan-com.md";
    await kb.writeConcept(owner, { type: "Repository", title: "saattrupdan.com Website" }, deploymentBody, "seed");
    await kb.writeConcept("/projects/saattrupdan-com.md", { type: "Project", title: "Site project" },
      "The website project references make deploy, without claiming the completed Git deployment.", "seed");
    generateObjectMock.mockImplementationOnce(async ({ prompt }: { prompt: string }) => {
      expect(prompt).toContain(provisionalDeploy);
      expect(prompt).toContain("dpl_F6vAFevPgFPEwLrD7YX9CDtjms4L");
      expect(prompt).toContain("e7159f328f1d54bf6c789b625429b1591e264381");
      return { object: { safe: true } };
    });
    expect(await runMutation(kb, confirmedDeployInstruction, { preflightInput: confirmedDeployInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).not.toContain(provisionalDeploy);
    expect(body).toContain("both changes were committed as `e7159f328f1d54bf6c789b625429b1591e264381`");
    expect(body).toContain("`dpl_6Q5jrRuUwBbkABbSHwqUhahUQDb8` from that exact commit reached READY");
    expect(body).toContain("dpl_F6vAFevPgFPEwLrD7YX9CDtjms4L");
    expect(body).toContain("`vercel --prod` returned Not authorized.");
    expect(await runMutation(kb, confirmedDeployInstruction, { preflightInput: confirmedDeployInstruction }))
      .toMatchObject({ ok: true, result: { filesChanged: [] } });
    expect((await kb.readConcept(owner)).body).toBe(body);
    expect(generateObjectMock).toHaveBeenCalledOnce();
  });

  it("defers the confirmed website deployment when a peer owns it or independent review declines", async () => {
    staged();
    const owner = "/repos/saattrupdan-com.md";
    const peer = "/repos/saattrupdan-com-deploy.md";
    await kb.writeConcept(owner, { type: "Repository", title: "saattrupdan.com Website" }, deploymentBody, "seed");
    await kb.writeConcept(peer, { type: "Project", title: "Website release" },
      "Commit e7159f328f1d54bf6c789b625429b1591e264381 was deployed as dpl_6Q5jrRuUwBbkABbSHwqUhahUQDb8.", "seed");
    expect(await runMutation(kb, confirmedDeployInstruction, { preflightInput: confirmedDeployInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("competing owner") });
    expect(generateObjectMock).not.toHaveBeenCalled();
    await kb.writeConcept(peer, { type: "Project", title: "Website release" }, "Other Vercel history.", "replace");
    generateObjectMock.mockResolvedValueOnce({ object: { safe: false } });
    expect(await runMutation(kb, confirmedDeployInstruction, { preflightInput: confirmedDeployInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("independent check declined") });
    expect((await kb.readConcept(owner)).body).toContain(provisionalDeploy);
  });

  it("does not rewrite a historical provisional sentence outside the deployment section", async () => {
    staged();
    const owner = "/repos/saattrupdan-com.md";
    const outside = deploymentBody.replace(provisionalDeploy, "The Git deployment policy is active.") +
      `\n\n# Dated observations\n\n${provisionalDeploy}`;
    await kb.writeConcept(owner, { type: "Repository", title: "saattrupdan.com Website" }, outside, "seed");
    expect(await runMutation(kb, confirmedDeployInstruction, { preflightInput: confirmedDeployInstruction }))
      .toMatchObject({ ok: false, status: "failed", error: expect.stringContaining("absent or ambiguous") });
    expect((await kb.readConcept(owner)).body).toBe(`${outside}\n`);
    expect(generateObjectMock).not.toHaveBeenCalled();
  });

  it("replaces only the obsolete site deployment policy and preserves dated CLI history", async () => {
    staged();
    const owner = "/repos/saattrupdan-com.md";
    const oldPolicy = "## Deployment Policy\n\n**`make deploy`** is the authoritative deployment command for saattrupdan.com.\n\n" +
      "The required workflow:\n1. Complete all work on a feature branch.\n2. Merge the completed feature branch into `main`.\n" +
      "3. Run `make deploy` to publish.\n4. Only after deployment succeeds, clean up (delete) the feature branch.\n\n" +
      "Never delete the feature branch before confirming deployment has succeeded — a failed deploy may require re-deploying from that branch.\n\n";
    const cliHistory = "## Vercel CLI `--prod` Authorization Gotcha\n\n" +
      "**Observed:** 18 Sep 2026. Running `vercel whoami` and project inspection succeeded, but `vercel --prod` returned `Not authorized`.\n";
    const instruction = "Update the saattrupdan.com deployment workflow knowledge: Vercel project my-website is Git-connected to saattrupdan/saattrupdan.com with main as production branch. " +
      "The 2026-10-14 production deployment dpl_F6vAFevPgFPEwLrD7YX9CDtjms4L was Git-triggered from commit 3dd02f09300a208f9addf936cc273a216ea449f9, Ready, aliased to www.saattrupdan.com; GitHub Vercel status success. " +
      "Replace obsolete guidance that `make deploy` is authoritative or must run before deleting a branch. The Vite build generates dist/atom.xml and dist/sitemap.xml via src/build/feeds.ts, independent of the legacy local Python feed target. " +
      "On 2026-10-14 the `deploy` target was removed from makefile and AGENTS.md updated locally (not yet committed/pushed in that session); push to main is the intended deployment mechanism. " +
      "Retain note that Vercel CLI `vercel --prod` returns Not authorized even though whoami and project inspection work.";
    const related = "/projects/saattrupdan-com.md";
    await kb.writeConcept(owner, { type: "Repository", title: "saattrupdan.com" }, `Historical notes.\n\n${oldPolicy}${cliHistory}`, "seed");
    await kb.writeConcept(related, { type: "Project", title: "saattrupdan.com deployment" },
      "The project records a Git-connected Vercel deployment.", "seed");
    vi.spyOn(kb, "search").mockResolvedValue([
      { path: related, type: "Project", title: "saattrupdan.com deployment", score: 100 },
    ]);
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: related,
      old_text: "**`make deploy`** is the authoritative deployment command for saattrupdan.com.",
      new_text: "Push to main is the intended deployment mechanism." } })
      .mockResolvedValueOnce({ object: { safe: true } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result, JSON.stringify(result)).toMatchObject({ ok: true, result: { filesChanged: [owner] } });
    const body = (await kb.readConcept(owner)).body;
    expect(body).toContain(cliHistory);
    expect(body).toContain("Historical notes.");
    expect(body).not.toContain("is the authoritative deployment command");
    expect(body).toContain("dpl_F6vAFevPgFPEwLrD7YX9CDtjms4L");
    expect(body).toContain("3dd02f09300a208f9addf936cc273a216ea449f9");
    expect(body).toContain("dist/atom.xml");
    expect(body).toContain("dist/sitemap.xml");
    expect(body).toContain("not yet committed or pushed in that session");
    expect((await kb.readConcept(related)).body).toContain("Git-connected Vercel deployment");
    expect(generateObjectMock).toHaveBeenCalledTimes(2);
  });

  it("blocks a deployment policy correction when another owner contains the same policy", async () => {
    staged();
    const owner = "/repos/saattrupdan-com.md";
    const peer = "/projects/saattrupdan-com.md";
    const policy = "## Deployment Policy\n\n**`make deploy`** is the authoritative deployment command for saattrupdan.com.\n\n" +
      "The required workflow:\n3. Run `make deploy` to publish.\n4. Only after deployment succeeds, clean up (delete) the feature branch.\n\n" +
      "Never delete the feature branch before confirming deployment has succeeded.\n\n";
    const instruction = "Update the saattrupdan.com deployment workflow knowledge: Vercel project my-website is Git-connected to saattrupdan/saattrupdan.com. " +
      "The 2026-10-14 production deployment dpl_F6vAFevPgFPEwLrD7YX9CDtjms4L was Git-triggered from commit 3dd02f09300a208f9addf936cc273a216ea449f9, " +
      "Ready, aliased to www.saattrupdan.com. Replace obsolete guidance that `make deploy` is authoritative. " +
      "The Vite build generates dist/atom.xml and dist/sitemap.xml via src/build/feeds.ts; push to main is the intended deployment mechanism. " +
      "Retain note that Vercel CLI `vercel --prod` returns Not authorized.";
    await kb.writeConcept(owner, { type: "Repository", title: "saattrupdan.com" },
      `${policy}## Vercel CLI note\n\nThe historical \`vercel --prod\` attempt returned Not authorized.`, "seed");
    await kb.writeConcept(peer, { type: "Project", title: "Site deployment" }, policy, "seed");
    vi.spyOn(kb, "search").mockResolvedValue([owner, peer].map((conceptPath, i) => ({
      path: conceptPath, type: i ? "Project" : "Repository", title: i ? "Site deployment" : "saattrupdan.com", score: 100 - i,
    })));
    generateObjectMock.mockResolvedValueOnce({ object: { ...baseProposal, action: "replace", path: owner,
      old_text: "**`make deploy`** is the authoritative deployment command for saattrupdan.com.",
      new_text: "Push to main is the intended deployment mechanism." } });
    const result = await runMutation(kb, instruction, { preflightInput: instruction });
    expect(result).toMatchObject({ ok: false, status: "failed" });
    expect((await kb.readConcept(owner)).body).toContain("is the authoritative deployment command");
    expect(generateObjectMock).toHaveBeenCalledOnce();
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

  it("reports a capped post-write generation as partial, not success", async () => {
    await billingFixture();
    generateTextMock.mockImplementationOnce(async (request: MutationRequest) => {
      await patchRequest(request);
      return { text: "Updated Billing API approval rule.", steps: [{ toolCalls: [], finishReason: "length" }] };
    });
    const result = await runMutation(kb, updateInstruction, { preflightInput: updateInstruction });
    expect(result).toMatchObject({ ok: false, status: "partial", filesChanged: ["/apis/billing-api.md"],
      error: expect.stringContaining("output cap") });
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
