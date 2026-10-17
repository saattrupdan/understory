import { describe, expect, it } from "vitest";
import { planMutationEvidence, type MutationConcept } from "./mutation-evidence.js";

const concept = (path: string, title: string, body: string): MutationConcept => ({ path, title, type: "record", body });

describe("planMutationEvidence", () => {
  it("does not treat the same review identifier in a different repository as ownership", () => {
    const result = planMutationEvidence(
      "Update review-731 for codehost://north/engine/reviews/731 using src/engine.ts",
      [concept("/teams/engine.md", "Engine", "Current review-731 is codehost://south/engine/reviews/731 in src/engine.ts")],
    );
    expect(result.uniquePath).toBeUndefined();
  });

  it("blocks competing peers with the same entity and artifact evidence", () => {
    const request = "Update review-731 for codehost://north/engine/reviews/731 from src/engine.ts";
    const peers = [
      concept("/groups/alpha.md", "Engine", "Current review-731 uses codehost://north/engine/reviews/731 from src/engine.ts"),
      concept("/groups/beta.md", "Engine", "Current review-731 uses codehost://north/engine/reviews/731 from src/engine.ts"),
    ];
    expect(planMutationEvidence(request, peers).disposition).toBe("conflict");
  });

  it("does not identify an owner from an artifact mentioned only in the new request", () => {
    const result = planMutationEvidence("Add codehost://north/engine/reviews/731", [
      concept("/groups/alpha.md", "Engine", "The team maintains the engine."),
    ]);
    expect(result.disposition).toBe("insufficient");
  });

  it("selects a sole old owner with matching source and artifact anchors", () => {
    const result = planMutationEvidence(
      "Update review-731 at codehost://north/engine/reviews/731 in src/engine.ts",
      [concept("/groups/alpha.md", "Engine", "Current review-731 is tracked at codehost://north/engine/reviews/731 in src/engine.ts")],
    );
    expect(result.uniquePath).toBe("/groups/alpha.md");
  });

  it("does not consider quoted or negated artifact mentions authoritative", () => {
    const result = planMutationEvidence(
      'Do not use codehost://north/engine/reviews/731; the queued request said "codehost://north/engine/reviews/731"',
      [concept("/groups/alpha.md", "Engine", "Current review-731 is codehost://north/engine/reviews/731 in src/engine.ts")],
    );
    expect(result.disposition).toBe("insufficient");
  });

  it("rejects oversized matching evidence rather than truncating it", () => {
    const body = `${"x".repeat(65_000)} codehost://north/engine/reviews/731 src/engine.ts review-731`;
    const result = planMutationEvidence(
      "Update codehost://north/engine/reviews/731 in src/engine.ts review-731",
      [concept("/groups/alpha.md", "Engine", body)],
    );
    expect(result.disposition).toBe("rejected");
    expect(result.candidates[0]?.rejected).toBe("oversized-matching-evidence");
  });
});
