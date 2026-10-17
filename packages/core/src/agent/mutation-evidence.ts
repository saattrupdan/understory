/** Domain-independent evidence planning for staged mutations. No model or filesystem access. */

export const MAX_EVIDENCE_BODY_CHARS = 64_000;
export const DEFAULT_EVIDENCE_BATCH_SIZE = 16;

export interface MutationConcept {
  path: string;
  title: string;
  type: string;
  body: string;
}

export type EvidenceKind = "claim" | "artifact" | "source" | "identifier" | "fingerprint";
export type AssertionStatus = "current" | "historical" | "quoted" | "negated" | "ambiguous";

export interface MutationEvidence {
  kind: EvidenceKind;
  value: string;
  status: AssertionStatus;
}

export interface EvidenceCandidate {
  path: string;
  title: string;
  type: string;
  evidence: MutationEvidence[];
  compositeAnchors: string[][];
  body?: string;
  rejected?: "oversized-matching-evidence";
}

export interface EvidencePlan {
  candidates: EvidenceCandidate[];
  batches: EvidenceCandidate[][];
  conflicts: string[];
  uniquePath?: string;
  disposition: "unique" | "conflict" | "insufficient" | "rejected";
}

const normalize = (value: string): string => value.toLocaleLowerCase().replace(/\\/g, "/").replace(/\s+/g, " ").trim();
const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function classify(text: string, start: number): AssertionStatus {
  const before = text.slice(Math.max(0, start - 100), start).toLowerCase();
  const around = text.slice(Math.max(0, start - 100), Math.min(text.length, start + 120)).toLowerCase();
  if (/\b(?:not|never|exclude|excluding|unrelated to|rather than|do not)\s+(?:the\s+)?[^.!?;]{0,80}$/i.test(before)) return "negated";
  const leading = text.slice(Math.max(0, start - 12), start);
  if (/["'“‘][^"'”’]*$/.test(leading) || /\b(?:requested|queued|prompted|quoted)\b[^.!?]{0,80}$/.test(before)) return "quoted";
  if (/\b(?:historically|previously|formerly|at the time|snapshot|as of)\b[^.!?]{0,100}$/.test(before)) return "historical";
  if (/\b(?:current|currently|now|latest|authoritative|as of today)\b[^.!?]{0,100}$/.test(before)) return "current";
  if (/\b(?:but|however|whereas|in contrast)\b/.test(around)) return "ambiguous";
  return "ambiguous";
}

function extract(text: string): MutationEvidence[] {
  const patterns: Array<[EvidenceKind, RegExp]> = [
    ["artifact", /\b[a-z][a-z0-9+.-]*:\/\/[^\s<>"')\]]+/gi],
    ["source", /(?:^|\s)(?:\.?\.?\/)?[\w.-]+(?:\/[\w.-]+)+\.[a-z0-9]{1,12}\b/gim],
    ["fingerprint", /\b(?:sha(?:-?\d+)?|hash|fingerprint)\s*(?:is|:|=)?\s*[`'" ]?([a-f0-9]{32,128})\b/gi],
    ["identifier", /\b(?:[a-z][a-z0-9]*[-_]){1,}[0-9]{1,}\b/gi],
    ["claim", /\b(?:was|is|are|has|had|contains|reports|records)\s+[^.!?;\n]{8,100}/gi],
  ];
  const found: MutationEvidence[] = [];
  for (const [kind, pattern] of patterns) {
    for (const match of text.matchAll(pattern)) {
      const value = (match[1] ?? match[0]).trim().replace(/[.,;]+$/, "");
      const index = match.index ?? 0;
      if (value) found.push({ kind, value, status: classify(text, index) });
    }
  }
  return found;
}

function composite(evidence: MutationEvidence[]): string[][] {
  const active = evidence.filter((item) => item.status !== "negated" && item.status !== "quoted");
  const pairs: string[][] = [];
  for (let i = 0; i < active.length; i++) for (let j = i + 1; j < active.length; j++) {
    if (active[i]!.kind !== active[j]!.kind) pairs.push([`${active[i]!.kind}:${active[i]!.value}`, `${active[j]!.kind}:${active[j]!.value}`]);
  }
  return pairs;
}

/**
 * Plan bounded evidence without treating missing text as proof of no conflict.
 * All paths are retained for inventory-level ambiguity; only selected evidence
 * bodies are exposed in capped batches for downstream adjudication.
 */
export function planMutationEvidence(
  request: string,
  concepts: readonly MutationConcept[],
  options: { batchSize?: number; maxBodyChars?: number } = {},
): EvidencePlan {
  const batchSize = Math.max(1, Math.min(64, Math.floor(options.batchSize ?? DEFAULT_EVIDENCE_BATCH_SIZE)));
  const maxBodyChars = Math.max(1, Math.min(MAX_EVIDENCE_BODY_CHARS, Math.floor(options.maxBodyChars ?? MAX_EVIDENCE_BODY_CHARS)));
  const requestEvidence = extract(request);
  const usableRequest = requestEvidence.filter((item) => item.status !== "negated" && item.status !== "quoted");
  const candidates: EvidenceCandidate[] = [];
  const requestNorm = normalize(request);
  for (const concept of concepts) {
    const pathAndTitle = `${concept.path} ${concept.title}`;
    const pathNorm = normalize(pathAndTitle);
    const inventoryMatch = usableRequest.some((item) => pathNorm.includes(normalize(item.value)) && item.kind !== "claim");
    const bodyFits = concept.body.length <= maxBodyChars;
    const oversizedMatch = !bodyFits && usableRequest.some((item) => normalize(concept.body).includes(normalize(item.value)));
    const bodyEvidence = bodyFits ? extract(concept.body) : [];
    const shared = bodyEvidence.filter((item) => usableRequest.some((requestItem) => requestItem.kind === item.kind && normalize(requestItem.value) === normalize(item.value)));
    const oldClaim = bodyEvidence.filter((item) => item.kind === "claim" && requestNorm.includes(normalize(item.value)));
    const evidence = [...shared, ...oldClaim];
    const candidate: EvidenceCandidate = {
      path: concept.path, title: concept.title, type: concept.type, evidence,
      compositeAnchors: composite(evidence),
    };
    if (!bodyFits && (inventoryMatch || oversizedMatch)) candidate.rejected = "oversized-matching-evidence";
    if (inventoryMatch || evidence.length) candidates.push(candidate);
  }
  const eligible = candidates.filter((candidate) => !candidate.rejected && candidate.compositeAnchors.length > 0);
  const conflicts = eligible.map((candidate) => candidate.path);
  const selected = conflicts.length === 1 ? eligible[0] : undefined;
  const visible = candidates.map((candidate) => candidate.body === undefined ? candidate : candidate);
  const batches: EvidenceCandidate[][] = [];
  for (let i = 0; i < visible.length; i += batchSize) batches.push(visible.slice(i, i + batchSize));
  const rejected = candidates.some((candidate) => candidate.rejected);
  return {
    candidates: visible,
    batches,
    conflicts,
    ...(selected ? { uniquePath: selected.path } : {}),
    disposition: rejected ? "rejected" : conflicts.length > 1 ? "conflict" : conflicts.length === 1 ? "unique" : "insufficient",
  };
}

/** Expose a body to adjudication only when it fits the explicit evidence cap. */
export function boundedEvidenceBody(concept: MutationConcept, maxChars = MAX_EVIDENCE_BODY_CHARS): string | undefined {
  return concept.body.length <= Math.min(MAX_EVIDENCE_BODY_CHARS, maxChars) ? concept.body : undefined;
}
