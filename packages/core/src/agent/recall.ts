import type { KnowledgeBase } from "../okf/index.js";
import { capEnv } from "../util/env.js";
import type { AgentOptions } from "./agent.js";
import { isMalformedAnswer, MALFORMED_ANSWER_MESSAGE } from "./answer-validation.js";

/**
 * Recall fast path: deterministic retrieval (keyword search plus a one-hop walk
 * over the link graph — no LLM, tens of milliseconds) followed by ONE tool-free
 * generation over the retrieved excerpts. It sits between hot memory and the
 * deep agent, so the common "what do we know about X" question costs a single
 * LLM round-trip instead of the deep loop's many sequential ones.
 *
 * The layer is deliberately conservative. Deterministic recall is keyword
 * based, so it can miss knowledge filed under different wording; when the
 * literal signal is weak, or the model reports the excerpts are not enough, it
 * returns null and the caller escalates to the deep agent. Speed must not be
 * bought with lost recall.
 *
 * Tunables (all optional):
 * - RECALL=false                disable the layer entirely
 * - RECALL_SEEDS                search hits to seed graph expansion (default 3)
 * - RECALL_CANDIDATES           concepts to read after expansion (default 6)
 * - RECALL_MIN_SCORE            top-hit confidence needed to trust literal search (default 20)
 * - RECALL_EXCERPT_CHARS        body characters per concept (default 6000)
 * - RECALL_MAX_OUTPUT_TOKENS    generation cap, REASONING TOKENS INCLUDED (default 2048)
 * - RECALL_THINKING_BUDGET      reasoning budget, honoured only by providers that
 *                               read chat_template_kwargs (vLLM); llama.cpp ignores
 *                               it and bills thinking against the cap (default 128)
 *
 * On excerpt size: concepts average a few KB, and the answer frequently sits
 * past the first screen. Trimming to a short excerpt makes the model report
 * "not enough" on material it actually holds — and an unnecessary deep run
 * costs far more than the extra prompt tokens. Widening the excerpts barely
 * moves latency, because prefill is cheap and cached while generation is not.
 *
 * On the output cap: llama.cpp counts thinking tokens against max_tokens, so
 * the cap has to hold the reasoning trace as well as the answer. Measured on
 * the live endpoint (llama-swap / llama.cpp 0.3.0, model lfm2.5-8b-a1b) with a
 * realistic ~9.4k-token recall prompt: the reasoning trace alone is ~3300
 * characters (~700 tokens) in every variant, so a cap of 900 ended with
 * finish_reason "length" at exactly 900 completion tokens — an answer cut off
 * mid-sentence — while 4000 ended with "stop" at 978 tokens in the same ~10 s
 * wall clock. The thinking knobs (chat_template_kwargs, enable_thinking,
 * reasoning_effort) are all inert there, so widening the cap is the only lever
 * that works; 2048 leaves the answer room without letting a runaway generation
 * run long. Do not lower it back to 900: a truncated string used to be returned
 * as a success, cached for 24 h, and the deep agent was never reached.
 */

const DEFAULT_SEEDS = 3;
const DEFAULT_CANDIDATES = 6;
const DEFAULT_MIN_SCORE = 20;
const DEFAULT_EXCERPT_CHARS = 6000;
const DEFAULT_MAX_OUTPUT_TOKENS = 2048;
const DEFAULT_THINKING_BUDGET = 128;

export interface RecallOutcome {
  /** The answer, or null when this layer declined and the caller must escalate. */
  answer: string | null;
  /**
   * Concepts retrieval put in front of the model. Useful even on a decline:
   * the deep agent should start from them instead of rediscovering them.
   */
  paths: string[];
}

/**
 * Why a tool-free generation stopped, collapsed to the three cases this layer
 * can act on: a complete answer, an answer cut off by the output cap, or
 * anything else (content filter, error, unknown).
 */
export type RecallFinish = "stop" | "length" | "other";

/**
 * What a tool-free generation reports back. Deliberately an object and not a
 * bare string: a fake in a test that still returns a string then throws on
 * `.text.trim()` instead of quietly asserting against nothing.
 */
export interface RecallGeneration {
  text: string;
  finishReason: RecallFinish;
}

export type RecallGenerate = (
  system: string,
  prompt: string,
  options: AgentOptions,
  /** Extra generation controls a tool-free call should honour. */
  controls: { maxOutputTokens: number; thinkingBudget: number }
) => Promise<RecallGeneration>;

function intEnv(value: string | undefined, fallback: number): number {
  const n = value === undefined ? NaN : Number.parseInt(value, 10);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

const MAX_RECALL_VARIANTS = 4;
const MAX_RECALL_ANCHORS = 2;
const MAX_RECALL_INTENTS = 3;
const EXPANSION_TOKEN_PATTERN = /[\p{L}\p{N}\p{M}\p{S}]+(?:[-._/][\p{L}\p{N}\p{M}\p{S}]+)*/gu;
const EXPANSION_PATH_SEPARATOR = /[/\.]+/u;
const EXPANSION_COMPONENT_SEPARATOR = /[/._-]+/u;

// These are deliberately generic operational intents, rather than names from
// the bundle. They let a compound question split into independently useful
// searches without inventing terms with an LLM.
const INTENT_ROOTS = new Set([
  "access",
  "artwork",
  "auth",
  "backup",
  "branch",
  "build",
  "configure",
  "create",
  "deploy",
  "export",
  "find",
  "icon",
  "import",
  "install",
  "launch",
  "logo",
  "login",
  "migrate",
  "open",
  "package",
  "remove",
  "rename",
  "restore",
  "run",
  "search",
  "setup",
  "start",
  "stop",
  "test",
  "update",
  "use",
  "version",
  "workflow",
]);
const EXPANSION_STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "be",
  "been",
  "by",
  "can",
  "could",
  "do",
  "does",
  "for",
  "from",
  "have",
  "how",
  "i",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "so",
  "the",
  "to",
  "was",
  "what",
  "where",
  "which",
  "with",
]);
const INTENT_MORPHOLOGY = new Map([
  ["artworks", "artwork"],
  ["branches", "branch"],
  ["branching", "branch"],
  ["configured", "configure"],
  ["configuring", "configure"],
  ["installations", "install"],
  ["installed", "install"],
  ["installing", "install"],
  ["icons", "icon"],
  ["launching", "launch"],
  ["logos", "logo"],
  ["migrating", "migrate"],
  ["opened", "open"],
  ["opening", "open"],
  ["packaging", "package"],
  ["restoring", "restore"],
  ["running", "run"],
  ["searching", "search"],
  ["setting", "setup"],
  ["starting", "start"],
  ["stopping", "stop"],
  ["testing", "test"],
  ["tested", "test"],
  ["tests", "test"],
  ["using", "use"],
  ["used", "use"],
  ["updated", "update"],
  ["updating", "update"],
]);

function expansionRoot(token: string): string {
  const lower = token.toLowerCase();
  return INTENT_MORPHOLOGY.get(lower) ?? lower;
}

function expansionParts(token: string): string[] {
  return token.split(EXPANSION_PATH_SEPARATOR).filter((part) => part.length > 1);
}

function expansionComponents(token: string): string[] {
  return token.split(EXPANSION_COMPONENT_SEPARATOR).filter((part) => part.length > 1);
}

/**
 * Derive a few deterministic, content-searchable views of a compound query.
 *
 * The whole question remains the authoritative first search. These variants
 * only exist when the question has both a project/entity anchor and at least
 * two recognisable operational intents, which avoids turning ordinary lookup
 * queries into a fan-out of generic searches.
 */
function recallQueryVariants(question: string): string[] {
  const tokens = question.slice(0, 4096).match(EXPANSION_TOKEN_PATTERN) ?? [];
  if (tokens.length === 0) return [];

  const anchors = new Map<string, number>();
  const intentScores = new Map<string, { score: number; order: number }>();
  let nextIntentOrder = 0;
  const rememberIntent = (root: string, score: number): void => {
    const previous = intentScores.get(root);
    if (!previous) {
      intentScores.set(root, { score, order: nextIntentOrder });
      nextIntentOrder += 1;
    } else if (score > previous.score) {
      previous.score = score;
    }
  };

  for (const token of tokens) {
    const parts = expansionParts(token);
    const tokenRoot = expansionRoot(token);
    if (INTENT_ROOTS.has(tokenRoot)) rememberIntent(tokenRoot, 1);

    if (parts.length > 1) {
      const nonIntentParts = parts.filter((part) => !INTENT_ROOTS.has(expansionRoot(part)));
      if (nonIntentParts.length > 0) {
        // Prefer useful intermediate path components such as `ptr-ms` over a
        // full path or a one-word component. Search itself still decomposes
        // each of these compounds and applies its normal confidence rules.
        for (const part of parts) {
          if (!INTENT_ROOTS.has(expansionRoot(part))) {
            anchors.set(part, part.includes("-") ? 3 : 1);
          }
        }
        for (let length = parts.length - 1; length > 1; length -= 1) {
          const prefix = parts.slice(0, length).join("/");
          if (parts.slice(0, length).some((part) => !INTENT_ROOTS.has(expansionRoot(part)))) {
            anchors.set(prefix, 2);
          }
        }
        anchors.set(token, 2);
      }
    } else if (
      token.length >= 3 &&
      /^[\p{Lu}]/u.test(token) &&
      !EXPANSION_STOPWORDS.has(tokenRoot) &&
      !INTENT_ROOTS.has(tokenRoot)
    ) {
      // A proper-cased token is a useful fallback anchor for questions that
      // name a project but do not include a path-like repository token.
      anchors.set(token, 2);
    }

    for (const part of expansionComponents(token)) {
      const root = expansionRoot(part);
      if (INTENT_ROOTS.has(root)) rememberIntent(root, parts.length > 1 ? 2 : 1);
    }
  }

  const hasIntentBoundary = /(?:^|\s)(?:and|or|versus|vs)(?:$|\s)/iu.test(question);
  const hasCompoundIntents = tokens.some(
    (token) =>
      expansionComponents(token).filter((part) => INTENT_ROOTS.has(expansionRoot(part))).length >= 2
  );
  if (anchors.size === 0 || intentScores.size < 2 || (!hasIntentBoundary && !hasCompoundIntents)) {
    return [];
  }

  const rankedAnchors = [...anchors.entries()]
    .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
    .slice(0, MAX_RECALL_ANCHORS)
    .map(([anchor]) => anchor);
  const boundedIntents = [...intentScores.entries()]
    .sort((left, right) => right[1].score - left[1].score || left[1].order - right[1].order)
    .slice(0, MAX_RECALL_INTENTS)
    .map(([intent]) => intent);
  const variants: string[] = [];

  for (const intent of boundedIntents) {
    for (const anchor of rankedAnchors) {
      const variant = `${anchor} ${intent}`;
      if (!variants.includes(variant)) variants.push(variant);
      if (variants.length >= MAX_RECALL_VARIANTS) return variants;
    }
  }
  return variants;
}

function trustedVariantHit(
  hit: { confidence?: number; confidenceQualified?: boolean },
  minScore: number
): boolean {
  return (hit.confidence ?? 0) >= minScore && hit.confidenceQualified !== false;
}

const MAX_RECALL_VARIANT_HITS = 8;

type RecallCandidateEvidence = {
  path: string;
  rankContribution: number;
  confidenceTotal: number;
  bestConfidence: number;
  bestScore: number;
  originalRank: number;
  variants: Set<string>;
  intents: Set<string>;
  anchors: Set<string>;
  anchorMatches: Set<string>;
};

function variantParts(variant: string): { anchor: string; intent: string } {
  const separator = variant.lastIndexOf(" ");
  return separator < 0
    ? { anchor: variant, intent: variant }
    : { anchor: variant.slice(0, separator), intent: variant.slice(separator + 1) };
}

function candidatePriority(candidate: RecallCandidateEvidence): number {
  // Variant coverage is deliberately the strongest signal: a hit returned by
  // several complementary views should be able to displace a one-off whole
  // query seed. Anchor matches are checked against the returned metadata rather
  // than inferred from a project name, so this stays useful across bundles.
  return (
    candidate.variants.size * 1000 +
    candidate.intents.size * 100 +
    candidate.anchors.size * 20 +
    candidate.anchorMatches.size * 200 +
    candidate.confidenceTotal +
    candidate.bestConfidence * 0.5 +
    candidate.rankContribution * 10 +
    (candidate.originalRank === Number.POSITIVE_INFINITY ? 0 : 10 / candidate.originalRank) +
    candidate.bestScore * 0.01
  );
}

function addRecallCandidate(
  pool: Map<string, RecallCandidateEvidence>,
  hit: {
    path: string;
    score: number;
    confidence?: number;
    title?: string;
    description?: string;
    snippet?: string;
  },
  rank: number,
  variant?: string
): void {
  const candidate =
    pool.get(hit.path) ??
    {
      path: hit.path,
      rankContribution: 0,
      confidenceTotal: 0,
      bestConfidence: 0,
      bestScore: 0,
      originalRank: Number.POSITIVE_INFINITY,
      variants: new Set<string>(),
      intents: new Set<string>(),
      anchors: new Set<string>(),
      anchorMatches: new Set<string>(),
    } satisfies RecallCandidateEvidence;
  pool.set(hit.path, candidate);

  candidate.rankContribution += 1 / rank;
  candidate.confidenceTotal += hit.confidence ?? 0;
  candidate.bestConfidence = Math.max(candidate.bestConfidence, hit.confidence ?? 0);
  candidate.bestScore = Math.max(candidate.bestScore, hit.score);
  if (!variant) {
    candidate.originalRank = Math.min(candidate.originalRank, rank);
    return;
  }

  candidate.variants.add(variant);
  const { anchor, intent } = variantParts(variant);
  candidate.anchors.add(anchor);
  candidate.intents.add(intent);
  const searchable = [hit.path, hit.title, hit.description, hit.snippet]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  if (searchable.includes(anchor.toLowerCase())) candidate.anchorMatches.add(anchor);
}

/** Neighbour sets over the concept link graph (undirected, 1 hop). */
async function neighboursOf(kb: KnowledgeBase): Promise<Map<string, string[]>> {
  const map = new Map<string, string[]>();
  const { edges } = await kb.graph();
  for (const e of edges) {
    if (!map.has(e.source)) map.set(e.source, []);
    if (!map.has(e.target)) map.set(e.target, []);
    map.get(e.source)!.push(e.target);
    map.get(e.target)!.push(e.source);
  }
  return map;
}

export async function runRecall(
  kb: KnowledgeBase,
  question: string,
  options: AgentOptions = {},
  // Injectable for tests.
  generate: RecallGenerate = defaultGenerate
): Promise<RecallOutcome> {
  if (process.env.RECALL === "false") return { answer: null, paths: [] };

  const seeds = intEnv(process.env.RECALL_SEEDS, DEFAULT_SEEDS);
  const maxCandidates = intEnv(process.env.RECALL_CANDIDATES, DEFAULT_CANDIDATES);
  const minScore = intEnv(process.env.RECALL_MIN_SCORE, DEFAULT_MIN_SCORE);
  const excerptChars = intEnv(process.env.RECALL_EXCERPT_CHARS, DEFAULT_EXCERPT_CHARS);

  // Keep the original whole-query search first. Its top hit remains the trust
  // gate for the fast path; expansion must never turn a weak/path-only query
  // into a generation request.
  const hits = await kb.search(question, { limit: Math.max(seeds, 1) });
  if (hits.length === 0) return { answer: null, paths: [] };
  // Ranking score deliberately rewards useful path decomposition, but is not
  // calibrated as evidence: ubiquitous components can still rank a hit first.
  // Only the corpus-aware confidence field can cross this gate. In particular,
  // a path-only hit must not fall back to its ranking score and trigger recall.
  const topConfidence = hits[0].confidence ?? 0;
  if (topConfidence < minScore) return { answer: null, paths: [] };

  // Keep every bounded search result in a small evidence pool. Selecting the
  // first new hit for each variant makes generic installer/branch notes consume
  // the budget before a concept that matches the entity across both intents can
  // contribute its evidence.
  const pool = new Map<string, RecallCandidateEvidence>();
  for (const [index, hit] of hits
    .slice(0, Math.min(seeds, maxCandidates))
    .entries()) {
    addRecallCandidate(pool, hit, index + 1);
  }

  // A compound question can contain several independent intents. Search every
  // bounded anchor/intent view, but admit only hits that pass the same
  // corpus-aware confidence contract as ordinary recall candidates. Fetch more
  // than one result per view so aggregation can distinguish repeated evidence.
  if (maxCandidates > 0) {
    const variants = recallQueryVariants(question);
    const variantLimit = Math.min(MAX_RECALL_VARIANT_HITS, Math.max(maxCandidates, 2));
    for (const variant of variants) {
      const variantHits = await kb.search(variant, { limit: variantLimit });
      for (const [index, hit] of variantHits.entries()) {
        if (trustedVariantHit(hit, minScore)) {
          addRecallCandidate(pool, hit, index + 1, variant);
        }
      }
    }
  }

  const ordered = [...pool.values()]
    .sort(
      (left, right) =>
        candidatePriority(right) - candidatePriority(left) || left.path.localeCompare(right.path)
    )
    .slice(0, maxCandidates)
    .map((candidate) => candidate.path);
  const chosen = new Set(ordered);

  // Keyword search is blind to synonyms; linked concepts are the cheapest
  // way to widen the net without another LLM call.
  if (ordered.length < maxCandidates) {
    const neighbours = await neighboursOf(kb);
    for (const seed of ordered.slice(0, 2)) {
      for (const n of neighbours.get(seed) ?? []) {
        if (chosen.size >= maxCandidates) break;
        if (!chosen.has(n)) {
          chosen.add(n);
          ordered.push(n);
        }
      }
      if (chosen.size >= maxCandidates) break;
    }
  }

  const sections: string[] = [];
  const paths: string[] = [];
  for (const p of ordered) {
    try {
      const c = await kb.readConcept(p); // fresh read — never stale
      const fm = c.frontmatter;
      sections.push(
        `CONCEPT ${c.path}${fm.title ? ` — ${fm.title}` : ""}${fm.description ? ` (${fm.description})` : ""}\n` +
          c.body.slice(0, excerptChars)
      );
      paths.push(c.path);
    } catch {
      // Deleted or unreadable since the search — skip it.
    }
  }
  if (sections.length === 0) return { answer: null, paths };

  const system =
    `You answer questions using ONLY the knowledge-base excerpts provided. ` +
    `These are a few concepts retrieved from a much larger knowledge base by ` +
    `keyword search, so they may be incomplete.\n\n` +
    `Start your reply with exactly one verdict word on a line of its own: ` +
    `SUFFICIENT when the excerpts answer the question, or UNKNOWN when they do ` +
    `not contain enough to answer confidently. Give the verdict first, before any ` +
    `reasoning. If SUFFICIENT, continue with a concise, factual answer citing the ` +
    `concept paths you used on a final "Sources:" line. If UNKNOWN, write nothing ` +
    `else at all.`;
  const prompt = `CONCEPTS:\n\n${sections.join("\n\n---\n\n")}\n\nQUESTION: ${question}`;

  const maxOutputTokens = capEnv(process.env.RECALL_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS);
  const generation = await generate(system, prompt, options, {
    maxOutputTokens,
    thinkingBudget: intEnv(process.env.RECALL_THINKING_BUDGET, DEFAULT_THINKING_BUDGET),
  });
  const text = generation.text.trim();

  // Ran out of output tokens: the reply is cut off mid-sentence and reads like
  // a complete answer, so it must never be passed on as one. The deep agent's
  // retry loop is the backstop, and a null answer makes the caller write no
  // cache entry — that is the whole point of declining here. This warning is
  // unconditional and is the only signal the layer ever dropped an answer for
  // this reason: a decline writes no trace at all, because the TraceRecorder
  // built in query-cache.ts is discarded when the answer comes back null.
  if (generation.finishReason === "length") {
    console.error(
      `[understory] recall declined: the generation hit its ${maxOutputTokens}-token output cap ` +
        `(RECALL_MAX_OUTPUT_TOKENS) and would have been a truncated answer: ` +
        `"${question.slice(0, 80)}"`
    );
    return { answer: null, paths };
  }

  // The verdict leads so that declining costs a couple of tokens, not an answer
  // the caller will throw away.
  if (isMalformedAnswer(text)) {
    console.error(`[understory] recall declined: ${MALFORMED_ANSWER_MESSAGE}`);
    return { answer: null, paths };
  }
  if (/^\s*UNKNOWN\b/i.test(text)) return { answer: null, paths };
  const answer = text.replace(/^\s*SUFFICIENT\s*\n?/i, "").trim();
  if (!answer) return { answer: null, paths };
  // An unrecognised finish reason with usable text: answer it, loudly.
  // Declining here would be the mirror image of the truncation bug above.
  // "other" is also where a provider that never reports a finish reason at all
  // lands, so treating it as a failure would switch recall off entirely for
  // that deployment and quietly hand every question to the deep agent. A
  // request that really failed surfaces as a throw, not as this bucket.
  if (generation.finishReason === "other") {
    console.error(
      `[understory] recall: the generation reported an unexpected finish reason; ` +
        `answering with the text it did produce: "${question.slice(0, 80)}"`
    );
  }
  return { answer, paths };
}

const defaultGenerate: RecallGenerate = async (system, prompt, options, controls) => {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const providers: any = await import("../providers/index.js");
  let model;
  if (typeof providers.resolveModel === "function") {
    model = await providers.resolveModel((options as any).provider, options.model);
  } else {
    const cfg = providers.resolveModelConfig(process.env);
    model = await providers.createModel({
      ...(options.model ? { ...cfg, model: options.model } : cfg),
      // A bounded reasoning trace: the answer needs grounding, not a long
      // chain of thought, and decoding thinking tokens is what makes these
      // calls slow. Ignored by providers that do not support it.
      extraBody: {
        ...(providers.thinkingBudgetBody?.(controls.thinkingBudget) ?? {}),
        max_tokens: controls.maxOutputTokens,
      },
    });
  }
  const { generateText } = await import("ai");
  const result = await generateText({
    model,
    system,
    prompt,
    temperature: 0,
    // The cap on the call as well as in extraBody. What each half does, said
    // straight: providers/index.ts exports resolveModelConfig and createModel
    // and no resolveModel at all, so the feature-detected branch above is dead
    // code today; and on the branch that does run, extraBody is applied by
    // transformRequestBody after the SDK has written max_tokens, so the spread
    // in providers/index.ts wins over this option wherever both are set. The
    // option stays because it is the only cap that would reach a model built
    // without extraBody — which is precisely the branch that does not exist yet.
    maxOutputTokens: controls.maxOutputTokens,
  });
  return {
    text: result.text,
    finishReason:
      result.finishReason === "stop" || result.finishReason === "length"
        ? result.finishReason
        : "other",
  };
};
