import type { KnowledgeBase } from "../okf/index.js";
import type { AgentOptions } from "./agent.js";

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
 * - RECALL_MIN_SCORE            top-hit score needed to trust literal search (default 20)
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

  const hits = await kb.search(question, { limit: Math.max(seeds, 1) });
  if (hits.length === 0) return { answer: null, paths: [] };
  // A thin literal match is exactly the case the deep agent exists for.
  if ((hits[0].score ?? 0) < minScore) return { answer: null, paths: [] };

  const ordered = hits.slice(0, seeds).map((h) => h.path);
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

  const maxOutputTokens = intEnv(process.env.RECALL_MAX_OUTPUT_TOKENS, DEFAULT_MAX_OUTPUT_TOKENS);
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
    console.warn(
      `[recall] declined: the generation hit its ${maxOutputTokens}-token output cap ` +
        `(RECALL_MAX_OUTPUT_TOKENS) and would have been a truncated answer: ` +
        `"${question.slice(0, 80)}"`
    );
    return { answer: null, paths };
  }

  // The verdict leads so that declining costs a couple of tokens, not an answer
  // the caller will throw away.
  if (/^\s*UNKNOWN\b/i.test(text)) return { answer: null, paths };
  const answer = text.replace(/^\s*SUFFICIENT\s*\n?/i, "").trim();
  if (!answer) return { answer: null, paths };
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
    // The cap on the call itself as well as in extraBody: the resolveModel
    // branch above hands back a model built without our extraBody, so the
    // spread in providers/index.ts would not reach it at all.
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
