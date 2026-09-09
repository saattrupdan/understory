import type { KnowledgeBase } from "../okf/index.js";
import { parseDuration } from "../util/duration.js";
import { capEnv } from "../util/env.js";
import type { AgentOptions } from "./agent.js";
import type { RecallFinish, RecallGeneration } from "./recall.js";
import { isMalformedAnswer, MALFORMED_ANSWER_MESSAGE } from "./answer-validation.js";

/**
 * Hot memory: a small working set of recently written concepts and recent
 * Q&A pairs. Queries consult it BEFORE the deep agent run — one cheap,
 * tool-free LLM call over a tiny context. Misses fall through to deep
 * memory (the full agent loop). Short-term memory in front of long-term.
 *
 * Staleness rules:
 * - Hot concepts are stored as PATHS and read fresh at lookup — never stale.
 * - Hot Q&A pairs are purged on any write (the write may contradict them).
 * - Everything expires after HOT_MEMORY_TTL (default 1h).
 *
 * Tunables (all optional):
 * - HOT_MEMORY=false                disable the layer entirely
 * - HOT_MEMORY_TTL                  how long an entry stays hot (default 1h)
 * - HOT_MEMORY_MAX_OUTPUT_TOKENS    generation cap, REASONING TOKENS INCLUDED
 *                                   (default 2048)
 *
 * On the output cap: the evidence gathered for the recall layer applies here
 * verbatim — llama.cpp bills thinking tokens against max_tokens and ignores
 * every thinking knob — and it matters more here, because this layer runs
 * first. With no cap of its own the only bound was whatever the server allows,
 * and a reply cut off mid-sentence was returned as a confident answer: it
 * short-circuited the recall layer and the deep agent both, and query-cache.ts
 * wrote the fragment into the 24 h exact cache with no trace at all. So the cap
 * is sent explicitly, and a run that hits it declines like any other miss.
 */

interface HotQA {
  question: string;
  answer: string;
  at: number;
}

const MAX_CONCEPTS = 10;
const MAX_QAS = 10;
const DEFAULT_TTL_MS = 3_600_000;
const MAX_EXCERPT_CHARS = 1500;
const DEFAULT_MAX_OUTPUT_TOKENS = 2048;

// Module-level: survives per-request McpServer instances (stateless HTTP).
const hotConcepts = new Map<string, number>(); // path → touchedAt
let hotQAs: HotQA[] = [];

/** Called by the write tools after any concept write/patch. */
export function recordHotWrite(path: string): void {
  hotConcepts.delete(path);
  hotConcepts.set(path, Date.now());
  while (hotConcepts.size > MAX_CONCEPTS) {
    const oldest = hotConcepts.keys().next().value;
    if (oldest === undefined) break;
    hotConcepts.delete(oldest);
  }
  // A write may contradict previous answers — drop them.
  hotQAs = [];
}

/** Called on deletes: the concept leaves the hot set; answers may be stale. */
export function recordHotDelete(path: string): void {
  hotConcepts.delete(path);
  hotQAs = [];
}

/** Called after a deep query completes. */
export function recordHotQuery(question: string, answer: string): void {
  hotQAs.push({ question, answer, at: Date.now() });
  if (hotQAs.length > MAX_QAS) hotQAs = hotQAs.slice(-MAX_QAS);
}

/** Test hook. */
export function clearHotMemory(): void {
  hotConcepts.clear();
  hotQAs = [];
}

/**
 * The generation seam, deliberately the same shape the recall layer uses: the
 * finish reason is the part that tells a truncated reply from a complete one,
 * and a bare string cannot carry it. The two layers share the types so their
 * handling of a truncated answer cannot drift apart.
 */
export type HotGenerate = (
  system: string,
  prompt: string,
  options: AgentOptions,
  /** Generation controls a tool-free call should honour. */
  controls: { maxOutputTokens: number }
) => Promise<RecallGeneration>;

/**
 * Try to answer from the hot set. Returns the answer, or null when hot
 * memory is empty/expired/disabled or can't answer confidently (the model
 * must reply UNKNOWN in that case, which falls through to deep memory).
 */
export async function hotLookup(
  kb: KnowledgeBase,
  question: string,
  options: AgentOptions = {},
  // Injectable for tests.
  generate: HotGenerate = defaultGenerate
): Promise<string | null> {
  if (process.env.HOT_MEMORY === "false") return null;
  const ttl = parseDuration(process.env.HOT_MEMORY_TTL) ?? DEFAULT_TTL_MS;
  const cutoff = Date.now() - ttl;

  const sections: string[] = [];

  for (const [path, touchedAt] of hotConcepts) {
    if (touchedAt < cutoff) continue;
    try {
      const c = await kb.readConcept(path); // fresh read — never stale
      const fm = c.frontmatter;
      sections.push(
        `CONCEPT ${c.path}${fm.title ? ` — ${fm.title}` : ""}${fm.description ? ` (${fm.description})` : ""}\n` +
          c.body.slice(0, MAX_EXCERPT_CHARS)
      );
    } catch {
      hotConcepts.delete(path); // deleted behind our back
    }
  }
  for (const qa of hotQAs) {
    if (qa.at < cutoff) continue;
    sections.push(`PREVIOUS Q&A\nQ: ${qa.question}\nA: ${qa.answer}`);
  }

  if (sections.length === 0) return null;

  const system =
    `You answer questions using ONLY the recent-memory excerpts provided. ` +
    `These are the most recently touched pieces of a larger knowledge base. ` +
    `If they fully and confidently answer the question, answer concisely (and ` +
    `cite concept paths when you used them). If they do NOT contain enough to ` +
    `answer confidently, reply with exactly: UNKNOWN`;
  const prompt = `RECENT MEMORY:\n\n${sections.join("\n\n---\n\n")}\n\nQUESTION: ${question}`;

  const maxOutputTokens = capEnv(
    process.env.HOT_MEMORY_MAX_OUTPUT_TOKENS,
    DEFAULT_MAX_OUTPUT_TOKENS
  );
  const generation = await generate(system, prompt, options, { maxOutputTokens });
  const text = generation.text.trim();

  // Ran out of output tokens: the reply is cut off mid-sentence and reads like
  // a complete answer. Never pass one on — this layer answers before the two
  // that were built to handle exactly this, and query-cache.ts would cache the
  // fragment for 24 h. Declining is a normal miss, so no trace is written; the
  // line below is the only signal the layer dropped an answer for this reason.
  if (generation.finishReason === "length") {
    console.error(
      `[understory] hot memory declined: the generation hit its ${maxOutputTokens}-token output cap ` +
        `(HOT_MEMORY_MAX_OUTPUT_TOKENS) and would have been a truncated answer: ` +
        `"${question.slice(0, 80)}"`
    );
    return null;
  }

  if (!text) return null;
  if (isMalformedAnswer(text)) {
    console.error(`[understory] hot memory declined: ${MALFORMED_ANSWER_MESSAGE}`);
    return null;
  }
  if (/^UNKNOWN\b/i.test(text)) return null;
  // An unrecognised finish reason with usable text: answer it, loudly. "other"
  // is also where a provider that never reports a finish reason at all lands,
  // so declining here would switch the layer off for that deployment; a request
  // that really failed surfaces as a throw, not as this bucket. Same rule as
  // the recall layer, so the two layers behave alike from the caller's seat.
  if (generation.finishReason === "other") {
    console.error(
      `[understory] hot memory: the generation reported an unexpected finish reason; ` +
        `answering with the text it did produce: "${question.slice(0, 80)}"`
    );
  }
  return text;
}

/**
 * One tool-free generation, capped like the recall layer's. Provider access is
 * feature-detected so this file works with both the current provider API
 * (resolveModel — not exported by providers/index.ts today, so that branch is
 * dead code) and the generic-slots API (resolveModelConfig/createModel).
 */
const defaultGenerate: HotGenerate = async (system, prompt, options, controls) => {
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const providers: any = await import("../providers/index.js");
  let model;
  if (typeof providers.resolveModel === "function") {
    model = await providers.resolveModel((options as any).provider, options.model);
  } else {
    const cfg = providers.resolveModelConfig(process.env);
    model = await providers.createModel({
      ...(options.model ? { ...cfg, model: options.model } : cfg),
      // Sent as extraBody exactly as recall sends it: transformRequestBody
      // applies it after the SDK writes max_tokens, so this wins over both the
      // call-level option below and LLM_MAX_OUTPUT_TOKENS in the environment —
      // HOT_MEMORY_MAX_OUTPUT_TOKENS is the knob that moves it. No thinking
      // budget is set here; the layer has no such knob.
      extraBody: { max_tokens: controls.maxOutputTokens },
    });
  }
  const { generateText } = await import("ai");
  const result = await generateText({
    model,
    system,
    prompt,
    temperature: 0,
    // The only cap that would reach a model built without extraBody, i.e. the
    // branch above that does not exist yet.
    maxOutputTokens: controls.maxOutputTokens,
  });
  const finishReason: RecallFinish =
    result.finishReason === "stop" || result.finishReason === "length"
      ? result.finishReason
      : "other";
  return { text: result.text, finishReason };
};
