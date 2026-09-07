import { createHash } from "node:crypto";
import { promises as fs } from "node:fs";
import type { KnowledgeBase } from "../okf/index.js";
import { parseDuration } from "../util/duration.js";
import { runQuery, type AgentOptions, type QueryResult } from "./agent.js";
import { hotLookup, recordHotQuery } from "./hot-memory.js";
import { runRecall, type RecallOutcome } from "./recall.js";
import { traceStore } from "./agent.js";
import { TraceRecorder } from "./trace.js";

export interface CachedQueryResult extends QueryResult {
  /** True when the answer came from the exact cache (no agent run, no trace). */
  cached: boolean;
  /** Which memory layer answered: exact cache, hot working set, deterministic
   * recall, or the deep agent. */
  source: "cache" | "hot" | "recall" | "deep";
}

const MAX_ENTRIES = 200;
const DEFAULT_TTL_MS = 24 * 3_600_000;

interface CacheEntry {
  expiresAt: number;
  result: QueryResult;
}

// Module-level so the cache survives the per-request McpServer instances of
// the stateless HTTP transport.
const cache = new Map<string, CacheEntry>();

/**
 * Content fingerprint of the bundle: path + mtime + size of every concept
 * file. Any write moves the fingerprint, which implicitly invalidates every
 * cached answer — no hooks into the write path needed.
 */
export async function bundleFingerprint(kb: KnowledgeBase): Promise<string> {
  const paths = await kb.bundle.listConceptPaths();
  const parts = await Promise.all(
    paths.map(async (p) => {
      try {
        const st = await fs.stat(kb.bundle.resolve(p));
        return `${p}:${st.mtimeMs}:${st.size}`;
      } catch {
        return `${p}:gone`;
      }
    })
  );
  return createHash("sha256").update(parts.join("|")).digest("hex");
}

/**
 * runQuery with a fingerprint-invalidated LRU cache (issue-adjacent: repeated
 * questions are common through MCP, and local models make every agent run
 * expensive). Disabled with QUERY_CACHE=false; TTL via QUERY_CACHE_TTL
 * (e.g. "1h", default 24h). Cache hits skip the agent entirely, so they
 * record no trace.
 */
export async function runQueryCached(
  kb: KnowledgeBase,
  question: string,
  options: AgentOptions = {},
  // Injectable for tests.
  runner: typeof runQuery = runQuery,
  hot: typeof hotLookup = hotLookup,
  recall: typeof runRecall = runRecall
): Promise<CachedQueryResult> {
  if (process.env.QUERY_CACHE === "false") {
    return { ...(await runner(kb, question, options)), cached: false, source: "deep" };
  }

  const fingerprint = await bundleFingerprint(kb);
  const key = createHash("sha256")
    .update(`${fingerprint}\n${normalize(question)}\n${options.model ?? ""}`)
    .digest("hex");

  // Layer 1: exact cache — same question, unchanged bundle.
  const hit = cache.get(key);
  if (hit && hit.expiresAt > Date.now()) {
    // Refresh recency (Map preserves insertion order — delete + set = LRU touch).
    cache.delete(key);
    cache.set(key, hit);
    return { ...hit.result, cached: true, source: "cache" };
  }

  const ttl = parseDuration(process.env.QUERY_CACHE_TTL) ?? DEFAULT_TTL_MS;

  // Layer 2: hot working set — recently written concepts + recent answers,
  // one tool-free LLM call. A confident hot answer also lands in the exact
  // cache so identical repeats become instant.
  //
  // A layer that cannot reach its model may only cost the query the attempt.
  // The deep agent is the one layer with a primary-then-fallback model chain
  // (resolveAgentModel + withFallback in agent.ts), so a connection error or a
  // non-retryable provider error thrown here used to unwind past it: the MCP
  // layer in packages/server turned it into an isError tool result, the one
  // layer that could still have answered never ran, and the query wrote no
  // trace at all. Degrade instead — and say so in the log, which is the only
  // place a swallowed layer failure is visible.
  let hotAnswer: string | null = null;
  try {
    hotAnswer = await hot(kb, question, options);
  } catch (err) {
    console.error(
      `[understory] hot memory failed, falling through to the next layer: ${errorMessage(err)}`
    );
  }
  if (hotAnswer !== null) {
    const result: QueryResult = { answer: hotAnswer, steps: 0, traceId: "" };
    store(key, result, ttl);
    return { ...result, cached: false, source: "hot" };
  }

  // Layer 3: deterministic recall — search + graph walk in code, then one
  // tool-free generation. Answers the ordinary question in a single
  // round-trip; returns null when retrieval looks too thin to trust, and then
  // the deep agent's retry loop is still the backstop. The recorder is made
  // before the call so the trace duration covers the retrieval and generation.
  //
  // A throwing recall degrades exactly like a hot miss: the deep agent still
  // gets the question. It is worth a trace, though — finalising that recorder
  // with the failed outcome is what makes a broken layer show up in the trace
  // store instead of vanishing, and the empty outcome keeps the candidate hint
  // honest, since nothing was located.
  const recorder = new TraceRecorder();
  let recalled: RecallOutcome = { answer: null, paths: [] };
  try {
    recalled = await recall(kb, question, options);
  } catch (err) {
    console.error(
      `[understory] recall failed, escalating to the deep agent: ${errorMessage(err)}`
    );
    const trace = recorder.finalize("query", question, errorMessage(err), "failed");
    await traceStore(kb).save(trace).catch(() => {
      /* a failed trace must not lose an answer */
    });
  }
  if (recalled.answer !== null) {
    // Traced like any other query, with the retrieval it did as its single
    // step — this is how a slow query is attributed to a layer later.
    recorder.record("recall", question, recalled.paths);
    const trace = recorder.finalize("query", question, recalled.answer, "success");
    await traceStore(kb).save(trace).catch(() => {
      /* a failed trace must not lose an answer */
    });
    const result: QueryResult = { answer: recalled.answer, steps: 1, traceId: trace.id };
    store(key, result, ttl);
    recordHotQuery(question, recalled.answer);
    return { ...result, cached: false, source: "recall" };
  }

  // Layer 4: deep memory — the full agent loop. Its answer feeds the hot set.
  // Anything recall already found is handed over, so a declined attempt buys
  // the deep run a head start instead of costing an extra round-trip.
  const result = await runner(kb, withCandidateHint(question, recalled.paths), options); // writes its own trace
  store(key, result, ttl);
  recordHotQuery(question, result.answer);
  return { ...result, cached: false, source: "deep" };
}

/**
 * When deterministic recall declines, the deep agent inherits what was already
 * found instead of repeating the search. Phrased as a head start, not a
 * boundary — the deep loop must still widen on its own judgement.
 */
export function withCandidateHint(question: string, paths: string[]): string {
  if (paths.length === 0) return question;
  return (
    `${question}\n\n[Retrieval hint] Keyword search and link expansion already ` +
    `located these related concepts. Read them first with one read_concepts ` +
    `call, then widen only if they do not answer the question:\n` +
    paths.map((p) => `- ${p}`).join("\n")
  );
}

function store(key: string, result: QueryResult, ttl: number): void {  cache.set(key, { expiresAt: Date.now() + ttl, result });
  while (cache.size > MAX_ENTRIES) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

/** Test hook: reset module-level cache state. */
export function clearQueryCache(): void {
  cache.clear();
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function normalize(question: string): string {
  return question.trim().toLowerCase().replace(/\s+/g, " ");
}
