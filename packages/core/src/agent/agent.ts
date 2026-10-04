import {
  generateText,
  generateObject,
  wrapLanguageModel,
  modelMessageSchema,
  streamText,
  stepCountIs,
  type LanguageModel,
  type ModelMessage,
} from "ai";
import type { KnowledgeBase } from "../okf/index.js";
import { z } from "zod";
import {
  createModel,
  resolveFallbackConfig,
  resolveModelConfig,
  type ModelConfig,
} from "../providers/index.js";
import { withFallback } from "../providers/fallback.js";
import {
  buildQuerySynthesisPrompt,
  buildSystemPrompt,
} from "./system-prompt.js";
import { buildReadTools, buildWriteTools, formatTree } from "./tools.js";
import {
  assertInputWithinLimit,
  resolveAgentLimits,
} from "./limits.js";
import { AgentRunContext, fitText } from "./run-context.js";
import { isAbortError, throwIfAborted } from "../util/abort.js";
import { sha256 } from "../util/hash.js";
import { TraceRecorder, TraceStore, type TraceTiming, type TraceUsage } from "./trace.js";
import {
  createProtocolLeakageGuard,
  isMalformedAnswer,
  isUnsafeSynthesisAnswer,
  MALFORMED_ANSWER_MESSAGE,
} from "./answer-validation.js";

export interface AgentOptions {
  model?: string;
  signal?: AbortSignal;
  /** Raw caller-provided facts, excluding server-added mutation instructions. */
  preflightInput?: string;
  /** Caller-suggested owner; it must be read and independently checked. */
  ownerHint?: string;
  /** Only memory_add may append the caller's exact fact without a generated edit. */
  directAdd?: boolean;
}

export interface QueryResult {
  answer: string;
  steps: number;
  traceId: string;
}

export interface MutationResult {
  summary: string;
  filesChanged: string[];
  steps: number;
  traceId: string;
}

export type MutationOutcome =
  | { ok: true; result: MutationResult }
  | { ok: false; status: "partial"; filesChanged: string[]; error: string; traceId: string }
  | { ok: false; status: "failed"; error: string };

interface ResolvedAgentModel {
  /** Model used by the tool loop; it may transparently fail over on transport errors. */
  model: LanguageModel;
  /** Tool-free synthesis repair model. Never wrap this in a second fallback chain. */
  synthesisModel: LanguageModel;
  /** Raw configured fallback for one bounded mutation-loop retry. */
  mutationRetryModel?: LanguageModel;
  modelChain: string[];
}

async function promptContext(
  kb: KnowledgeBase,
  mode: "query" | "mutate" | "chat",
  state: AgentRunContext
) {
  const [types, tree] = await Promise.all([kb.listTypes(), kb.listTree()]);
  return {
    existingTypes: state.systemTypes(types),
    treeSummary: state.systemTree(formatTree(tree, 0, false)),
    mode,
  };
}

export async function resolveAgentModel(
  options: AgentOptions,
  mode: "query" | "mutate" | "chat",
  env: NodeJS.ProcessEnv = process.env,
  onProviderCall?: (call: NonNullable<TraceTiming["providerCalls"]>[number]) => void
): Promise<ResolvedAgentModel> {
  const scopedConfig = (config: ModelConfig | null | undefined): ModelConfig | null | undefined => {
    const disableThinking =
      (mode === "query" && env.QUERY_ENABLE_THINKING === "false") ||
      (mode === "mutate" && env.MUTATION_ENABLE_THINKING === "false");
    if (!disableThinking || !config) return config;
    const extraBody = config.extraBody ?? {};
    const templateKwargs = extraBody.chat_template_kwargs;
    return {
      ...config,
      extraBody: {
        ...extraBody,
        chat_template_kwargs: {
          ...(templateKwargs && typeof templateKwargs === "object" ? templateKwargs : {}),
          enable_thinking: false,
        },
      },
    };
  };
  const primaryConfig = scopedConfig(
    withModelOverride(resolveModelConfig(env), options.model)
  )!;
  throwIfAborted(options.signal);
  const primaryModel = await createModel(primaryConfig, options.signal);
  const primary = onProviderCall
    ? withProviderTiming(primaryModel, modelLabel(primaryConfig), onProviderCall)
    : primaryModel;
  const fallbackConfig = scopedConfig(resolveFallbackConfig(env));

  if (!fallbackConfig) {
    return {
      model: primary,
      synthesisModel: primary,
      modelChain: [modelLabel(primaryConfig)],
    };
  }

  const allowFor = resolveAllowFor(env.LLM_FALLBACK_ALLOW_FOR);
  if (allowFor && !allowFor.has(mode)) {
    return {
      model: primary,
      synthesisModel: primary,
      modelChain: [modelLabel(primaryConfig)],
    };
  }

  const fallbackModel = await createModel(fallbackConfig, options.signal);
  const fallback = onProviderCall
    ? withProviderTiming(fallbackModel, modelLabel(fallbackConfig), onProviderCall)
    : fallbackModel;
  return {
    // The initial loop keeps the existing transport-only fallback behaviour.
    model: withFallback(primary, fallback, {
      retry429: env.LLM_FALLBACK_RETRY_429 === "true",
    }),
    // Queries retain synthesis repair; mutations may retry the complete tool loop once.
    synthesisModel: mode === "query" ? fallback : primary,
    mutationRetryModel: mode === "mutate" ? fallback : undefined,
    modelChain: [modelLabel(primaryConfig), modelLabel(fallbackConfig)],
  };
}

function resolveAllowFor(raw: string | undefined): Set<string> | null {
  if (!raw || raw === "*") return null;
  return new Set(raw.split(",").map((s) => s.trim()).filter(Boolean));
}

function withModelOverride(config: ModelConfig, model: string | undefined): ModelConfig {
  return model ? { ...config, model } : config;
}

// No baseURL here by design: traces persist under <bundle>/.traces/, and a
// published bundle would otherwise leak internal hostnames/IPs/ports.
function modelLabel(config: ModelConfig): string {
  return `${config.format}:${config.model || "auto"}`;
}

export function traceStore(kb: KnowledgeBase): TraceStore {
  return new TraceStore(kb.bundle.root);
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function chatFailureMessage(message: string, filesChanged: Set<string>): string {
  const files = [...filesChanged].sort();
  if (files.length === 0) return message;
  return `${message}\n\n⚠ Partial mutation: ${files.length} file(s) changed before failure.\nFiles changed:\n${files.map((file) => `- ${file}`).join("\n")}`;
}

/** Disable tools on the final allowed generation so it can only synthesise. */
export function prepareFinalSynthesisStep(maxSteps: number) {
  return ({ stepNumber }: { stepNumber: number }) =>
    stepNumber >= maxSteps - 1 ? { activeTools: [] } : undefined;
}

function assertSynthesised(steps: ReadonlyArray<{ toolCalls?: unknown[] }>): void {
  const last = steps.at(-1);
  if (last?.toolCalls && last.toolCalls.length > 0) {
    throw new Error("Agent reached the step limit before producing a final answer");
  }
}

/** Sum token usage across the run's steps (issue #15). Undefined when the provider reports none. */
function sumStepsUsage(
  steps: ReadonlyArray<{ usage?: { inputTokens?: number; outputTokens?: number } }>
): TraceUsage | undefined {
  let inputTokens = 0;
  let outputTokens = 0;
  let reported = false;
  for (const step of steps) {
    const u = step.usage;
    if (!u || (u.inputTokens == null && u.outputTokens == null)) continue;
    reported = true;
    inputTokens += u.inputTokens ?? 0;
    outputTokens += u.outputTokens ?? 0;
  }
  return reported ? { inputTokens, outputTokens } : undefined;
}

const READ_ONLY_TOOL_NAMES = new Set([
  "search_knowledge",
  "read_concept",
  "read_concepts",
  "list_directory",
  "lint_knowledge",
]);
const MAX_REPAIR_EVIDENCE_CHARS = 16_000;
const MAX_REPAIR_EVIDENCE_ITEMS = 64;
const MAX_REPAIR_EVIDENCE_VALUE_DEPTH = 5;
const MAX_REPAIR_EVIDENCE_VALUE_CHARS = 8_000;

function isReadOnlyToolName(value: unknown): value is string {
  return typeof value === "string" && READ_ONLY_TOOL_NAMES.has(value);
}

function recordSafeTranscriptMessage(value: unknown): ModelMessage | undefined {
  if (!value || typeof value !== "object") return undefined;
  const message = value as { role?: unknown; content?: unknown };
  if (!Array.isArray(message.content)) return undefined;

  if (message.role === "assistant") {
    const parts = message.content
      .filter(
        (part): part is Record<string, unknown> =>
          !!part && typeof part === "object" && part.type === "tool-call"
      )
      .filter(
        (part) =>
          typeof part.toolCallId === "string" &&
          isReadOnlyToolName(part.toolName) &&
          Object.prototype.hasOwnProperty.call(part, "input")
      )
      .map((part) => ({
        type: "tool-call" as const,
        toolCallId: part.toolCallId as string,
        toolName: part.toolName as string,
        input: part.input,
      }));
    if (parts.length === 0) return undefined;
    const parsed = modelMessageSchema.safeParse({ role: "assistant", content: parts });
    return parsed.success ? parsed.data : undefined;
  }

  if (message.role === "tool") {
    const parts = message.content
      .filter(
        (part): part is Record<string, unknown> =>
          !!part && typeof part === "object" && part.type === "tool-result"
      )
      .filter(
        (part) =>
          typeof part.toolCallId === "string" &&
          isReadOnlyToolName(part.toolName) &&
          isToolResultOutput(part.output)
      )
      .map((part) => ({
        type: "tool-result" as const,
        toolCallId: part.toolCallId as string,
        toolName: part.toolName as string,
        output: part.output,
      }));
    if (parts.length === 0) return undefined;
    const parsed = modelMessageSchema.safeParse({ role: "tool", content: parts });
    return parsed.success ? parsed.data : undefined;
  }
  return undefined;
}

function isToolResultOutput(
  value: unknown
): value is { type: string; value: unknown } {
  if (!value || typeof value !== "object") return false;
  const output = value as { type?: unknown; value?: unknown };
  return (
    (output.type === "text" ||
      output.type === "json" ||
      output.type === "error-text" ||
      output.type === "error-json" ||
      output.type === "content") &&
    Object.prototype.hasOwnProperty.call(output, "value")
  );
}

/**
 * Keep only the tool transcript from a completed deep run. AI SDK v5's
 * `result.response.messages` also contains the final assistant text; feeding
 * that text back would re-inject the malformed answer into the repair prompt.
 */
function safeRepairMessages(
  question: string,
  steps: ReadonlyArray<Record<string, unknown>>,
  responseMessages: unknown[] | undefined
): ModelMessage[] | undefined {
  const transcript: ModelMessage[] = [];
  const collect = (messages: unknown[] | undefined): boolean => {
    if (!messages) return true;
    for (const message of messages) {
      const safe = recordSafeTranscriptMessage(message);
      if (safe) {
        transcript.push(safe);
      } else if (
        message &&
        typeof message === "object" &&
        (message as { role?: unknown }).role === "tool"
      ) {
        // A response tool message is only usable as a whole. Silently dropping
        // a raw/invalid result would leave a misleading, incomplete transcript.
        return false;
      }
    }
    return true;
  };
  // AI SDK v5 exposes the combined response transcript on the result. It is
  // already ordered and must be preferred: each StepResult.response.messages is
  // a cumulative clone, so collecting every step would duplicate tool calls.
  if (!collect(responseMessages)) transcript.length = 0;
  if (transcript.length === 0) {
    const lastStep = steps.at(-1);
    const response = lastStep?.response;
    const lastStepMessages =
      response && typeof response === "object"
        ? (response as { messages?: unknown }).messages
        : undefined;
    if (Array.isArray(lastStepMessages)) {
      transcript.length = 0;
      if (!collect(lastStepMessages)) transcript.length = 0;
    }
  }

  // Test seams and older provider adapters may omit response messages.
  // Reconstruct the same v5 assistant/tool message shape from the successful
  // tool calls/results, never from step.text.
  if (transcript.length === 0) {
    for (const step of steps) {
      const calls = Array.isArray(step.toolCalls)
        ? step.toolCalls
        : Array.isArray(step.staticToolCalls)
          ? step.staticToolCalls
          : Array.isArray(step.dynamicToolCalls)
            ? step.dynamicToolCalls
            : [];
      const results = Array.isArray(step.toolResults)
        ? step.toolResults
        : Array.isArray(step.staticToolResults)
          ? step.staticToolResults
          : Array.isArray(step.dynamicToolResults)
            ? step.dynamicToolResults
            : [];
      const assistant = recordSafeTranscriptMessage({ role: "assistant", content: calls });
      const tool = recordSafeTranscriptMessage({
        role: "tool",
        content: results.map((result) => {
          if (!result || typeof result !== "object") return result;
          const part = result as Record<string, unknown>;
          return {
            ...part,
            // StepResult tool results expose the raw tool output. Prompt
            // messages require the v5 LanguageModelV2ToolResultOutput union.
            output: { type: "json", value: part.output },
          };
        }),
      });
      if (assistant) transcript.push(assistant);
      if (tool) transcript.push(tool);
    }
  }

  return transcript.length > 0 ? [{ role: "user", content: question }, ...transcript] : undefined;
}

interface RepairEvidence {
  tool: string;
  value: unknown;
}

/**
 * Turn successful read results into data-only evidence for a retry. In
 * particular, never pass assistant messages, tool-call ids, or write-tool
 * results to the model: KAT treats those protocol-shaped messages as a cue to
 * emit another XML call.
 */
function safeRepairEvidence(
  steps: ReadonlyArray<Record<string, unknown>>,
  responseMessages: unknown[] | undefined,
  maxChars: number
): string | undefined {
  const evidence: RepairEvidence[] = [];

  const collectMessages = (messages: unknown[] | undefined): void => {
    if (!messages) return;
    for (const message of messages) {
      if (!message || typeof message !== "object") continue;
      const record = message as { role?: unknown; content?: unknown };
      if (record.role !== "tool" || !Array.isArray(record.content)) continue;
      for (const part of record.content) {
        if (!part || typeof part !== "object") continue;
        const result = part as {
          type?: unknown;
          toolName?: unknown;
          output?: unknown;
        };
        if (
          result.type !== "tool-result" ||
          !isReadOnlyToolName(result.toolName) ||
          !isToolResultOutput(result.output)
        ) {
          continue;
        }
        evidence.push({ tool: result.toolName, value: result.output.value });
      }
    }
  };

  // Prefer the combined response transcript, which is not duplicated per step.
  collectMessages(responseMessages);
  if (evidence.length === 0) {
    for (const step of steps) {
      const results = Array.isArray(step.toolResults)
        ? step.toolResults
        : Array.isArray(step.staticToolResults)
          ? step.staticToolResults
          : Array.isArray(step.dynamicToolResults)
            ? step.dynamicToolResults
            : [];
      for (const result of results) {
        if (!result || typeof result !== "object") continue;
        const record = result as Record<string, unknown>;
        if (!isReadOnlyToolName(record.toolName)) continue;
        evidence.push({ tool: record.toolName, value: record.output });
      }
    }
  }
  if (evidence.length === 0 || maxChars <= 0) return undefined;

  const lines = evidence
    .slice(0, MAX_REPAIR_EVIDENCE_ITEMS)
    .map((entry, index) => {
      const value = sanitiseRepairValue(entry.value);
      return `[${index + 1}] ${entry.tool}: ${JSON.stringify(value)}`;
    });
  return fitText(lines.join("\n"), maxChars, "\n...[read-only evidence truncated]");
}

function sanitiseRepairValue(
  value: unknown,
  depth = 0,
  seen = new WeakSet<object>()
): unknown {
  if (typeof value === "string") {
    return sanitiseRepairText(value.slice(0, MAX_REPAIR_EVIDENCE_VALUE_CHARS));
  }
  if (value === null || typeof value === "boolean" || typeof value === "number") {
    return value;
  }
  if (depth >= MAX_REPAIR_EVIDENCE_VALUE_DEPTH) return "[nested value omitted]";
  if (typeof value !== "object") return `[${typeof value} value omitted]`;
  if (seen.has(value)) return "[cyclic value omitted]";
  seen.add(value);
  if (Array.isArray(value)) {
    return value.slice(0, MAX_REPAIR_EVIDENCE_ITEMS).map((item) =>
      sanitiseRepairValue(item, depth + 1, seen)
    );
  }
  const result: Record<string, unknown> = {};
  for (const key of Object.keys(value).slice(0, MAX_REPAIR_EVIDENCE_ITEMS)) {
    result[sanitiseRepairText(key, 500)] = sanitiseRepairValue(
      (value as Record<string, unknown>)[key],
      depth + 1,
      seen
    );
  }
  return result;
}

function sanitiseRepairText(value: string, maxChars = MAX_REPAIR_EVIDENCE_VALUE_CHARS): string {
  return value
    .slice(0, maxChars)
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/<\|(?:tool_call_start|tool_call_end)\|>/gi, "[protocol marker removed]")
    .replace(/<\/?(?:tool_call|function(?:=[^>]+)?)>/gi, "[protocol marker removed]");
}

/** Read-only Q&A over the bundle. */
function withProviderTiming(
  model: LanguageModel,
  label: string,
  record: (call: NonNullable<TraceTiming["providerCalls"]>[number]) => void
): LanguageModel {
  return wrapLanguageModel({
    model: model as Extract<LanguageModel, { doGenerate: unknown }>,
    middleware: {
      wrapGenerate: async ({ doGenerate }) => {
        const started = Date.now();
        try {
          const result = await doGenerate();
          const usage = result.usage;
          try {
            record({
              model: label,
              durationMs: Date.now() - started,
              ...(usage.inputTokens === undefined ? {} : { inputTokens: usage.inputTokens }),
              ...(usage.outputTokens === undefined ? {} : { outputTokens: usage.outputTokens }),
            });
          } catch { /* telemetry must never affect model behavior */ }
          return result;
        } catch (error) {
          try { record({ model: label, durationMs: Date.now() - started }); }
          catch { /* telemetry must never mask the provider error */ }
          throw error;
        }
      },
    },
  });
}

export async function runQuery(
  kb: KnowledgeBase,
  question: string,
  options: AgentOptions = {}
): Promise<QueryResult> {
  const limits = resolveAgentLimits();
  assertInputWithinLimit(
    question,
    limits.maxInputChars,
    "Query input"
  );
  const state = new AgentRunContext(limits, options.signal);
  throwIfAborted(options.signal);
  const promptStarted = Date.now();
  const ctx = await promptContext(kb, "query", state);
  const promptContextMs = Date.now() - promptStarted;
  throwIfAborted(options.signal);
  const recorder = new TraceRecorder();
  const generateTextCalls: NonNullable<TraceTiming["generateTextCalls"]> = [];
  const providerCalls: NonNullable<TraceTiming["providerCalls"]> = [];
  const generationStarted = Date.now();
  const maxSteps = limits.maxSteps;
  let modelChain: string[] = [];
  try {
    const resolved = await resolveAgentModel(options, "query", process.env, (call) => providerCalls.push(call));
    modelChain = resolved.modelChain;
    const callStarted = Date.now();
    const result = await generateText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      prompt: question,
      tools: buildReadTools(kb, recorder, state),
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
      abortSignal: options.signal,
    });
    generateTextCalls.push({ model: resolved.modelChain.join(" → "), durationMs: Date.now() - callStarted, ...sumStepsUsage(result.steps) });
    throwIfAborted(options.signal);
    assertSynthesised(result.steps);

    let finalText = result.text;
    const allSteps: Array<{
      usage?: { inputTokens?: number; outputTokens?: number };
    }> = [...result.steps];
    if (isMalformedAnswer(finalText) || isUnsafeSynthesisAnswer(finalText)) {
      console.error(`[understory] query answer rejected: ${MALFORMED_ANSWER_MESSAGE}`);
      const repairMessages = safeRepairMessages(
        question,
        result.steps as unknown as ReadonlyArray<Record<string, unknown>>,
        result.response?.messages
      );
      if (!repairMessages) {
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
      const repairStarted = Date.now();
      const repair = await generateText({
        model: resolved.synthesisModel,
        system: buildQuerySynthesisPrompt(),
        messages: repairMessages,
        tools: {},
        abortSignal: options.signal,
      });
      generateTextCalls.push({ model: resolved.modelChain.at(-1) ?? "configured", durationMs: Date.now() - repairStarted, ...sumStepsUsage(repair.steps) });
      throwIfAborted(options.signal);
      assertSynthesised(repair.steps);
      if (isMalformedAnswer(repair.text) || isUnsafeSynthesisAnswer(repair.text)) {
        // KAT can interpret the valid assistant/tool transcript above as a
        // request to continue the tool protocol. Give it one final chance,
        // but only with bounded, quoted data from read-only tool results.
        const evidence = safeRepairEvidence(
          result.steps as unknown as ReadonlyArray<Record<string, unknown>>,
          result.response?.messages,
          Math.min(MAX_REPAIR_EVIDENCE_CHARS, limits.maxToolResultChars, limits.maxInputChars)
        );
        if (!evidence) {
          throw new Error(MALFORMED_ANSWER_MESSAGE);
        }
        const secondRepairStarted = Date.now();
        const secondRepair = await generateText({
          model: resolved.synthesisModel,
          system: buildQuerySynthesisPrompt(),
          messages: [
            {
              role: "user",
              content:
                `Original question:\n${question}\n\n` +
                "BEGIN UNTRUSTED READ-ONLY EVIDENCE\n" +
                evidence +
                "\nEND UNTRUSTED READ-ONLY EVIDENCE",
            },
          ],
          tools: {},
          abortSignal: options.signal,
        });
        generateTextCalls.push({ model: resolved.modelChain.at(-1) ?? "configured", durationMs: Date.now() - secondRepairStarted, ...sumStepsUsage(secondRepair.steps) });
        throwIfAborted(options.signal);
        assertSynthesised(secondRepair.steps);
        if (
          isMalformedAnswer(secondRepair.text) ||
          isUnsafeSynthesisAnswer(secondRepair.text)
        ) {
          throw new Error(MALFORMED_ANSWER_MESSAGE);
        }
        finalText = secondRepair.text;
        allSteps.push(...repair.steps, ...secondRepair.steps);
      } else {
        finalText = repair.text;
        allSteps.push(...repair.steps);
      }
    }

    const trace = recorder.finalize(
      "query",
      question,
      finalText,
      "success",
      modelChain,
      sumStepsUsage(allSteps),
      undefined,
      undefined,
      { generateTextCalls, providerCalls, promptContextMs, generationMs: Date.now() - generationStarted }
    );
    await traceStore(kb).save(trace).catch(() => {
      /* telemetry persistence must not change query behavior */
    });
    return { answer: finalText, steps: allSteps.length, traceId: trace.id };
  } catch (err) {
    const trace = recorder.finalize("query", question, errorMessage(err), "failed", modelChain);
    await traceStore(kb).save(trace).catch(() => {
      /* telemetry persistence must not mask the original query failure */
    });
    throw err;
  }
}

// The staged path sends complete owner bodies, never a truncated page. The
// model's context accommodates these bounds; larger evidence fails closed.
const STAGED_MAX_OWNER_CHARS = 64_000;
const STAGED_MAX_EVIDENCE_CHARS = 80_000;

/** Distinct words used only as a conservative support check, never as proof of meaning. */
function stagedWords(text: string): string[] {
  const stop = new Set(["the", "and", "for", "with", "that", "this", "from", "into", "about", "record", "remember", "concept", "knowledge", "policy", "distinct", "unrelated", "existing", "only"]);
  const compounds = text.toLowerCase().match(/[\p{L}\p{N}][\p{L}\p{N}_-]{2,}/gu) ?? [];
  // Entity names such as "Memory-Async" must match the "async" component
  // of a concept slug, without losing exact compound matches elsewhere.
  return [...new Set(compounds.flatMap((word) => [word, ...word.split(/[-_]/)])
    .filter((word) => word.length >= 3 && !stop.has(word)))];
}

function stagedOverlap(text: string, evidence: string): number {
  const words = stagedWords(text);
  const supported = new Set(stagedWords(evidence));
  return words.length ? words.filter((word) => supported.has(word)).length / words.length : 0;
}

function stagedChangedClauses(before: string, after: string): number {
  // A model can approve a correct change to one assertion while missing an
  // unrelated change to another. Preserve all other sentence/semicolon clauses.
  const split = (text: string) => text.split(/;\s*|(?<=[.!?])\s+|\n+/).map((part) => part.trim()).filter(Boolean);
  const oldClauses = split(before);
  const newClauses = split(after);
  if (oldClauses.length !== newClauses.length) return Infinity;
  return oldClauses.filter((clause, index) => clause !== newClauses[index]).length;
}

function stagedCorrectionIntent(input: string): boolean {
  // Ordinary add payloads often discuss a dataset "update" or say "rather
  // than" without asking to change an existing assertion. Treat only an
  // explicit leading edit instruction as a correction for memory_add.
  return /^(?:(?:please|remember|record)\s+)?(?:correct|correction|update|replace|amend|supersede|retract)\b/i.test(input.trim());
}

function stagedCorrection(input: string): boolean {
  return /\b(?:correct|correction|update|replace|supersede|instead|formerly|previously|no longer|only after|now|rather than|don['’]t|doesn['’]t)\b/i.test(input);
}

function stagedOwnerAnchor(input: string, conceptPath: string): boolean {
  const slug = conceptPath.split("/").pop()?.replace(/\.md$/, "").replace(/[-_]/g, " ") ?? "";
  const words = new Set(stagedWords(input));
  return stagedWords(slug).some((word) => words.has(word));
}

function stagedTitleMention(input: string, title: string): boolean {
  return title.length >= 4 && input.toLowerCase().includes(title.toLowerCase()) &&
    !new RegExp(`\\b(?:unrelated to|rather than changing|do not change)\\s+(?:the\\s+)?${title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`, "i").test(input);
}

/** Knowledge add/update — full toolset, low temperature. */
async function runStagedMutation(
  kb: KnowledgeBase,
  instruction: string,
  options: AgentOptions,
  state: AgentRunContext,
  recorder: TraceRecorder,
): Promise<MutationOutcome> {
  const started = Date.now();
  const providerCalls: NonNullable<TraceTiming["providerCalls"]> = [];
  let modelChain: string[] = [];
  let changed: string[] = [];
  try {
    const raw = options.preflightInput ?? instruction;
    const correction = options.directAdd ? stagedCorrectionIntent(raw) : stagedCorrection(raw);
    if (options.ownerHint && (options.ownerHint.length > 512 ||
        !/^\/[a-z0-9_/-]+\.md$/.test(options.ownerHint) || options.ownerHint.split("/").includes(".."))) {
      throw new Error("Staged mutation deferred: owner hint is not a bounded concept path.");
    }
    const query = mutationSearchQuery(raw);
    if (!query) throw new Error("Staged mutation deferred: no bounded search terms.");
    state.checkCancellation();
    const hits = (await kb.search(query)).slice(0, 5);
    recorder.record("search_knowledge", query, hits.map((hit) => hit.path));
    state.checkCancellation();
    const first = hits[0];
    const second = hits[1];
    // A caller-supplied path is a locator, not a search-ranking requirement.
    // Read it only when it already exists, then subject it to the same complete
    // evidence, independent consistency and stale-body checks as any other owner.
    // A missing hint may still be the desired path of a distinct new concept.
    const quotedOwner = !options.directAdd ? raw.match(/`(\/[a-z0-9_/-]+\.md)`/i)?.[1] : undefined;
    const ownerPath = options.ownerHint ?? quotedOwner;
    const hintedPath = ownerPath && /^\/[a-z0-9_/-]+\.md$/.test(ownerPath) && await kb.bundle.exists(ownerPath)
      ? ownerPath : undefined;
    const named = hits.filter((hit) => hit.title && stagedTitleMention(raw, hit.title));
    const explicitOwner = named.length === 1 ? named[0] : undefined;
    const dominant = first && first.confidenceQualified === true && (!second ||
      (first.confidence ?? 0) >= (second.confidence ?? 0) + 20 &&
      (first.confidence ?? 0) >= (second.confidence ?? 0) * 1.5);
    const candidates: Array<{ path: string; frontmatter: { title?: string; type?: string }; body: string }> = [];
    if (hintedPath || explicitOwner || dominant) {
      const concept = await kb.readConcept(hintedPath ?? (explicitOwner ?? first).path);
      recorder.record("read_concept", concept.path, [concept.path]);
      if (concept.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: owner body exceeds evidence limit.");
      candidates.push({ path: concept.path, frontmatter: concept.frontmatter, body: concept.body });
    } else if (hits.length > 0) {
      // Give the proposer complete bodies for up to three competing owners.
      // Its choice is checked again against the alternative titles below.
      for (const hit of hits.slice(0, 3)) {
        const concept = await kb.readConcept(hit.path);
        recorder.record("read_concept", concept.path, [concept.path]);
        if (concept.body.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: candidate body exceeds evidence limit.");
        candidates.push({ path: concept.path, frontmatter: concept.frontmatter, body: concept.body });
      }
    }
    const evidence = JSON.stringify(candidates);
    if (evidence.length + raw.length > STAGED_MAX_EVIDENCE_CHARS) {
      throw new Error("Staged mutation deferred: evidence exceeds prompt budget.");
    }
    state.checkCancellation();
    const resolved = await resolveAgentModel(options, "mutate", process.env, (call) => providerCalls.push(call));
    modelChain = resolved.modelChain;
    // An explicit correction supplies its own exact old/new text. Avoid a
    // generated proposal, but still check the full owner, support, and CAS.
    const quotedCorrection = !options.directAdd && options.preflightInput
      ? raw.match(/\breplace\s+(?:the\s+)?(?:exact\s+)?(?:phrase|text|claim)?\s*`([^`]{12,2000})`\s+with\s+`([^`]{1,2000})`/i)
      : null;
    if (quotedCorrection) {
      const [, oldText, newText] = quotedCorrection;
      const matching = candidates.filter((candidate) => candidate.body.split(oldText).length === 2);
      if (matching.length === 1) {
        const target = matching[0];
        if (oldText === newText || stagedChangedClauses(oldText, newText) !== 1 ||
            stagedWords(newText).some((word) => !stagedWords(oldText).includes(word) && !stagedWords(raw).includes(word))) {
          throw new Error("Staged mutation rejected: quoted replacement changes unsupported claims.");
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent safety check. Does the caller's exact quoted replacement faithfully apply the instruction to this observed owner, preserve every unrelated claim, and avoid a contradiction? A quoted path alone is not proof of ownership. If uncertain, safe=false.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nOLD TEXT:\n${oldText}\nNEW TEXT:\n${newText}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: quoted correction failed consistency check.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        const replacement = target.body.replace(oldText, newText);
        if (replacement.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: replacement exceeds evidence limit.");
        state.checkCancellation();
        await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: exact quoted correction", sha256(target.body), options.signal, (path) => { changed = [path]; });
        const summary = `Staged mutation changed ${changed.join(", ")}.`;
        const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
        await traceStore(kb).save(trace);
        return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
      }
    }
    // For memory_add, preserve the caller's exact fact rather than generating
    // replacement prose. A fast read-only selector handles unhinted candidates;
    // one thinking-model check must still independently approve the chosen owner
    // and consistency. memory_update keeps the conservative edit proposal below.
    let directPath = hintedPath;
    if (!directPath && options.directAdd && options.preflightInput && !correction &&
        candidates.length > 0 && raw.trim().length >= 30 && raw.trim().length <= 4_000) {
      const selector = await resolveAgentModel(options, "mutate", { ...process.env, MUTATION_ENABLE_THINKING: "false" }, (call) => providerCalls.push(call));
      const selection = await generateObject({
        model: selector.model,
        schema: z.object({ path: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Choose the ONE existing concept whose topic is directly about the FACT. Use only an exact candidate path. If the fact is unrelated to all candidates, or ownership is truly ambiguous, return an empty path. A mere mention or negative reference is not ownership. Do not create or change any content. JSON only.\nFACT:\n${raw}\nCANDIDATES:\n${evidence}`,
      });
      state.checkCancellation();
      if (candidates.some((candidate) => candidate.path === selection.object.path)) directPath = selection.object.path;
    }
    if (directPath && options.directAdd && options.preflightInput && !correction &&
        raw.trim().length >= 30 && raw.trim().length <= 4_000) {
      const target = candidates.find((candidate) => candidate.path === directPath)!;
      const fact = raw.trim();
      if (!stagedOwnerAnchor(fact, target.path) ||
          (target.frontmatter.title && fact.toLowerCase().includes(target.frontmatter.title.toLowerCase()) &&
            !stagedTitleMention(fact, target.frontmatter.title))) {
        throw new Error("Staged mutation deferred: suggested owner lacks a positive entity anchor.");
      }
      if (!target.body.includes(fact)) {
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ safe: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Independent safety check for appending the caller's EXACT fact, without rewriting it. Is every assertion supported by the caller, truly about this owner rather than a more specific alternative, not already recorded, and consistent with the full existing body? The suggested path is not proof. If uncertain, safe=false.\nFACT:\n${fact}\nOWNER ${target.path} (${target.frontmatter.title ?? "untitled"}):\n${target.body}\nOTHER SEARCH HITS (titles and paths only):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}`,
        });
        state.checkCancellation();
        if (!judgement.object.safe) throw new Error("Staged mutation deferred: exact-fact append failed consistency check.");
        const appended = `${target.body.trimEnd()}\n\n${fact}\n`;
        if (appended.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: appended body exceeds evidence limit.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
        state.checkCancellation();
        await kb.patchConcept(target.path, { replaceBody: appended }, "Staged mutation: append exact verified fact", sha256(target.body), options.signal, (path) => { changed = [path]; });
      } else {
        // A substring may be quoted only to retract it later in the body. Do
        // not equate textual presence with a currently supported assertion.
        const following = target.body.slice(target.body.indexOf(fact) + fact.length, target.body.indexOf(fact) + fact.length + 160);
        if (/^\s*(?:[,;:-]\s*)?(?:but|however|although|no longer|not|never)\b/i.test(following)) {
          throw new Error("Staged mutation deferred: matching text is immediately qualified or negated.");
        }
        const judgement = await generateObject({
          model: resolved.model,
          schema: z.object({ supported: z.boolean() }).strict(),
          temperature: 0,
          abortSignal: options.signal,
          prompt: `Does the full existing body affirm the caller's EXACT fact as presently true, without a later correction, negation, or contradictory qualification? If the words are quoted only to retract them, supported=false. If uncertain, supported=false. Return JSON only.\nFACT:\n${fact}\nOWNER ${target.path}:\n${target.body}`,
        });
        state.checkCancellation();
        if (!judgement.object.supported) throw new Error("Staged mutation deferred: matching text does not establish the requested fact.");
        const current = await kb.readConcept(target.path);
        if (current.body !== target.body || !current.body.includes(fact)) throw new Error("Staged mutation rejected: no-op evidence changed before verification.");
      }
      const summary = changed.length ? `Staged mutation changed ${changed.join(", ")}.` : "No change needed; exact fact verified.";
      const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
    }
    const proposalSchema = z.object({
      action: z.enum(["replace", "append", "create", "noop", "defer"]),
      path: z.string(),
      old_text: z.string(),
      new_text: z.string(),
      body: z.string(),
      frontmatter: z.object({ type: z.string(), title: z.string(), description: z.string() }).strict(),
      claim: z.string(),
      reason: z.string(),
    }).strict();
    const proposal = await generateObject({
      model: resolved.model,
      schema: proposalSchema,
      temperature: 0,
      abortSignal: options.signal,
      prompt: `Propose exactly one safe knowledge-base mutation from this instruction and complete observed candidate evidence. Fill EVERY JSON field: unused strings must be empty and unused frontmatter must have empty type/title/description. For replace, supply candidate path, an exact old_text substring, and new_text that removes the old claim. For append, supply candidate path and the complete new fact in new_text only if genuinely non-conflicting. For create, supply a new path, nonempty body and frontmatter type/title/description; never create a duplicate owner or an unrelated backlink. For noop, quote a nonempty exact substring of an observed body in claim that establishes the requested knowledge. Otherwise defer and explain why. Never invent evidence. A suggested path is not proof of ownership; defer if it does not fit.\nINSTRUCTION:\n${raw}\nSUGGESTED EXISTING OWNER:\n${options.ownerHint ?? "none"}\nCANDIDATES:\n${evidence}`,
    });
    state.checkCancellation();
    const p = proposal.object;
    if (p.action === "defer") throw new Error(`Staged mutation deferred: ${p.reason || "proposal is not safely actionable"}`);
    if (p.action === "noop") {
      const claim = p.claim.trim();
      const requestedClaim = raw.trim().replace(/^(?:remember|record|persist)\s+(?:that\s+)?/i, "").trim();
      const normalise = (text: string) => text.replace(/[.!?]+$/, "").replace(/\s+/g, " ").trim().toLowerCase();
      const target = candidates.find((candidate) => candidate.path === p.path);
      if (!claim || claim.length < 20 || correction || !target ||
          normalise(requestedClaim) !== normalise(claim) || !target.body.includes(claim)) {
        throw new Error("Staged mutation rejected: no-op claim was not verified against the request.");
      }
      state.checkCancellation();
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body || !current.body.includes(claim)) {
        throw new Error("Staged mutation rejected: no-op evidence changed before verification.");
      }
      const trace = recorder.finalize("mutation", instruction, "Verified exact no-op.", "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: true, result: { summary: "No change needed; exact claim verified.", filesChanged: [], steps: 1, traceId: trace.id } };
    }
    if (p.action === "append") {
      const target = candidates.find((c) => c.path === p.path);
      const title = target?.frontmatter.title;
      const titleNegated = !!title && raw.toLowerCase().includes(title.toLowerCase()) && !stagedTitleMention(raw, title);
      if (!target || !title || titleNegated ||
          !(stagedTitleMention(raw, title) || hintedPath === target.path || stagedOwnerAnchor(raw, target.path)) || correction ||
          p.new_text.length < 20 || p.new_text.length > 4_000 ||
          stagedOverlap(p.new_text, raw) < 0.8 || stagedOverlap(raw, p.new_text) < 0.6 ||
          target.body.toLowerCase().includes(p.new_text.trim().toLowerCase())) {
        throw new Error("Staged mutation deferred: append owner or factual support is uncertain.");
      }
      // The proposer may overlook a contradiction. A separate, tool-free
      // judgement sees the unchanged owner and must explicitly approve it.
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent safety check. Is the proposed addition fully supported by the instruction, truly about this exact owner rather than a better-matching alternative, not already present, and consistent with every existing assertion? A suggested owner path is not proof. Answer safe=false on uncertainty, changed values, or an unrelated backlink.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nALTERNATIVE SEARCH HITS (path and title only):\n${JSON.stringify(hits.filter((hit) => hit.path !== target.path).map((hit) => ({ path: hit.path, title: hit.title })))}\nPROPOSED ADDITION:\n${p.new_text}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: append failed independent consistency check.");
      const appended = `${target.body.trimEnd()}\n\n${p.new_text.trim()}\n`;
      if (appended.length > STAGED_MAX_OWNER_CHARS) throw new Error("Staged mutation deferred: appended body exceeds evidence limit.");
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      state.checkCancellation();
      await kb.patchConcept(target.path, { replaceBody: appended }, "Staged mutation: append verified fact", sha256(target.body), options.signal, (path) => { changed = [path]; });
    } else if (p.action === "replace") {
      const target = candidates.find((c) => c.path === p.path);
      if (!target || p.old_text.length < 12 || !p.new_text || target.body.split(p.old_text).length !== 2 ||
          stagedOverlap(p.new_text, `${raw} ${p.old_text}`) < 0.65 ||
          stagedChangedClauses(p.old_text, p.new_text) !== 1) throw new Error("Staged mutation rejected: replacement is not one supported local change.");
      const oldWords = new Set(stagedWords(p.old_text));
      const requestedWords = new Set(stagedWords(raw));
      if (stagedWords(p.new_text).some((word) => !oldWords.has(word) && !requestedWords.has(word))) {
        throw new Error("Staged mutation rejected: replacement introduces facts absent from the request.");
      }
      const addedNegations = ["not", "never", "without", "cannot", "doesn't", "don't"].filter((word) =>
        stagedWords(p.new_text).includes(word) && !stagedWords(p.old_text).includes(word));
      if (addedNegations.some((word) => !stagedWords(raw).includes(word))) {
        throw new Error("Staged mutation rejected: an added negation was not requested.");
      }
      const replacement = target.body.replace(p.old_text, p.new_text);
      if (replacement === target.body || replacement.length > STAGED_MAX_OWNER_CHARS) {
        throw new Error("Staged mutation rejected: replacement is empty or oversized.");
      }
      const judgement = await generateObject({
        model: resolved.model,
        schema: z.object({ safe: z.boolean(), reason: z.string() }).strict(),
        temperature: 0,
        abortSignal: options.signal,
        prompt: `Independent safety check. Does the exact proposed replacement faithfully apply the instruction to the observed owner, remove the superseded claim, preserve unrelated claims, and avoid a new contradiction? If uncertain, safe=false.\nINSTRUCTION:\n${raw}\nOWNER ${target.path}:\n${target.body}\nOLD TEXT:\n${p.old_text}\nNEW TEXT:\n${p.new_text}`,
      });
      state.checkCancellation();
      if (!judgement.object.safe) throw new Error("Staged mutation deferred: replacement failed independent consistency check.");
      const current = await kb.readConcept(target.path);
      if (current.body !== target.body) throw new Error(`Concept changed while it was being read: ${target.path}`);
      await kb.patchConcept(target.path, { replaceBody: replacement }, "Staged mutation: replace verified claim", sha256(target.body), options.signal, (path) => { changed = [path]; });
    } else if (p.action === "create") {
      const explicitlyDistinct = /\b(?:distinct|standalone|stand-alone|new concept|new policy|unrelated)\b/i.test(raw);
      if ((candidates.length > 0 && !explicitlyDistinct) || !p.path || !p.body || !p.frontmatter.type || !p.frontmatter.title ||
          !/^\/[a-z0-9_/-]+\.md$/.test(p.path) || p.path.split("/").includes("..") || p.body.length > (state.maxDocumentChars ?? 12_000) ||
          stagedOverlap(p.frontmatter.title, raw) < 0.7 || stagedOverlap(p.body, raw) < 0.65 ||
          stagedOverlap(raw, p.body) < 0.8 ||
          !(raw.includes(p.body.trim()) || p.body.includes(raw.trim()))) {
        throw new Error("Staged mutation rejected: create is not distinct or valid.");
      }
      const mainClaim = p.body.trim().split(/[.!?]\s+/)[0]?.trim() ?? "";
      if (mainClaim.length >= 20 && candidates.some((candidate) => candidate.body.includes(mainClaim))) {
        throw new Error("Staged mutation deferred: requested claim already belongs to an observed concept.");
      }
      const similar = await kb.search(p.frontmatter.title);
      state.checkCancellation();
      if (similar.some((hit) => hit.title && stagedOverlap(p.frontmatter.title, hit.title) >= 0.75 && hit.confidenceQualified)) {
        throw new Error("Staged mutation deferred: a similarly named concept already exists.");
      }
      state.checkCancellation();
      await kb.createConcept(p.path, p.frontmatter, p.body, "Staged mutation: create distinct concept", options.signal, (path) => { changed = [path]; });
    }
    if (!changed.length) throw new Error("Staged mutation rejected: no verified write occurred.");
    const summary = `Staged mutation changed ${changed.join(", ")}.`;
    const trace = recorder.finalize("mutation", instruction, summary, "success", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
    await traceStore(kb).save(trace);
    return { ok: true, result: { summary, filesChanged: changed, steps: 1, traceId: trace.id } };
  } catch (err) {
    const message = errorMessage(err);
    if (changed.length) {
      const trace = recorder.finalize("mutation", instruction, `Partial mutation: ${changed.length} file(s) changed before failure. Error: ${message}`, "partial", modelChain, undefined, undefined, undefined, { providerCalls, generationMs: Date.now() - started });
      await traceStore(kb).save(trace);
      return { ok: false, status: "partial", filesChanged: changed, error: message, traceId: trace.id };
    }
    if (isAbortError(err, options.signal)) throw err;
    return { ok: false, status: "failed", error: message };
  }
}

function mutationSearchQuery(input: string): string {
  const words = input
    .replace(/https?:\/\/\S+/gi, " ")
    .toLowerCase()
    .match(/[\p{L}\p{N}][\p{L}\p{N}_-]{2,}/gu) ?? [];
  const stop = new Set(["the", "and", "for", "with", "that", "this", "into", "from", "about", "existing", "concept", "knowledge", "base", "update", "change", "please", "persist", "following"]);
  return [...new Set(words.filter((word) => !stop.has(word)))].slice(0, 10).join(" ").slice(0, 500);
}

export async function runMutation(
  kb: KnowledgeBase,
  instruction: string,
  options: AgentOptions = {}
): Promise<MutationOutcome> {
  const limits = resolveAgentLimits();
  assertInputWithinLimit(
    instruction,
    limits.maxInputChars,
    "Mutation input"
  );
  const state = new AgentRunContext(limits, options.signal);
  throwIfAborted(options.signal);
  const ctx = await promptContext(kb, "mutate", state);
  throwIfAborted(options.signal);
  const recorder = new TraceRecorder();
  if (process.env.MUTATION_STAGED === "true") {
    return runStagedMutation(kb, instruction, options, state, recorder);
  }
  let preflightHint = "";
  if (process.env.MUTATION_PREFLIGHT === "true") {
    try {
      const raw = options.preflightInput ?? instruction;
      const query = mutationSearchQuery(raw);
      if (query) {
        state.checkCancellation();
        const hits = await kb.search(query);
        recorder.record("search_knowledge", query, hits.slice(0, 5).map((hit) => hit.path));
        state.checkCancellation();
        // Only a unique or clearly dominant content-backed hit may seed the
        // whole-body write guard. Related but weaker hits remain available to
        // the agent through its normal search tools; close matches stay ambiguous.
        const candidates = hits.filter((hit) => hit.confidenceQualified === true);
        const first = candidates[0];
        const second = candidates[1];
        const clearOwner = first && (!second ||
          ((first.confidence ?? 0) >= (second.confidence ?? 0) + 20 &&
            (first.confidence ?? 0) >= (second.confidence ?? 0) * 1.5));
        if (clearOwner) {
          const concept = await kb.readConcept(first.path);
          state.checkCancellation();
          recorder.record("read_concept", concept.path, [concept.path]);
          const page = {
            path: concept.path,
            frontmatter: concept.frontmatter,
            frontmatter_truncated: false,
            body: concept.body,
            offset: 0,
            total_chars: concept.body.length,
            truncated: false,
            next_offset: null,
          };
          const evidence = JSON.stringify([page]);
          const heading = `\n\nDETERMINISTIC PREFLIGHT (one confidence-qualified candidate; complete body follows):\n- ${concept.path}\n`;
          // Never cut the evidence: only mark a body as read if its complete,
          // unchanged contents are actually present in the prompt.
          if (concept.body.length <= (state.maxDocumentChars ?? Number.POSITIVE_INFINITY) &&
              heading.length + evidence.length <= Math.min(12_000, state.payloadBudget ?? 12_000)) {
            preflightHint = heading + evidence;
            state.recordBodyPage(concept.path, 0, concept.body, concept.body, concept.body);
          }
        }
      }
    } catch {
      // Preflight is advisory. The regular agent search/read path remains authoritative.
    }
    throwIfAborted(options.signal);
  }
  const maxSteps = limits.maxSteps;
  const filesChanged = new Set<string>();
  const providerCalls: NonNullable<TraceTiming["providerCalls"]> = [];
  const generationStarted = Date.now();
  let modelChain: string[] = [];
  try {
    const resolved = await resolveAgentModel(
      options,
      "mutate",
      process.env,
      (call) => providerCalls.push(call)
    );
    modelChain = resolved.modelChain;
    const mutationRequest = (model: LanguageModel) => generateText({
      model,
      system: buildSystemPrompt(ctx),
      prompt: instruction + preflightHint,
      tools: {
        ...buildReadTools(kb, recorder, state),
        ...buildWriteTools(kb, filesChanged, recorder, state),
      },
      stopWhen: stepCountIs(maxSteps),
      prepareStep: prepareFinalSynthesisStep(maxSteps),
      temperature: 0.2,
      abortSignal: options.signal,
    });
    let result = await mutationRequest(resolved.model);
    throwIfAborted(options.signal);
    assertSynthesised(result.steps);
    const allSteps: Array<{
      usage?: { inputTokens?: number; outputTokens?: number };
    }> = [...result.steps];
    if (isMalformedAnswer(result.text) || isUnsafeSynthesisAnswer(result.text)) {
      console.error(`[understory] mutation summary rejected: ${MALFORMED_ANSWER_MESSAGE}`);
      // A malformed summary cannot establish which writes in a multi-write
      // instruction completed. Preserve those writes as partial; never replay.
      if (filesChanged.size > 0 || !resolved.mutationRetryModel) {
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
      result = await mutationRequest(resolved.mutationRetryModel);
      throwIfAborted(options.signal);
      assertSynthesised(result.steps);
      allSteps.push(...result.steps);
      if (isMalformedAnswer(result.text) || isUnsafeSynthesisAnswer(result.text)) {
        throw new Error(MALFORMED_ANSWER_MESSAGE);
      }
    }
    const summary = result.text;
    const trace = recorder.finalize(
      "mutation",
      instruction,
      summary,
      "success",
      modelChain,
      sumStepsUsage(allSteps),
      undefined,
      undefined,
      { providerCalls, generationMs: Date.now() - generationStarted }
    );
    await traceStore(kb).save(trace);
    return {
      ok: true,
      result: {
        summary,
        filesChanged: [...filesChanged].sort(),
        steps: allSteps.length,
        traceId: trace.id,
      },
    };
  } catch (err) {
    const files = [...filesChanged].sort();
    const message = errorMessage(err);
    // A cancelled mutation with no writes must remain a cancellation so MCP and
    // callers do not mistake it for a completed agent response. Once a write has
    // landed, retain the existing partial-mutation report instead.
    if (files.length === 0 && isAbortError(err, options.signal)) {
      const trace = recorder.finalize("mutation", instruction, message, "failed", modelChain, undefined, undefined, undefined, {
        providerCalls,
        generationMs: Date.now() - generationStarted,
      });
      await traceStore(kb).save(trace).catch(() => {
        // Preserve the provider cancellation even if trace persistence also fails.
      });
      throw err;
    }
    if (files.length > 0) {
      const summary = `Partial mutation: ${files.length} file(s) changed before failure. Error: ${message}`;
      const trace = recorder.finalize("mutation", instruction, summary, "partial", modelChain, undefined, undefined, undefined, {
        providerCalls,
        generationMs: Date.now() - generationStarted,
      });
      await traceStore(kb).save(trace);
      return { ok: false, status: "partial", filesChanged: files, error: message, traceId: trace.id };
    }
    const trace = recorder.finalize("mutation", instruction, message, "failed", modelChain, undefined, undefined, undefined, {
      providerCalls,
      generationMs: Date.now() - generationStarted,
    });
    await traceStore(kb).save(trace);
    return { ok: false, status: "failed", error: message };
  }
}

/** Interactive chat — full toolset, streaming. Caller converts to a UI stream response. */
export async function streamChat(
  kb: KnowledgeBase,
  messages: ModelMessage[],
  options: AgentOptions = {}
) {
  const state = AgentRunContext.unbounded(options.signal);
  throwIfAborted(options.signal);
  const ctx = await promptContext(kb, "chat", state);
  throwIfAborted(options.signal);
  const recorder = new TraceRecorder();
  const filesChanged = new Set<string>();
  let modelChain: string[] = [];
  // The user turn that started this run, for the trace record.
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const input =
    typeof lastUser?.content === "string"
      ? lastUser.content
      : lastUser?.content
          ?.map((part) => (part.type === "text" ? part.text : ""))
          .join(" ")
          .trim() ?? "(chat)";
  let traceFinalised = false;
  const finaliseChatTrace = async (
    answer: string,
    outcome: "success" | "partial" | "failed",
    usage?: TraceUsage
  ): Promise<void> => {
    // AI SDK callbacks can race: onError may arrive while onFinish is still
    // persisting. Claim the trace before awaiting the filesystem write.
    if (traceFinalised) return;
    traceFinalised = true;
    if (outcome === "success" && recorder.steps.length === 0) return;
    try {
      await traceStore(kb).save(
        recorder.finalize("chat", input, answer, outcome, modelChain, usage)
      );
    } catch (traceError) {
      // A trace must never turn a provider failure into an unobserved promise
      // rejection or hide the original stream error.
      console.error(`[understory] chat trace save failed: ${errorMessage(traceError)}`);
    }
  };

  try {
    const resolved = await resolveAgentModel(options, "chat");
    modelChain = resolved.modelChain;
    const protocolGuard = createProtocolLeakageGuard({
      errorMessage: () => chatFailureMessage(MALFORMED_ANSWER_MESSAGE, filesChanged),
    });
    const result = streamText({
      model: resolved.model,
      system: buildSystemPrompt(ctx),
      messages,
      tools: {
        ...buildReadTools(kb, recorder, state),
        ...buildWriteTools(kb, filesChanged, recorder, state),
      },
      // An empty condition list lets AI SDK v5 continue until the model stops,
      // without imposing an application-level step budget.
      stopWhen: [],
      abortSignal: options.signal,
      // AI SDK v5 applies this transform before toUIMessageStreamResponse(),
      // so malformed text is dropped before it can reach the HTTP client.
      experimental_transform: protocolGuard.transform,
      onFinish: async ({ text, totalUsage, steps }) => {
        const usage =
          totalUsage && (totalUsage.inputTokens != null || totalUsage.outputTokens != null)
            ? {
                inputTokens: totalUsage.inputTokens ?? 0,
                outputTokens: totalUsage.outputTokens ?? 0,
              }
            : undefined;
        try {
          assertSynthesised(steps);
          if (protocolGuard.wasMalformed() || isMalformedAnswer(text)) {
            const message = chatFailureMessage(MALFORMED_ANSWER_MESSAGE, filesChanged);
            console.error(`[understory] chat answer rejected: ${MALFORMED_ANSWER_MESSAGE}`);
            // The guard has already placed a client-visible AI SDK error part
            // before the terminal events. Do not throw here: an onFinish throw
            // makes AI SDK discard that response stream before the caller sees
            // the error, while still leaving the failed/partial trace intact.
            await finaliseChatTrace(message, filesChanged.size > 0 ? "partial" : "failed", usage);
            return;
          }
          await finaliseChatTrace(text, "success", usage);
        } catch (error) {
          const outcome = filesChanged.size > 0 ? "partial" : "failed";
          await finaliseChatTrace(chatFailureMessage(errorMessage(error), filesChanged), outcome, usage);
          throw error;
        }
      },
      onError: async ({ error }) => {
        const outcome = filesChanged.size > 0 ? "partial" : "failed";
        await finaliseChatTrace(chatFailureMessage(errorMessage(error), filesChanged), outcome);
      },
      onAbort: async () => {
        const outcome = filesChanged.size > 0 ? "partial" : "failed";
        await finaliseChatTrace(chatFailureMessage("Chat stream aborted", filesChanged), outcome);
      },
    });
    return { result, filesChanged };
  } catch (err) {
    const outcome = filesChanged.size > 0 ? "partial" : "failed";
    await finaliseChatTrace(chatFailureMessage(errorMessage(err), filesChanged), outcome);
    throw err;
  }
}
